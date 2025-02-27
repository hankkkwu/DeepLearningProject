# RAFT optical flow algorithm
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from collections import OrderedDict
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from yolov4.tf import YOLOv4    # Run a YOLOv4-tiny Object Detection algorithm to identify individual objects
import tensorflow as tf
import time

from torchvision.transforms import Compose
import dpt_package.util.io
from dpt_package.dpt.models import DPTDepthModel
from dpt_package.dpt.transforms import Resize, NormalizeImage, PrepareForNet


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()    # make it [channel, height, width]
    frame = frame.unsqueeze(0)    # make it [1, channel, height, width]
    frame = frame.to(device)
    return frame

def get_cpu_model(model):
    # OrderedDict: 根據key被插入的先後順序做排列，如果更新了某個key的value值並不會影響他在OrderedDict的排序位置，
    #              除非這個key被刪除並被重新插入，才會從原始的排序位置變成最末位，因為其變成最新插入的一個key。
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def load_model(weights_path):
    model = RAFT()
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
    return model

def inference_imgs(model, frame_1, frame_2):
    """
    Do optical flow inference on 2 images
    :param model: pre-trained weight on KITTI dataset
    :param frame_1: frame at time t
    :param frame_2: frame at time t+1
    :return:
        flow_up: predicted flow map of shape [B, 2, H, W]
        flo (np.ndarray): Flow visualization image of shape [H,W,3]
    """
    # change model's mode to evaluation
    model.eval()
    device="cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        # Read images
        frame_1 = frame_preprocess(frame_1, device)
        frame_2 = frame_preprocess(frame_2, device)
        # preprocessing
        padder = InputPadder(frame_1.shape, mode="kitti")    # Pads images such that dimensions are divisible by 8
        frame_1, frame_2 = padder.pad(frame_1, frame_2)

        # predict the flow
        # flow_low: the flow map that is not upsampled with shape = (47, 156, 3)
        # flow_up: upsampled flow map with shape = (376, 1248, 3)
        flow_low, flow_up = model(frame_1, frame_2, iters=12, test_mode=True)   # flow_low is not used

        # transform to image
        flo = flow_up[0].permute(1,2,0).cpu().numpy()
        flo = flow_viz.flow_to_image(flo)
    return flow_up, flo

def run_obstacle_detection(img):
    """
    Do YOLOv4 object detection
    :param img: input image for
    :return:
        result: image with bounding boxes draw on it
        pred_bboxes: predicted bounding boxes (candidates, (x, y, w, h, class_id, prob))
    """
    start_time=time.time()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
    # 0 ~ 255 to 0.0 ~ 1.0
    resized_image = resized_image / 255.
    #input_data == Dim(1, input_size, input_size, channels)
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)

    _candidates = []
    result = img.copy()
    pred_bboxes = []
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
        #candidates == Dim(batch, candidates, (bbox))
        candidates = np.concatenate(_candidates, axis=1)
        #pred_bboxes == Dim(candidates, (x, y, w, h, class_id, prob))
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] #https://stackoverflow.com/questions/35673095/python-how-to-eliminate-all-the-zero-rows-from-a-matrix-in-numpy?lq=1
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        exec_time = time.time() - start_time
        #print("time: {:.2f} ms".format(exec_time * 1000))
        result = yolo.draw_bboxes(img, pred_bboxes)
    return result, pred_bboxes


def add_arrow_and_depth(result, pred_bboxes, fl_vectors, depth_map):
    """
    Evaluate the Motion of each obstacle through time,
    draw arrow on each predicted bounding box to present the motion of each object
    :param result: image with bounding boxes draw on it
    :param pred_bboxes: predicted bounding boxes (candidates, (x, y, w, h, class_id, prob))
    :param fl_vectors: predicted flow map of shape [B, 2, H, W]
    :return:
        image_arr: image with bounding boxes and arrows on it
    """
    h, w, _ = result.shape

    #For each box, add an arrow that shows the flow of each obstacle
    for bbox in pred_bboxes:
        center_x = int(bbox[0]*w)
        center_y = int(bbox[1]*h)
        width = int(bbox[2]*w)
        height = int(bbox[3]*h)

        start_point = (center_x, center_y)
        # print("start point: ", start_point)

        top_left_x = int(center_x - width * 0.5)
        top_left_y = int(center_y - height * 0.5)
        bot_right_x = int(center_x + width * 0.5)
        bot_right_y = int(center_y + height * 0.5)

        #############
        # Add arrow #
        #############
        arrow_vec_x = fl_vectors[0][0][top_left_y:bot_right_y, top_left_x:bot_right_x]
        arrow_len_x = arrow_vec_x.mean()
        arrow_vec_y = fl_vectors[0][1][top_left_y:bot_right_y, top_left_x:bot_right_x]
        arrow_len_y = arrow_vec_y.mean()
        # print("arrow length x: ", arrow_len_x)
        # print("arrow length y: ", arrow_len_y)

        # end point cannot exceed the image width and height
        end_point =(min(int(center_x + arrow_len_x), w), min(int(center_y + arrow_len_y), h))
        # print("end point:", end_point)
        # image_arr = cv2.arrowedLine(result, start_point, end_point, (0,255,0), 5)
        result = cv2.arrowedLine(result, start_point, end_point, (0,255,0), 5)

        ##################
        # Add depth info #
        ##################
        depth = depth_map[center_y][center_x]     # center of the bounding box
        result = cv2.putText(result, '{0:.2f} m'.format(depth), (int(center_x - width*0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return result


def depth_prediction(input_image, optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_image (BGR image): cv2 captured image
        model_path (str): path to saved model
    """
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load network

    # Original kitti image size = (1242, 374)
    # we want it to be (1216, 352) so it can be divided by 32
    net_w = 1216
    net_h = 352

    model = DPTDepthModel(
        path="dpt_package/weights/dpt_hybrid_kitti-cb926ef4.pt",
        scale=0.00006016,
        shift=0.00579,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # print(model)
    model.eval()

    if optimize == True and device == torch.device("cuda"):
        # for better performance
        model = model.to(memory_format=torch.channels_last)
        model = model.half()    # make it torch.float16

    model.to(device)

    # input
    # this will convert the image to RGB, and do normalization by /255
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    # util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)
    return prediction


def inference_video(video_path):
    """
    Run on a Video

    :return:
        video_frames_arrow: RGB images with bounding boxes and arrows
        video_frames_flow: flow maps
    """
    model = load_model("models/raft-kitti.pth")
    # change model's mode to evaluation
    model.eval()
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    video_frames_arrow = []
    video_frames_flow = []
    video_frames_depth = []

    # Original kitti image size = (1242, 374)
    # we want it to be (1216, 352) so it can be divided by 32
    height, width, _ = prev_frame.shape
    top = height - 352
    left = (width - 1216) // 2
    prev_frame = prev_frame[top : top + 352, left : left + 1216, :]

    # with torch.no_grad():
    while True:
        # read the next frame
        ret, cur_frame = cap.read()
        if not ret:
            break
        # Original kitti image size = (1242, 374)
        # we want it to be (1216, 352) so it can be divided by 32
        height, width, _ = cur_frame.shape
        top = height - 352
        left = (width - 1216) // 2
        cur_frame = cur_frame[top : top + 352, left : left + 1216, :]

        # Predict the optical flow map
        flow_up, flo = inference_imgs(model, prev_frame.copy(), cur_frame.copy())
        # Run obstacle Detection
        result, pred_bboxes = run_obstacle_detection(bgr2rgb(cur_frame.copy()))
        # Run depth prediction
        depth_map = depth_prediction(cur_frame.copy())

        # Add motion prediction and depth prediction
        image_result = bgr2rgb(add_arrow_and_depth(result, pred_bboxes, flow_up, depth_map))

        # add depth map to a list
        depth_map *= 256
        video_frames_depth.append(depth_map)

        video_frames_arrow.append(image_result)
        video_frames_flow.append(flo)
        # mode forward one frame
        prev_frame = cur_frame
    return video_frames_arrow, video_frames_flow, video_frames_depth


# Load Yolov4 object detection model
yolo = YOLOv4(tiny=True)
yolo.classes = "raft_data/coco.names"
yolo.make_model()
yolo.load_weights("raft_data/yolov4-tiny.weights", weights_type="yolo")

"""
##############################################
# RAFT Inference for 2 images (at t and t+1) #
##############################################
img_1 = cv2.imread("raft_data/0000000148.png")
img_2 = cv2.imread("raft_data/0000000149.png")
model = load_model("models/raft-kitti.pth")

flow_up, flo = inference_imgs(model, img_1, img_2)

f, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(20,10))
ax0.imshow(bgr2rgb(img_1))
ax1.imshow(bgr2rgb(img_2))
ax2.imshow(flo)
plt.show()

print(flow_up.shape)
print(flow_up[0][0])
print("u at coordinate [0,0]: ", flow_up[0][0][0][0])
print("v at coordinate [0,0]: ", flow_up[0][1][0][0])
print(flow_up[0][0].shape)


#####################################
# RAFT + YOLOv4 Inference for image #
#####################################
result, pred_bboxes = run_obstacle_detection(bgr2rgb(img_1))
input_image = np.copy(result)
image_arr = add_arrow_to_box(input_image, pred_bboxes, flow_up)
plt.imshow(image_arr)
plt.show()

_, (ax0, ax1)= plt.subplots(1, 2, figsize=(20,10))
ax0.imshow(image_arr)
ax1.imshow(flo)
plt.show()

print(flo.shape)
print(image_arr.shape)
"""

#####################################
# RAFT + YOLOv4 Inference for video #
#####################################
video_frames_arrow, video_frames_flow, video_frames_depth = inference_video("raft_data/kitti_1.mp4")

# out = cv2.VideoWriter("outputs/output_flow3.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (video_frames_flow[0].shape[1] ,video_frames_flow[0].shape[0]))
# for i in range(len(video_frames_flow)):
#     out.write(video_frames_flow[i][:,:,::-1].astype(np.uint8))
# out.release()

out = cv2.VideoWriter("outputs/output1.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (video_frames_arrow[0].shape[1] ,video_frames_arrow[0].shape[0]))
for i in range(len(video_frames_arrow)):
    out.write(video_frames_arrow[i].astype(np.uint8))
out.release()
