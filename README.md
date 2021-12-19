# Deep Learning Projecet
Using [RAFT(Recurrent All Pairs Field Transforms for Optical Flow)](https://github.com/princeton-vl/RAFT) optical flow algorithm and YOLOv4 object detection algorithm to visualize object's motion, then using [DPT(Dense Prediction Transformers)](https://github.com/isl-org/DPT) to predict the object's depth.

The optical flow color coding:

![color coding](https://github.com/hankkkwu/RAFTwithYOLOv4/blob/main/color_coding.png)

Here is the result of the RAFT on video1:

![optical flow1](https://github.com/hankkkwu/RAFTwithYOLOv4/blob/main/outputs/output_flow1.gif)

Here is the result with object's depth and motion on video1:
(Green arrows are the motion of the objects)

![result1](https://github.com/hankkkwu/RAFTwithYOLOv4/blob/main/outputs/output1.gif)

Here is the result of the RAFT  on video2:

![optical flow2](https://github.com/hankkkwu/RAFTwithYOLOv4/blob/main/outputs/output_flow2.gif)

Here is the result with object's depth and motion on video2:
(Green arrows are the motion of the objects)

![result2](https://github.com/hankkkwu/RAFTwithYOLOv4/blob/main/outputs/output2.gif)

The DPT architecture:

![DPT architecture](https://github.com/hankkkwu/RAFTwithYOLOv4/blob/main/DPT.png)

Reference papers:

[Link to the RAFT paper](https://arxiv.org/abs/2003.12039)

[Link to the DPT paper](https://arxiv.org/abs/2103.13413)
