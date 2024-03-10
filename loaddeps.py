import torch
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses, resize_image

# Load the pre-trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Load openpose
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")