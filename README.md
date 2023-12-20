# fast-sam-labeler
This tool uses OpenCV, PyTorch, and the FastSam model to assist you in the tedious task of labeling image datasets with bounding boxes. <br>
It features a manual mode that allows you to change labels, draw, and erase boxes, and an automatic mode that draws the bounding box automatically using FastSam. <br>
If your dataset encounters issues with this mode, try adjusting the model's sensitivity using the '+' and '-' keys. <br>
I hope you enjoy it! Regards!

## Usage:
Set image_folder, output_folder, labels variables 

Keys:
 - a -> Previous image 
 - d -> Next image 
 - 0-9 -> Label selection 
 - r/m -> Change mode (auto/manual) 
 - esc/q -> Exit
 - +/- -> Increase or decrease confidence for SAM auto mode
   
Mouse:
 - left click -> Draw
 - center click -> Erase 

## Requeriments
OpenCV<br>
Pytorch (gpu support optional)<br>
Ultralytics

## Citation:
Original Fast Sam implementation from these great guys:<br>
Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, Jinqiao Wang<br>
Can be found in Arxiv:<br>
[https://arxiv.org/abs/2306.12156](https://arxiv.org/abs/2306.12156)
