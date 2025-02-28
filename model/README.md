# Model repository

This folder should contain the model files that the inference module can load.

## TensorRT export command

```bash
cd src/inference_ros2/model
trtexec --onnx=yolov8-keypoint-det-cropweed.onnx --saveEngine=yolov8-keypoint-det-cropweed-nuc-fp32-23.10.engine --allowGPUFallback --memPoolSize=workspace:5000
```
