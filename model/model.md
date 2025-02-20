# TensorRT model directory
The `model` directory contains the TensorRT engine files for the keypoint detection. The default naming convention is as follows:
```
yolov8-keypoint-det-<model_name>-<platform>-<precision>-<tensorrt_version>-<image_size>.engine
```

Where:
- `<model_name>` is the name of the model. This stays constant for lalweco models as `yolov8-keypoint-det`.
- `<platform>` is the platform for which the tensorrt engine is built. e.g. `nuc`, for Intel NUC PC.
- `<precision>` is the precision of the model. e.g. `fp32`, `fp16`, `int8`.
- `<tensorrt_version>` is the version of TensorRT used to build the model. e.g. `8.6.1`.
- `<image_size>` is the size of the image used at inference time. e.g. `800`.

