# inference_ros2
A ROS 2 based generic inference module

# Installation
## Preparing the model files
TensorRT uses `.engine` files to achieve its fast model performance. These files only work on the GPU architecture that they were created on. This means `.engine` files that were e.g. created on a RTX 4090 cannot be used on the robot, which uses an RTX 2060. 

Get the trained model file (yolov8-keypoint-det-cropweed.onnx) from ? and put it in the `model` folder.
The following command will mount the `.onnx` model file into the TensorRT docker container and its included `trtexec` cli tool will convert the `.onnx` file into the platform dependent `.engine.` file. By included the `--user` flag the saved model will have the correct owner and not be owned by `root`.

```bash
cd model

docker run --gpus all \
    --user $(id -u):$(id -g) \
    -v .:/models \
    nvcr.io/nvidia/tensorrt:23.10-py3 \
    trtexec --onnx=/models/yolov8-keypoint-det-cropweed.onnx \
    --saveEngine=/models/yolov8-keypoint-det-cropweed-nuc-fp32-23.10.engine \
    --memPoolSize=workspace:5000

chmod +x yolov8-keypoint-det-cropweed-nuc-fp32-23.10.engine 
```

Fetch dependencies before building package
```bash 
rosdep install --from-paths src --ignore-src -r -y
```

```bash
mkdir ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/LALWeco/inference_ros2.git
git clone https://github.com/LALWeco/vision_msgs.git
cd ~/ros2_ws
# source ros2
source /opt/ros/foxy/setup.bash
colcon build --packages-select inference_ros2 vision_msgs
source install/setup.bash
ros2 run inference_ros2 instance_detector   # Instance segmentation
ros2 run inference_ros2 keypoint_detector   # Keypoint detection
```

# ToDo
- [x] Define a ros2 package for Instance segmentation and Keypoint detection node.
- [x] Create a custom message definition for Keypoint-based detection.
    - [x] Bridge from ROS1 to ROS2. 
- [x] Import ONNX model and publisher of messages.
- [ ] Import TensorRT model and infer with varying bit depth. 
- [ ] Add a visualization flag for debugging.
  - [ ] A subscriber node for visualizing messages for Instance segmentation and Keypoint detection.
