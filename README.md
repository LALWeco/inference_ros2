# inference_ros2
A ROS 2 based generic inference module for LALWeco. The package contains inference nodes for 2D keypoint detection and 2D to 3D estimation.

# Installation

## Docker (recommended)
See the [Docker build instructions](docs/docker.md)

## Manual
```bash
mkdir ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/LALWeco/inference_ros2.git
git clone https://github.com/ros-perception/vision_msgs.git
git clone https://github.com/LALWeco/lalweco_perception_msgs.git
git clone https://github.com/LALWeco/lalweco_laser_module_ros.git
cd ~/ros2_ws
colcon build --packages-select inference_ros2 vision_msgs lalweco_perception_msgs lalweco_laser_module_ros
source install/setup.bash
ros2 launch inference_ros2 keypoint_detector.launch.py
```
# FAQs
## How to run the keypoint detector and 2D to 3D estimation?
```bash
ros2 launch inference_ros2 keypoint_detector.launch.py
```
## How to generate the TensorRT engine? 
See the [TensorRT Model generation](docs/model.md)

## Development container
See the [Docker build instructions](docs/docker.md)

## Foxglove setup
See the [Foxglove setup](docs/foxglove.md)

## TensorRT engine naming convention
See the [TensorRT engine naming convention](model/model.md)

# ToDo
- [ ] Integrate the ByteTrack tracker to the keypoint detector.
- [ ] Test visualization with detections, tracks and keypoints.
