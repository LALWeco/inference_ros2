# inference_ros2
A ROS 2 based generic inference module for LALWeco. The package contains nodes for 2D keypoint detection, motion estimation and tracking, and 2D to 3D estimation.

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
```

# Available Nodes

## Keypoint Detector Node
Performs 2D keypoint detection using TensorRT for inference.
- Input: Camera images
- Output: Keypoint detections with bounding boxes

## Motion Tracking Node
Provides motion estimation and tracking using ByteTrack.
- Input: Camera images and keypoint detections
- Output: Tracked keypoints with motion compensation
- Features: Track history visualization, motion-compensated tracking

## Point Estimator Node
Performs 2D to 3D estimation for detected keypoints.

# FAQs

## How to run keypoint detection only?
```bash
ros2 launch inference_ros2 inference.launch.py
```
This will start the keypoint detector node, which publishes to:
- `/inference/Keypoint2DDetArray`: Keypoint detections
- `/inference/detection_image`: Visualization with detections

## How to run keypoint detection with motion tracking?
```bash
ros2 launch inference_ros2 tracking.launch.py
```
This launches both the keypoint detector and motion tracking nodes. Additional topics:
- `/tracking/tracked_keypoints`: Tracked keypoints with IDs
- `/tracking/visualization`: Visualization with tracks and track history

## How to visualize the results?
Use Foxglove to visualize the detection and tracking results. See the [Foxglove setup](docs/foxglove.md) for details.
Available visualization topics:
- `/inference/detection_image`: Shows raw detections
- `/tracking/visualization`: Shows tracks with history and motion vectors

## How to generate the TensorRT engine? 
See the [TensorRT Model generation](docs/model.md)

## Development container
See the [Docker build instructions](docs/docker.md)

## TensorRT engine naming convention
See the [TensorRT engine naming convention](model/model.md)

# ToDo
- [x] Integrate the ByteTrack tracker to the keypoint detector
- [x] Test visualization with detections, tracks and keypoints
- [ ] Add documentation for motion tracking parameters
- [ ] Implement additional track filtering options
