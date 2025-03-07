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
cd ~/ros2_ws
colcon build --packages-select inference_ros2 vision_msgs lalweco_perception_msgs
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
Performs 2D to 3D estimation for detected keypoints using either geometric or depth-based methods.

### Geometric Estimation Method
- Estimates 3D point positions using camera intrinsics and height
- Assumes:
  * Camera is looking straight down at the ground
  * Keypoints lie on the ground plane
  * Ground plane is at z = camera_height
- Input: 2D keypoint detections
- Output: 3D points in ROS coordinate frame (X right, Y forward, Z up)
- Parameters:
  * `camera_height`: Height of camera from ground (meters)
  * `estimation_method`: Set to "geometric" to use this method

### Depth-Based Method
- Uses depth images to estimate 3D positions
- Input: 2D keypoint detections and depth images
- Output: 3D points based on actual depth measurements
- Parameters:
  * `depth_sample_size`: Size of depth sampling window
  * `estimation_method`: Set to "depth" to use this method

### Output Topics
- `/cropweed/keypoints_3d`: 3D point markers in ROS coordinate frame

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
