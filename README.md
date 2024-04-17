# inference_ros2
A ROS 2 based generic inference module

# Installation
## ROS1 Bridge
```bash
sudo apt install ros-foxy-ros1-bridge 
# we need to source both ros1 and ros2 for the bridge to work
source /opt/ros/noetic/setup.bash 
source /opt/ros/foxy/setup.bash 
# Check all the bridge topic pairs
ros2 run ros1_bridge dynamic_bridge--print-pairs 
# Run the ros1_bridge 
ros2 run ros1_bridge dynamic_bridge--bridge-alltopics
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
ros2 run inference_ros2 crop_detector
```

# ToDo
- [x] Define a basic ros2 package for crop_detector node
- [ ] Create a custom message definition for Keypoint-based detection
    - [ ] Variable keypoint detection [Interesting discussion](https://github.com/ultralytics/ultralytics/issues/5364)
    - [ ] Bridge from ROS1 to ROS2 (Optional?)
- [ ] Import ONNX model and publisher of messages

