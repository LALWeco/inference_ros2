# inference_ros2
A ROS 2 based generic inference module

# Installation
## ROS1 Bridge
```bash
sudo apt install ros-foxy-ros1-bridge 
source /opt/ros/noetic/setup.bash 
source /opt/ros/foxy/setup.bash 
ros2 run ros1_bridge dynamic_bridge--print-pairs 
ros2 run ros1_bridge dynamic_bridge--bridge-alltopics
```

Fetch dependencies before building package
```bash 
rosdep install --from-paths src --ignore-src -r -y
```

# ToDo
- [x] Define a basic ros2 package for crop_detector node
- [ ] Create a custom message definition for Keypoint-based detection
    - [ ] Variable keypoint detection [Interesting discussion](https://github.com/ultralytics/ultralytics/issues/5364)
    - [ ] Bridge from ROS1 to ROS2 (Optional?)
- [ ] Import ONNX model and publisher of messages

