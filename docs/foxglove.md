# Foxglove setup
To play the ros2 bag files in mcap format, you need to install the `mcap` plugin.
# `mcap` plugin
```bash
sudo apt-get install ros-$ROS_DISTRO-rosbag2-storage-mcap
```
To visualize the ros2 bag files in foxglove, you need to install the rosbridge and launch the foxglove bridge server.
# `rosbridge` plugin
```bash
sudo apt-update && sudo apt install ros-$ROS_DISTRO-rosbridge-suite
ros2 launch rosbridge_server foxglove_bridge_launch.xml
```



	