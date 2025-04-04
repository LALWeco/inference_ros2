#!/bin/bash

# Source ROS setup files
source /opt/ros/humble/setup.bash
source /home/docker/ros2_ws/install/setup.bash

# Export the model path
export MODEL_PATH=/home/docker/ros2_ws/src/inference_ros2/model

# Check if a command is provided
if [ "$#" -eq 0 ]; then
    # No command provided, execute the default command
    exec ros2 launch inference_ros2 tracking.launch.py params:=${CONFIG}
else
    # Command provided, execute it
    exec "$@"
fi