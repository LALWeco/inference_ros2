#!/bin/bash
set -e

# Source ROS 2 setup files
source /opt/ros/$ROS_DISTRO/setup.bash
source /ros2_ws/install/setup.bash
# Execute the command passed to the docker run
exec "$@"