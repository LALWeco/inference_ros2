docker run -it --runtime=nvidia \
  --rm \
  --env TERM=xterm-256color \
  --network=host \
  --ipc=host \
  -v /dev/shm:/dev/shm \
  -v /home/lero02/bags:/home/docker/bags:ro \
  -v /home/niqbal/ros2_ws:/home/docker/ros_ws:ro \
  -e DISPLAY=$DISPLAY \
  --user $(id -u):$(id -g) \
  --name=crop_tracker_leroc2 \
  lalweco/crop_tracker:23.10-humble-py3 \
  /bin/bash -c "source /opt/ros/humble/setup.bash && source /home/docker/ros_ws/install/setup.bash && export MODEL_PATH=/home/docker/ros_ws/src/inference_ros2/model && ros2 launch inference_ros2 tracking.launch.py params:=/home/docker/ros_ws/src/inference_ros2/inference_ros2/config/default_params.yaml"