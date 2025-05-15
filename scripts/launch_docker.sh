docker run -it --runtime=nvidia \
  --env TERM=xterm-256color \
  --network=host \
  --ipc=host \
  -v /dev/shm:/dev/shm \
  -v /home/lero02/bags:/home/docker/bags:ro \
  -e DISPLAY=$DISPLAY \
  -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
  -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
  --user $(id -u):$(id -g) \
  --name=lero_perception_lalweco \
  lalweco/crop_tracker:23.10-humble-py3
