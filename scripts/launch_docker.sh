docker run -it --runtime=nvidia \
  --rm \
  --env TERM=xterm-256color \
  --network=host \
  --ipc=host \
  -v /dev/shm:/dev/shm \
  -v /home/lero02/bags:/home/docker/bags:ro \
  -e DISPLAY=$DISPLAY \
  --user $(id -u):$(id -g) \
  --name=crop_tracker_leroc2 \
  lalweco/crop_tracker:23.10-humble-py3 \
  /bin/bash 
