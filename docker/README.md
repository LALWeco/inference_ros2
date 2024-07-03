# Docker 

The repository has been tested with Ubuntu20.04 and ROS 2 Foxy. 

Build the image
````bash
cd <root director of this repo>
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/amd64/Dockerfile -t niqbal996/inference_ros2:23.04-foxy-py3 .
```
Run the container
```bash
docker run -it --runtime=nvidia --net=host -v/home/niqbal/ros2_ws:/root/ros2_ws -v/home/niqbal/git/aa_detectors:/opt/workspace/ -v /dev/shm:/dev/shm --name=infer_ros2 niqbal996/inference_ros2:23.04-foxy-py3
```

