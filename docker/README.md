# Docker 

The repository has been tested with Ubuntu20.04 and ROS 2 Foxy. 

Build the image

## ROS 2 Foxy (Ubuntu 20.04)
```bash
cd <root directory of this repo>
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/amd64/Dockerfile_foxy -t niqbal996/inference_ros2:23.04-foxy-py3 .
```
Run the container
```bash
docker run -it --runtime=nvidia --net=host --ipc=host -v /home/niqbal/ros2_ws:/root/ros2_ws -v /dev/shm:/dev/shm --name=infer_ros2 niqbal996/inference_ros2:23.04-foxy-py3
```

## ROS 2 Humble (Ubuntu 22.04)

```bash
cd <root directory of this repo>
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/amd64/Dockerfile_foxy -t niqbal996/inference_ros2:23.10-humble-py3 .
```
Run the container
```bash
docker run -it --runtime=nvidia --net=host --ipc=host -v /home/niqbal/ros2_ws:/root/ros2_ws -v /dev/shm:/dev/shm --name=infer_ros2 niqbal996/inference_ros2:23.10-humble-py3
```