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
docker run -it --runtime=nvidia --net=host --ipc=host -v /home/niqbal/ros2_ws:/root/ros2_ws -v /dev/shm:/dev/shm --name=infer_ros2_foxy niqbal996/inference_ros2:23.04-foxy-py3
```

## ROS 2 Humble (Ubuntu 22.04)

```bash
cd <root directory of this repo>
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/amd64/Dockerfile_humble -t niqbal996/inference_ros2:23.10-humble-py3 .
```
Run the container
```bash
docker run -it --runtime=nvidia --net=host --ipc=host -v /home/niqbal/ros2_ws:/root/ros2_ws -v /dev/shm:/dev/shm -e ROS_DOMAIN_ID --name=infer_ros2_humble niqbal996/inference_ros2:23.10-humble-py3
```

# NVidia Driver and CUDA installation on Intel NUC
According to official NVIDIA [documentation](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-toolkit-release-notes/index.html), for Cuda 12.x, the driver version has >=5.25 on Linux machines. 

```bash
sudo apt update
sudo apt install nvidia-driver-535
sudo apt install nvidia-utils-535
```
Nvidia-Cuda toolkit 
```bash
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-container-toolkit
```
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt install nvidia-container-toolkit 
sudo apt install nvidia-container-toolkit-base
```