FROM ubuntu:20.04

WORKDIR /ros2_ws/src/inference_ros2
COPY . /ros2_ws/src/inference_ros2
USER root
SHELL ["/bin/bash", "-c"]
# INSTALL python dependencies
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt update && \
    apt install -y wget software-properties-common gnupg2 curl ffmpeg libsm6 libxext6 && \ 
    apt-add-repository universe && \
    apt-add-repository multiverse && \ 
    apt-add-repository restricted && \
    python3 -m pip install --upgrade pip && \
    apt autoremove

# Run pip install using the download cache
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install onnxruntime-gpu imutils opencv-python opencv-contrib-python torchvision==0.14.1 torch==1.13.1

RUN apt update && apt install locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    LANG=en_US.UTF-8

# Install ROS and use the download cache
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    # sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    # apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    # apt-get update && \
    # apt-get install -y ros-noetic-ros-base && \
    # apt install -y python3-rosdep python3-wstool build-essential  && \
    # apt install -y python3-rospkg && \
    # apt install -y ros-noetic-rostopic ros-noetic-rospy ros-noetic-roslaunch ros-noetic-sensor-msgs && \
    # apt install -y python3-rosdep && rosdep init && rosdep update && \
    add-apt-repository universe  && \
    apt update && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt update && \
    apt upgrade -y && \
    apt install -y ros-foxy-desktop ros-dev-tools python3-argcomplete && \
    apt install -y qtbase5-dev qtdeclarative5-dev && \
    apt autoremove && \
    echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/workspace/ros_5g/docker/amd64/entrypoint.sh"]
