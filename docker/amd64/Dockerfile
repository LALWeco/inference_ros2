FROM nvcr.io/nvidia/tensorrt:23.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=en_US.UTF-8
ENV CONFIG="/home/docker/ros2_ws/src/inference_ros2/inference_ros2/config/default_params.yaml"
# Add non-root user
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=1000

# Set shell
SHELL ["/bin/bash", "-c"]

# Install system dependencies
ARG PACKAGES="wget software-properties-common gnupg2 curl ffmpeg libsm6 libxext6 locales qtbase5-dev qtdeclarative5-dev"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends $PACKAGES && \
    apt-add-repository universe && \
    apt-add-repository multiverse && \
    apt-add-repository restricted && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# Install ROS2
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -y ros-humble-desktop ros-dev-tools python3-argcomplete && \
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install keypoint detection custom message definitions
WORKDIR /home/$USERNAME/ros2_ws/src

# Clone repositories with error handling
RUN git clone https://github.com/LALWeco/lalweco_perception_msgs.git || echo "Failed to clone lalweco_perception_msgs" && \
    git clone https://github.com/ros-perception/vision_msgs.git || echo "Failed to clone vision_msgs" && \
    git clone https://github.com/LALWeco/inference_ros2.git || echo "Failed to clone inference_ros2" && \
    cd inference_ros2 && git checkout dev 

# Create workspace directory structure
RUN mkdir -p /home/$USERNAME/ros2_ws/src/inference_ros2

# Build ROS2 workspace (initial build without source code)
WORKDIR /home/$USERNAME/ros2_ws/
RUN . /opt/ros/humble/setup.sh && colcon build

RUN echo "source /home/${USERNAME}/ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up color prompt and aliases properly
RUN echo "PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /home/$USERNAME/.bashrc && \
    echo "force_color_prompt=yes" >> /home/$USERNAME/.bashrc && \
    echo "alias ls='ls --color=auto'" >> /home/$USERNAME/.bashrc && \
    echo "alias grep='grep --color=auto'" >> /home/$USERNAME/.bashrc

# Source ROS in the correct user's bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "source /home/${USERNAME}/ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "export MODEL_PATH=/home/${USERNAME}/ros2_ws/src/inference_ros2/model" >> /home/$USERNAME/.bashrc

# Set ownership of all workspace files
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# Create .ros directory with correct permissions
RUN mkdir -p /home/$USERNAME/.ros && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.ros

# Switch to non-root user
USER $USERNAME

WORKDIR /home/$USERNAME/ros2_ws/src/inference_ros2
# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install -e ./inference_ros2/bytetrack-pip 

# Set the entrypoint dynamically using the USERNAME value at build time
ARG ENTRYPOINT_PATH="/home/${USERNAME}/ros2_ws/src/inference_ros2/entrypoint.sh"
ENTRYPOINT ["/bin/bash", "-c", "${ENTRYPOINT_PATH}"]

CMD ["/bin/bash", "-c", "ros2", "launch", "inference_ros2", "tracking.launch.py", "--params-file", "${CONFIG}"]