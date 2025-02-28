import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_name = "inference_ros2"
    package_dir = get_package_share_directory(package_name)

    config_file_path = os.path.join(package_dir, "config", "laser_module.yaml")

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_file_path,
        description="Path to the configuration YAML file.",
    )

    return LaunchDescription(
        [
            config_file_arg,
            Node(
                package="inference_ros2",
                executable="keypoint_depth_estimator",
                name="keypoint_depth_estimator",
                output="screen",
                parameters=[LaunchConfiguration("config_file")],
            ),
        ]
    )
