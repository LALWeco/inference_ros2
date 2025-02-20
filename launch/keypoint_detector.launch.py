from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    engine_path_arg = DeclareLaunchArgument(
        "engine_path",
        default_value="/home/docker/ros2_ws/src/inference_ros2/model/yolov8-keypoint-det-cropweed-nuc-fp32-23.10.engine",
        description="Path to the TensorRT engine file",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/sensors/zed_laser_module/zed_node/rgb/image_rect_color/compressed",
        description="Topic name for the input image",
    )

    depth_image_topic_arg = DeclareLaunchArgument(
        "depth_image_topic",
        default_value="/sensors/zed_laser_module/zed_node/depth/depth_registered",
        description="Topic name for the depth image",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/sensors/zed_laser_module/zed_node/rgb_gray/camera_info",
        description="Topic name for the camera info",
    )

    return LaunchDescription(
        [
            engine_path_arg,
            image_topic_arg,
            depth_image_topic_arg,
            camera_info_topic_arg,
            Node(
                package="inference_ros2",
                executable="keypoint_detector_trt",
                name="keypoint_detector",
                output="screen",
                parameters=[
                    {
                        "engine_path": LaunchConfiguration("engine_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                    }
                ],
            ),
            Node(
                package="inference_ros2",
                executable="target_3d_keypoint_estimation",
                name="target_3d_keypoint_estimation",
                output="screen",
                parameters=[
                    {
                        "depth_image_topic_arg": LaunchConfiguration("depth_image_topic_arg"),
                        "camera_info_topic_arg": LaunchConfiguration("camera_info_topic_arg"),
                    }
                ],
            ),
        ]
    )
