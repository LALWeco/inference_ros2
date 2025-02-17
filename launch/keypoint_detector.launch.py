from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='inference_ros2',
        #     executable='keypoint_detector_trt',
        #     name='keypoint_detector',
        #     output='screen'
        # ),
        Node(
            package='inference_ros2',
            executable='target_3d_keypoint_estimation',
            name='target_3d_keypoint_estimation',
            output='screen'
        )
    ])
