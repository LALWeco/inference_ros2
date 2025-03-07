from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Generate launch description for inference nodes."""
    
    pkg_dir = get_package_share_directory('inference_ros2')
    default_params_path = os.path.join(pkg_dir, 'config', 'default_params.yaml')
    
    # Launch arguments
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_path,
        description='Full path to params file'
    )
    
    model_path = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'model'),
        description='Path to model directory'
    )
    
    # Keypoint detector node
    keypoint_detector = Node(
        package='inference_ros2',
        executable='keypoint_detector_node',
        name='keypoint_detector',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'model_path': PathJoinSubstitution([
                    FindPackageShare('inference_ros2'),
                    'model',
                    'yolov8-keypoint-det-cropweed-nuc-fp32-23.10.engine'  # Current engine file name
                ]),
            }
        ],
        output='screen'
    )
    
    # Point estimator node
    point_estimator = Node(
        package='inference_ros2',
        executable='point_estimator_node',
        name='point_estimator',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )

    return LaunchDescription([
        params_file,
        model_path,
        keypoint_detector,
        point_estimator,
    ])
