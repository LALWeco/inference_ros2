from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Generate launch description for tracking and inference nodes."""
    
    pkg_dir = get_package_share_directory('inference_ros2')
    default_params_path = os.path.join(pkg_dir, 'config', 'default_params.yaml')
    
    # Launch arguments
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_path,
        description='Full path to params file'
    )
    
    # Include the inference launch file for keypoint detection
    inference_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('inference_ros2'),
                'launch',
                'inference.launch.py'
            ])
        ])
    )
    
    # Motion tracking node
    motion_tracking = Node(
        package='inference_ros2',
        executable='motion_tracking_node',
        name='motion_tracking_node', # This has to match the namespace in the config/params.yaml
        parameters=[LaunchConfiguration('params_file')],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        params_file,
        inference_launch,
        motion_tracking,
    ])
