from setuptools import setup, find_namespace_packages
import os

package_name = 'inference_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_namespace_packages(include=[f'{package_name}*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         [os.path.join(f'{package_name}/launch', f) for f in os.listdir(f'{package_name}/launch')]),
        # Include config files
        (os.path.join('share', package_name, 'config'),
         [os.path.join(f'{package_name}/config', f) for f in os.listdir(f'{package_name}/config')]),
        # Include model directory
        (os.path.join('share', package_name, 'model'),
         [os.path.join('model', f) for f in os.listdir('model')]),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'imutils',
        'tensorrt',
        'pycuda',
    ],
    zip_safe=True,
    maintainer='lalweco',
    maintainer_email='info@lalweco.com',
    description='ROS2 package for real-time keypoint detection and 3D estimation',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keypoint_detector_node = inference_ros2.src.nodes.keypoint_detector_node:main',
            'point_estimator_node = inference_ros2.src.nodes.point_estimator_node:main',
        ],
    },
)
