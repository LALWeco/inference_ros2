from setuptools import setup
from setuptools import find_packages
package_name = 'inference_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='niqbal',
    maintainer_email='naeemiqbal996@gmail.com',
    description='An inference module for ROS2 that subscribes to Image messages and publishes detections from a YOLOv7 Instance Segmentation model.', 
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crop_detector = inference_ros2.crop_detector:main'
        ],
    },
)
