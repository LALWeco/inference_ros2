import os
from glob import glob

from setuptools import find_packages, setup

package_name = "inference_ros2"
import warnings

import setuptools.dist

warnings.filterwarnings("ignore", category=UserWarning, module="setuptools.dist")

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="niqbal",
    maintainer_email="naeemiqbal996@gmail.com",
    description="An inference module for ROS2 that subscribes to Image messages and publishes detections from a YOLOv7/v8 Instance Segmentation model.",
    license="MIT License",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "keypoint_detector_trt = inference_ros2.keypoint_detector_trt:main",
            "target_3d_keypoint_estimation = inference_ros2.3d_keypoint_estimation:main",
        ],
    },
)
