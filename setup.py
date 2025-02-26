import os
import warnings
from glob import glob

from generate_parameter_library_py.setup_helper import generate_parameter_module
from setuptools import find_packages, setup

package_name = "inference_ros2"

warnings.filterwarnings("ignore", category=UserWarning, module="setuptools.dist")
generate_parameter_module(
    "inference_parameters", os.path.join(package_name, "inference_parameters.yaml")
)

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
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
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
            "keypoint_detector = inference_ros2.keypoint_detector_trt:main",
            "keypoint_depth_estimator = inference_ros2.keypoint_depth_estimator:main",
        ],
    },
)
