import threading
import time

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from lalweco_laser_module_msgs.action import ControlLaser
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from lalweco_perception_msgs.msg import Keypoint2DArray, Keypoint3D, Keypoint3DArray


class Point3DEstimator(Node):
    def __init__(self, use_depth=True):
        super().__init__("point_3d_estimator")
        self.use_depth = use_depth
        self.bridge = CvBridge()

        if use_depth:
            # Synchronized subscribers
            self.keypoint_sub = Subscriber(
                self, Keypoint2DArray, "/inference/Keypoint2DDetArray"
            )
            self.depth_sub = Subscriber(
                self, Image, "/sensors/zed_laser_module/zed_node/depth/depth_registered"
            )

            # Synchronize messages within 0.1 seconds
            self.ts = ApproximateTimeSynchronizer(
                [self.keypoint_sub, self.depth_sub], queue_size=10, slop=0.1
            )
            self.ts.registerCallback(self.sync_callback)

            self.declare_parameter("depth_sample_size", 5)
            self.depth_sample_size = self.get_parameter("depth_sample_size").value
        else:
            # Regular subscriber for non-depth method
            self.keypoint_sub = self.create_subscription(
                Keypoint2DArray,
                "/inference/Keypoint2DDetArray",
                self.keypoint_callback,
                10,
            )

            self.declare_parameters(
                namespace="",
                parameters=[
                    ("camera_height", 1.0),
                    ("camera_tilt", 30.0),
                ],
            )
            self.camera_height = self.get_parameter("camera_height").value
            self.camera_tilt = np.radians(self.get_parameter("camera_tilt").value)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/sensors/zed_laser_module/zed_node/rgb_gray/camera_info",
            self.camera_info_callback,
            10,
        )

        self.camera_matrix = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.point3d_pub = self.create_publisher(
            Keypoint3DArray, "/cropweed/keypoints_3d", 10
        )

        # Add laser control action client
        self._control_laser_event = threading.Event()
        self._control_laser_action = ActionClient(
            self, ControlLaser, "/lalweco_laser_module_driver/control_laser"
        )
        self._target_id = 0

    def sync_callback(self, keypoint_msg, depth_msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg)
        self.process_keypoints(keypoint_msg)

    def keypoint_callback(self, msg):
        self.process_keypoints(msg)

    def process_keypoints(self, msg):
        if not msg.keypoints:
            return

        keypoint3d_array = Keypoint3DArray()
        keypoint3d_array.header = msg.header

        if not msg.keypoints:
            return

        # Find the keypoint closest to the center
        center_x, center_y = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        closest_keypoint = min(
            msg.keypoints,
            key=lambda kp: (kp.position.x - center_x) ** 2
            + (kp.position.y - center_y) ** 2,
        )

        point3d = (
            self.estimate_3d_point_depth(closest_keypoint)
            if self.use_depth
            else self.estimate_3d_point_geometry(closest_keypoint)
        )

        if point3d is not None:
            # Send laser control action before publishing
            self.send_laser_control(point3d)

            keypoint3d = Keypoint3D()
            keypoint3d.id = "1"  # Assign ID 1
            keypoint3d.point = point3d
            keypoint3d_array.keypoints.append(keypoint3d)

        self.point3d_pub.publish(keypoint3d_array)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def get_depth_at_point(self, x, y):
        if self.latest_depth is None:
            return None

        x, y = int(x), int(y)
        size = self.depth_sample_size
        h, w = self.latest_depth.shape

        # Define sampling box
        x1 = max(0, x - size // 2)
        x2 = min(w, x + size // 2 + 1)
        y1 = max(0, y - size // 2)
        y2 = min(h, y + size // 2 + 1)

        # Get median depth in sampling box
        depth_region = self.latest_depth[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0]

        return np.median(valid_depths) if len(valid_depths) > 0 else None

    def estimate_3d_point_depth(self, keypoint):
        x, y = keypoint.position.x, keypoint.position.y
        depth = self.get_depth_at_point(x, y)

        if depth is None:
            return None

        # Back-project to 3D
        x_normalized = (x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_normalized = (y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]

        point = Point()
        point.z = float(depth)
        point.x = x_normalized * depth
        point.y = y_normalized * depth

        return point

    def estimate_3d_point_geometry(self, keypoint):
        if self.camera_matrix is None:
            return None

        x, y = keypoint.position.x, keypoint.position.y

        # Get ray direction in camera coordinates
        x_normalized = (x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_normalized = (y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        ray = np.array([x_normalized, y_normalized, 1.0])
        ray = ray / np.linalg.norm(ray)

        # Apply camera tilt (90° - tilt_angle to make 0° look down)
        tilt_rad = np.pi / 2 - self.camera_tilt
        ray_rotated = np.array(
            [
                ray[0],
                ray[1] * np.cos(tilt_rad) - ray[2] * np.sin(tilt_rad),
                ray[1] * np.sin(tilt_rad) + ray[2] * np.cos(tilt_rad),
            ]
        )

        # Fix horizontal case
        if ray_rotated[2] == 0:  # Ray doesn't intersect ground
            return None

        t = -self.camera_height / ray_rotated[2]

        point = Point()
        point.x = t * ray_rotated[0]
        point.y = t * ray_rotated[1]
        point.z = 0.0

        return point

    def send_laser_control(self, point):
        goal = ControlLaser.Goal()
        goal.target_id = self._target_id
        self._target_id += 1

        goal.header.frame_id = "laser_module_r"
        goal.header.stamp = self.get_clock().now().to_msg()

        # These are magic values that seemed to work during testing
        goal.target_position = Point(
            x=-point.x / 1000.0 + 0.028,
            y=point.y / 1000.0 + 0.148,
            z=-point.z / 1000.0,
        )
        goal.duration = 0.1  # Short duration just for aiming
        goal.beam_diameter = 0.0
        goal.power = 0.0  # No power, just aiming

        # Send goal and wait for completion
        self._control_laser_event.clear()
        future = self._control_laser_action.send_goal_async(goal)
        future.add_done_callback(self._control_goal_cb)
        time.sleep(0.1)
        # self._control_laser_event.wait()

    def _control_goal_cb(self, future):
        goal_handle = future.result()
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._control_result_cb)

    def _control_result_cb(self, future):
        self._control_laser_event.set()


def main():
    rclpy.init()
    node = Point3DEstimator(use_depth=True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
