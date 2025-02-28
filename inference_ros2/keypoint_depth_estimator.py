import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from lalweco_perception_msgs.msg import Keypoint2DArray, Keypoint3D, Keypoint3DArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from inference_ros2.inference_parameters import inference


class KeypointDepthEstimator(Node):
    def __init__(self):
        super().__init__("keypoint_depth_estimator")
        param_listener = inference.ParamListener(self)
        params = param_listener.get_params()

        self.bridge = CvBridge()

        self.use_depth = params.depth_estimation.use_depth

        if self.use_depth:
            depth_image_topic = params.depth_estimation.depth_image_topic
            self.get_logger().info(f"Subscribing to {depth_image_topic}")
            self.depth_sub = Subscriber(self, Image, depth_image_topic)

            self.keypoint_sub = Subscriber(self, Keypoint2DArray, "/inference/keypoints_2d")

            # Synchronize messages within 0.1 seconds
            self.ts = ApproximateTimeSynchronizer(
                [self.keypoint_sub, self.depth_sub], queue_size=10, slop=0.1
            )
            self.ts.registerCallback(self.sync_callback)

            self.depth_sample_size = params.depth_estimation.depth_sample_size
        else:
            # Regular subscriber for non-depth method
            self.keypoint_sub = self.create_subscription(
                Keypoint2DArray,
                "/inference/keypoints_2d",
                self.keypoint_callback,
                10,
            )

            self.camera_height = params.depth_estimation.camera_height
            self.camera_tilt = np.radians(params.depth_estimation.camera_tilt)

        camera_info_topic = params.depth_estimation.camera_info_topic
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.camera_matrix = None
        self.point3d_pub = self.create_publisher(Keypoint3DArray, "/inference/keypoints_3d", 10)

    def sync_callback(self, keypoint_msg, depth_msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(depth_msg)
        self.process_keypoints(keypoint_msg)

    def keypoint_callback(self, msg):
        self.process_keypoints(msg)

    def process_keypoints(self, msg):
        if not msg.keypoints:
            return

        keypoint3d_array = Keypoint3DArray()
        keypoint3d_array.header = msg.header

        for point in msg.keypoints:
            point3d = (
                self.estimate_3d_point_depth(point)
                if self.use_depth
                else self.estimate_3d_point_geometry(point)
            )

            if point3d is not None:
                keypoint3d = Keypoint3D()
                keypoint3d.id = "1"  # Assign ID 1
                keypoint3d.point = point3d
                keypoint3d_array.keypoints.append(keypoint3d)

        self.get_logger().info(f"Published {len(keypoint3d_array.keypoints)} 3D keypoints.")
        self.point3d_pub.publish(keypoint3d_array)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def get_depth_at_point(self, x, y):
        if self.latest_depth_img is None:
            return None

        x, y = int(x), int(y)
        size = self.depth_sample_size
        h, w = self.latest_depth_img.shape

        # Define sampling box
        x1 = max(0, x - size // 2)
        x2 = min(w, x + size // 2 + 1)
        y1 = max(0, y - size // 2)
        y2 = min(h, y + size // 2 + 1)

        # Get median depth in sampling box
        depth_region = self.latest_depth_img[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0]

        return np.median(valid_depths) if len(valid_depths) > 0 else None

    def estimate_3d_point_depth(self, keypoint):
        x, y = keypoint.position.x, keypoint.position.y
        depth = self.get_depth_at_point(x, y)

        if depth is None:
            return None

        # Back-project to 3D
        # x_normalized = (x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        # y_normalized = (y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        # point.x = x_normalized * depth
        # point.y = y_normalized * depth

        point = Point()
        point.x = x
        point.y = y
        point.z = float(depth)

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


def main():
    rclpy.init()
    node = KeypointDepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
