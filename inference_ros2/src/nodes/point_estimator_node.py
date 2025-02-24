#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from lalweco_perception_msgs.msg import Keypoint2D, Keypoint2DArray, Keypoint3D, Keypoint3DArray
from lalweco_laser_module_msgs.action import ControlLaser

from ..core.estimation.depth_estimator import DepthEstimator
from ..core.estimation.geometric_estimator import GeometricEstimator

class PointEstimatorNode(Node):
    """ROS2 node for 3D point estimation."""
    
    def __init__(self):
        """Initialize the node."""
        super().__init__("point_estimator")
        
        # Declare parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("estimation_method", "depth"),
                ("keypoint_topic", "/inference/Keypoint2DDetArray"),
                ("depth_topic", "/sensors/zed_laser_module/zed_node/depth/depth_registered"),
                ("camera_info_topic", "/sensors/zed_laser_module/zed_node/rgb_gray/camera_info"),
                ("depth_sample_size", 5),
                ("camera_height", 1.0),
                ("camera_tilt", 30.0),
                ("sync_queue_size", 10),
                ("sync_slop", 0.1),
                ("target_duration", 0.1),
                ("laser_offset.x", 0.028),
                ("laser_offset.y", 0.148)
            ]
        )
        
        # Get parameters
        self.method = self.get_parameter("estimation_method").value
        self.use_depth = self.method == "depth"
        
        # Initialize components
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        # Initialize appropriate estimator
        if self.use_depth:
            self.estimator = DepthEstimator(
                depth_sample_size=self.get_parameter("depth_sample_size").value
            )
            
            # Set up synchronized subscribers
            self.keypoint_sub = Subscriber(
                self,
                Keypoint2DArray,
                self.get_parameter("keypoint_topic").value
            )
            self.depth_sub = Subscriber(
                self,
                Image,
                self.get_parameter("depth_topic").value
            )
            
            # Create synchronizer
            self.ts = ApproximateTimeSynchronizer(
                [self.keypoint_sub, self.depth_sub],
                queue_size=self.get_parameter("sync_queue_size").value,
                slop=self.get_parameter("sync_slop").value
            )
            self.ts.registerCallback(self.sync_callback)
            
        else:
            self.estimator = GeometricEstimator(
                camera_height=self.get_parameter("camera_height").value,
                camera_tilt_degrees=self.get_parameter("camera_tilt").value
            )
            
            # Set up regular subscriber
            self.keypoint_sub = self.create_subscription(
                Keypoint2DArray,
                self.get_parameter("keypoint_topic").value,
                self.keypoint_callback,
                10
            )
            
        # Set up camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            10
        )
        
        # Set up publisher
        self.point3d_pub = self.create_publisher(
            Keypoint3DArray,
            "/cropweed/keypoints_3d",
            10
        )
        
        # Set up laser control
        self._control_laser_event = threading.Event()
        self._control_laser_action = ActionClient(
            self,
            ControlLaser,
            "/lalweco_laser_module_driver/control_laser"
        )
        self._target_id = 0
        
        self.get_logger().info(
            f"Initialized point estimator node using {self.method} method"
        )
        
    def sync_callback(self, keypoint_msg, depth_msg):
        """Handle synchronized keypoint and depth messages.
        
        Args:
            keypoint_msg: Keypoint detection message
            depth_msg: Depth image message
        """
        # Update depth image
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
        self.estimator.set_depth_image(depth_image)
        
        # Process keypoints
        self.process_keypoints(keypoint_msg)
        
    def keypoint_callback(self, msg):
        """Handle keypoint messages for geometric estimation.
        
        Args:
            msg: Keypoint detection message
        """
        self.process_keypoints(msg)
        
    def camera_info_callback(self, msg):
        """Handle camera calibration info.
        
        Args:
            msg: Camera info message
        """
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.estimator.set_camera_matrix(self.camera_matrix)
            
    def process_keypoints(self, msg):
        """Process keypoint detections and estimate 3D points.
        
        Args:
            msg: Keypoint detection message
        """
        if not msg.keypoints:
            return
            
        keypoint3d_array = Keypoint3DArray()
        keypoint3d_array.header = msg.header
        
        # Find keypoint closest to image center
        center_x = self.camera_matrix[0, 2]
        center_y = self.camera_matrix[1, 2]
        
        closest_keypoint = min(
            msg.keypoints,
            key=lambda kp: (kp.position.x - center_x) ** 2 + 
                         (kp.position.y - center_y) ** 2
        )
        
        # Estimate 3D point
        point3d = self.estimator.estimate_3d_point(closest_keypoint)
        
        if point3d is not None:
            # Send laser control action
            self.send_laser_control(point3d)
            
            # Create and publish 3D keypoint message
            keypoint3d = Keypoint3D()
            keypoint3d.id = "1"
            keypoint3d.point = point3d
            keypoint3d_array.keypoints.append(keypoint3d)
            
        self.point3d_pub.publish(keypoint3d_array)
        
    def send_laser_control(self, point: Point):
        """Send laser control action.
        
        Args:
            point: Target 3D point
        """
        goal = ControlLaser.Goal()
        goal.target_id = self._target_id
        self._target_id += 1
        
        goal.header.frame_id = "laser_module_r"
        goal.header.stamp = self.get_clock().now().to_msg()
        
        # Convert to meters and apply offset
        x_offset = self.get_parameter("laser_offset.x").value
        y_offset = self.get_parameter("laser_offset.y").value
        
        goal.target_position = Point(
            x=-point.x / 1000.0 + x_offset,
            y=point.y / 1000.0 + y_offset,
            z=-point.z / 1000.0
        )
        
        goal.duration = self.get_parameter("target_duration").value
        goal.beam_diameter = 0.0
        goal.power = 0.0  # Just aiming
        
        # Send goal
        self._control_laser_event.clear()
        future = self._control_laser_action.send_goal_async(goal)
        future.add_done_callback(self._control_goal_callback)
        
    def _control_goal_callback(self, future):
        """Handle laser control goal response.
        
        Args:
            future: Goal response future
        """
        goal_handle = future.result()
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._control_result_callback)
        
    def _control_result_callback(self, future):
        """Handle laser control action result.
        
        Args:
            future: Result future
        """
        self._control_laser_event.set()

def main():
    rclpy.init()
    node = PointEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
