#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
from lalweco_perception_msgs.msg import Keypoint2DArray

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
                ("estimation_method", rclpy.Parameter.Type.STRING),
                ("keypoint_topic", rclpy.Parameter.Type.STRING),
                ("depth_topic", rclpy.Parameter.Type.STRING),
                ("camera_info_topic", rclpy.Parameter.Type.STRING),
                ("depth_sample_size", rclpy.Parameter.Type.INTEGER),
                ("camera_height", rclpy.Parameter.Type.DOUBLE),
                ("camera_tilt", rclpy.Parameter.Type.DOUBLE),
                ("sync_queue_size", rclpy.Parameter.Type.INTEGER),
                ("sync_slop", rclpy.Parameter.Type.DOUBLE)
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
            MarkerArray,
            "/cropweed/keypoints_3d",
            10
        )
        
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
        if self.camera_matrix is None:
            self.get_logger().warn("Camera matrix not set yet. Skipping keypoint processing.")
            return
            
        self.process_keypoints(msg)
        
    def camera_info_callback(self, msg):
        """Handle camera calibration info.
        
        Args:
            msg: Camera info message
        """
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.estimator.set_camera_matrix(self.camera_matrix)
            self.get_logger().info("Camera matrix set successfully")
            
    def process_keypoints(self, msg):
        """Process keypoint detections and estimate 3D points.
        
        Args:
            msg: Keypoint detection message
        """
        if not msg.keypoints:
            return
            
        if not msg.detections:
            self.get_logger().warn("Received keypoints but no detections")
            return
            
        marker_array = MarkerArray()
        
        for keypoint, detection in zip(msg.keypoints, msg.detections):
            # Estimate 3D point
            point3d = self.estimator.estimate_3d_point(keypoint.position)
            if point3d is not None:
                self.get_logger().debug(f"Estimated 3D point for ID {detection.id}: {point3d}")
                
                # Create marker
                marker = Marker()
                marker.header = msg.header
                marker.ns = "tracked_keypoints"
                marker.id = int(detection.id)
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                
                # Set frame_id if not set in header
                if not marker.header.frame_id:
                    marker.header.frame_id = "camera_link"  # Use camera frame
                
                # Add point to points array
                marker.points.append(point3d)
                
                # Set scale and color
                marker.scale.x = 0.02  # Point width
                marker.scale.y = 0.02  # Point height
                marker.color.r = 0.0
                marker.color.g = 1.0  # Green
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                # Set lifetime
                marker.lifetime.sec = 1  # Show marker for 1 second
                
                marker_array.markers.append(marker)
            else:
                self.get_logger().warn(f"Failed to estimate 3D point for keypoint ID {detection.id}")
                
        if marker_array.markers:
            self.point3d_pub.publish(marker_array)
            self.get_logger().debug(f"Published {len(marker_array.markers)} 3D point markers")
        
def main():
    rclpy.init()
    node = PointEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
