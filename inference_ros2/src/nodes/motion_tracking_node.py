#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D
from lalweco_perception_msgs.msg import Keypoint2D, Keypoint2DArray
import message_filters

from ..core.tracking.motion import MotionEstimator
from ..utils.visualization import draw_detections, draw_tracks
from bytetracker.byte_tracker import BYTETracker

class MotionTrackingNode(Node):
    """ROS2 node for motion estimation and tracking."""
    
    def __init__(self):
        """Initialize the node."""
        super().__init__("motion_tracking_node")    # This has to match the namespace in the config/params.yaml
        
        # Declare parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/sensors/zed_r/zed_node/rgb/image_rect_color"),
                ("detection_topic", "/inference/Keypoint2DDetArray"),
                ("roi.height", 800),
                ("roi.x_min", 360),
                ("roi.x_max", 1160)
            ]
        )
        
        # Get parameters
        self.image_topic = self.get_parameter("image_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.roi = {
            "height": self.get_parameter("roi.height").value,
            "x_min": self.get_parameter("roi.x_min").value,
            "x_max": self.get_parameter("roi.x_max").value
        }
        
        # Initialize components
        self.bridge = CvBridge()
        self.tracker = BYTETracker(
            track_thresh=0.3,
            track_buffer=30,
            match_thresh=0.9,
            frame_rate=5,
            odom_std_weight=0.025
        )
        self.motion_estimator = MotionEstimator()
        self.prev_frame = None
        self.latest_detections = None
        
        # Set up publishers
        self.track_pub = self.create_publisher(
            Keypoint2DArray,
            "/tracking/tracked_keypoints",
            10
        )
        self.viz_pub = self.create_publisher(
            Image,
            "/tracking/visualization",
            10
        )
        
        # Set up synchronized subscribers
        topic_type = CompressedImage if "compressed" in self.image_topic else Image
        self.image_sub = message_filters.Subscriber(
            self,
            topic_type,
            self.image_topic
        )
        self.det_sub = message_filters.Subscriber(
            self,
            Keypoint2DArray,
            self.detection_topic
        )

        # Time synchronizer for image and detection messages
        self.ts = message_filters.TimeSynchronizer(
            [self.image_sub, self.det_sub],
            10  # Queue size
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        self.get_logger().info("Initialized motion tracking node")
        
    def synchronized_callback(self, image_msg, det_msg):
        """Process synchronized image and detection messages.
        
        Args:
            image_msg: ROS image message
            det_msg: Detection array message
        """
        try:
            # Convert message to OpenCV image
            if isinstance(image_msg, CompressedImage):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)
                
            if cv_image.shape[2] != 3:
                cv_image = cv_image[:, :, :3]
            
            # Apply ROI cropping
            cv_image = cv_image[:self.roi["height"],
                              self.roi["x_min"]:self.roi["x_max"]]
            
            # Store original image
            orig_image = cv_image.copy()
            
            # Convert detections to format expected by tracker
            dets = self.convert_detections(det_msg)
            
            if len(dets):
                # Update motion estimation
                motion = self.motion_estimator.estimate_motion(cv_image, self.prev_frame)
                self.prev_frame = cv_image.copy()
                
                # Update tracker
                self.online_targets = self.tracker.update(
                    dets,
                    None,
                    odom_vx=motion.translation_x,
                    odom_vy=motion.translation_y,
                    odom_uncertainty=(motion.uncertainty_x, motion.uncertainty_y)
                )
                
                # Publish tracked results
                self.publish_tracks(self.online_targets, image_msg.header)
                
                # Draw and publish visualization
                viz_img = draw_tracks(
                    orig_image, 
                    self.online_targets,
                    draw_history=True,
                    active_color=(0, 255, 0),
                    history_color=(255, 0, 255)
                )
                self.publish_visualization(viz_img, image_msg.header)
                    
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
    
    def convert_detections(self, det_msg: Keypoint2DArray) -> np.ndarray:
        """Convert detection message to numpy array format for tracker.
        
        Args:
            det_msg: Detection message
            
        Returns:
            Detection array
        """
        dets = []
        for det, kpt in zip(det_msg.detections, det_msg.keypoints):
            bbox = det.bbox
            x1 = bbox.center.position.x - bbox.size_x/2
            y1 = bbox.center.position.y - bbox.size_y/2
            x2 = x1 + bbox.size_x
            y2 = y1 + bbox.size_y
            conf = det.results[0].hypothesis.score
            cls = det.results[0].hypothesis
            
            # Format: [x1, y1, x2, y2, conf, cls, kpt_x, kpt_y, kpt_conf]
            det_array = [x1, y1, x2, y2, conf, 
                        0 if cls.class_id == "weed" else 1,  # 0=weed, 1=crop
                        kpt.position.x, kpt.position.y, kpt.confidence]
            dets.append(det_array)
            
        return np.array(dets) if dets else np.zeros((0, 9))
        
    def publish_tracks(self, tracks, header):
        """Publish tracked keypoints.
        
        Args:
            tracks: List of track objects
            header: ROS message header
        """
        msg = Keypoint2DArray()
        msg.header = header
        
        for track in tracks:
            det = Detection2D()
            tlwh = track.tlwh
            
            # Convert to xyxy format
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            # Get keypoint from track
            # BYTETracker stores keypoints directly in the track object
            kpt = track.keypoint
            
            # Create keypoint message
            keypoint = Keypoint2D()
            keypoint.position.x = float(kpt[0])
            keypoint.position.y = float(kpt[1])
            keypoint.confidence = 1.0  # Tracked points are considered confident
            
            # Set detection info
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = w
            det.bbox.size_y = h
            det.id = str(track.track_id)
            
            msg.keypoints.append(keypoint)
            msg.detections.append(det)
            
        self.track_pub.publish(msg)
        
    def publish_visualization(self, image: np.ndarray, header):
        """Publish visualization image.
        
        Args:
            image: Visualization image
            header: ROS message header
        """
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header = header
        self.viz_pub.publish(msg)

def main():
    rclpy.init()
    node = MotionTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
