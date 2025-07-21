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
                ("image_topic", rclpy.Parameter.Type.STRING),
                ("detection_topic", rclpy.Parameter.Type.STRING),
                ("roi.height", rclpy.Parameter.Type.INTEGER),
                ("roi.x_min", rclpy.Parameter.Type.INTEGER),
                ("roi.x_max", rclpy.Parameter.Type.INTEGER),
                ("tracking.track_thresh", rclpy.Parameter.Type.DOUBLE),
                ("tracking.track_buffer", rclpy.Parameter.Type.INTEGER),
                ("tracking.match_thresh", rclpy.Parameter.Type.DOUBLE),
                ("tracking.frame_rate", rclpy.Parameter.Type.INTEGER),
                ("tracking.odom_std_weight", rclpy.Parameter.Type.DOUBLE),
                ("use_cuda", rclpy.Parameter.Type.BOOL),  # Use CUDA for GPU acceleration
                ("visualization", rclpy.Parameter.Type.BOOL)  # Enable visualization
            ]
        )
        
        # Get parameters
        self.image_topic = self.get_parameter("image_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.use_cuda = self.get_parameter("use_cuda").value
        self.visualization = self.get_parameter("visualization").value
        self.queue_size = self.get_parameter("queue_size").value
        self.roi = {
            "height": self.get_parameter("roi.height").value,
            "x_min": self.get_parameter("roi.x_min").value,
            "x_max": self.get_parameter("roi.x_max").value
        }
        
        # Initialize components
        self.bridge = CvBridge()
        self.tracker = BYTETracker(
            track_thresh=self.get_parameter("tracking.track_thresh").value,
            track_buffer=self.get_parameter("tracking.track_buffer").value,
            match_thresh=self.get_parameter("tracking.match_thresh").value,
            frame_rate=self.get_parameter("tracking.frame_rate").value,
            odom_std_weight=self.get_parameter("tracking.odom_std_weight").value
        )
        self.motion_estimator = MotionEstimator()
        self.prev_frame = None
        self.latest_detections = None
        
        # Set up publishers
        self.track_pub = self.create_publisher(
            Keypoint2DArray,
            "/tracking/tracked_keypoints",
            self.queue_size
        )
        if self.visualization:
            self.viz_pub = self.create_publisher(
                Image,
                "/tracking/visualization",
                self.queue_size
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
            self.queue_size  # Queue size
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Pre-allocate arrays to avoid memory allocation overhead
        self.prev_pts_buffer = np.zeros((1000, 2), dtype=np.float32)
        self.curr_pts_buffer = np.zeros((1000, 2), dtype=np.float32)

        # Use OpenCV optimizations
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)  # Use multiple cores
        
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
                if self.use_cuda:
                    motion = self.motion_estimator.estimate_motion_cuda(cv_image, self.prev_frame)
                else:
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
            det_msg: Detection message (coordinates relative to full image)
            
        Returns:
            Detection array (coordinates relative to cropped image)
        """
        dets = []
        for det, kpt in zip(det_msg.detections, det_msg.keypoints):
            bbox = det.bbox
            # Convert received full-image coordinates to cropped-image coordinates for the tracker
            x1_full = bbox.center.position.x - bbox.size_x/2
            y1_full = bbox.center.position.y - bbox.size_y/2
            x2_full = x1_full + bbox.size_x
            y2_full = y1_full + bbox.size_y
            
            x1_cropped = x1_full - self.roi["x_min"]
            y1_cropped = y1_full # Assuming ROI starts at y=0
            x2_cropped = x2_full - self.roi["x_min"]
            y2_cropped = y2_full # Assuming ROI starts at y=0
            
            conf = det.results[0].hypothesis.score
            cls = det.results[0].hypothesis
            
            # Keypoint coordinates relative to cropped image
            kpt_x_cropped = kpt.position.x - self.roi["x_min"]
            kpt_y_cropped = kpt.position.y # Assuming ROI starts at y=0
            
            # Format: [x1, y1, x2, y2, conf, cls, kpt_x, kpt_y, kpt_conf]
            # Use cropped coordinates for the tracker
            det_array = [x1_cropped, y1_cropped, x2_cropped, y2_cropped, conf, 
                        0 if cls.class_id == "weed" else 1,  # 0=weed, 1=crop
                        kpt_x_cropped, kpt_y_cropped, kpt.confidence]
            dets.append(det_array)
            
        return np.array(dets) if dets else np.zeros((0, 9))
        
    def publish_tracks(self, tracks, header):
        """Publish tracked keypoints (relative to full image).
        
        Args:
            tracks: List of track objects (coordinates relative to cropped image)
            header: ROS message header
        """
        msg = Keypoint2DArray()
        msg.header = header
        
        for track in tracks:
            det = Detection2D()
            # tlwh = track.tlwh # tlwh is relative to cropped image
            tlbr = track._detection
            # Convert cropped tlwh to full-image center and size
            x1_cropped, y1_cropped, x2_cropped, y2_cropped = tlbr
            w = x2_cropped - x1_cropped
            h = y2_cropped - y1_cropped
            
            # Center in cropped image
            center_x_cropped = x1_cropped + w / 2.0
            center_y_cropped = y1_cropped + h / 2.0
            
            # Convert to full image coordinates
            center_x_full = center_x_cropped + self.roi["x_min"]
            center_y_full = center_y_cropped # Assuming ROI starts at y=0
            
            # Get keypoint from track (relative to cropped image)
            kpt_cropped = track.keypoint 
            
            # Create keypoint message (add offset back for full image coordinates)
            keypoint = Keypoint2D()
            keypoint.position.x = float(kpt_cropped[0]) + self.roi["x_min"] # Add offset back
            keypoint.position.y = float(kpt_cropped[1]) # No y-offset needed if roi starts at y=0
            keypoint.confidence = 1.0  # Tracked points are considered confident
            
            # Set detection info (using full image coordinates)
            det.bbox.center.position.x = center_x_full
            det.bbox.center.position.y = center_y_full
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
