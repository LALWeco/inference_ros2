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
import time
import gc
import os

# Import psutil with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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
                ("visualization", rclpy.Parameter.Type.BOOL),  # Enable visualization
                ("queue_size", rclpy.Parameter.Type.INTEGER),  # Queue size for subscribers, publishers
                ("tracker_reset_interval", rclpy.Parameter.Type.INTEGER),  # Reset tracker every N frames
                ("max_tracks", rclpy.Parameter.Type.INTEGER),  # Maximum number of tracks to maintain
                ("debug_mode", rclpy.Parameter.Type.BOOL),  # Enable debug mode for additional logging
            ]
        )
        
        # Get parameters
        self.image_topic = self.get_parameter("image_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.use_cuda = self.get_parameter("use_cuda").value
        self.debug_mode = self.get_parameter("debug_mode").value
        # Set default values for parameters that might not be in config
        try:
            self.visualization = self.get_parameter("visualization").value
        except:
            self.visualization = True  # Default to True if not specified
        try:
            self.queue_size = self.get_parameter("queue_size").value
        except:
            self.queue_size = 10  # Default queue size
        try:
            self.tracker_reset_interval = self.get_parameter("tracker_reset_interval").value
        except:
            self.tracker_reset_interval = 1000  # Default: reset every 1000 frames
        try:
            self.max_tracks = self.get_parameter("max_tracks").value
        except:
            self.max_tracks = 50  # Default: max 50 tracks
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
        self.callback_count = 0  # Add callback counter for debugging
        
        # Debug timing variables
        self.last_image_time = None
        self.last_detection_time = None
        self.last_callback_time = None
        
        # Performance tracking for periodic summaries
        self.sync_issues_count = 0
        self.total_processing_time = 0
        # Initialize with ROS time instead of system time
        self.last_summary_time = self.get_clock().now().nanoseconds * 1e-9
        
        # Memory monitoring and cleanup
        self.processed_frames = 0
        # Use the configured values instead of hardcoded ones
        # self.tracker_reset_interval and self.max_tracks are now set from parameters
        
        # Create periodic diagnostic timer
        self.create_timer(30.0, self.log_performance_summary)  # Every 30 seconds
        
        # Create periodic cleanup timer
        self.create_timer(60.0, self.periodic_cleanup)  # Every 60 seconds
        
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

        # Add individual callbacks for debugging message arrival times
        self.image_sub.registerCallback(self.debug_image_callback)
        self.det_sub.registerCallback(self.debug_detection_callback)

        # Time synchronizer for image and detection messages
        # Use ApproximateTimeSynchronizer for more flexible timing
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.det_sub],
            self.queue_size,  # Queue size
            0.4  # 300ms tolerance - adjust based on your system
        )
        if self.debug_mode:
            # x, y coordinates for fixed tracks in debug mode
            # These coordinates are relative to the full image size
            self.kpt_coords = [
                            [180,434],
                            [398,502],
                            [770,408],
                            [1096,510],
                            [965,687],
                            [656,679],
                            [757,840],
                            [1149,1018],
                            [953,1015],
                            [546,1031],
                            [320,1023]
                            ]
            self.ts.registerCallback(self.synchronized_callback_debug)
        else:
            self.ts.registerCallback(self.synchronized_callback)
        
        # Pre-allocate arrays to avoid memory allocation overhead
        self.prev_pts_buffer = np.zeros((1000, 2), dtype=np.float32)
        self.curr_pts_buffer = np.zeros((1000, 2), dtype=np.float32)

        # Use OpenCV optimizations
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)  # Use multiple cores
        
        self.get_logger().warn("=== Motion Tracking Node Initialized Successfully ===")
        self.get_logger().warn(f"Config: CUDA={self.use_cuda}, Visualization={self.visualization}, Queue={self.queue_size}")
        if self.debug_mode:
            self.get_logger().warn(f"Launching motion tracking in DEBUG MODE. Fixed tracks will be published at [x,y] coordinates: \n {self.kpt_coords}")

    def debug_image_callback(self, msg):
        """Debug callback to track image message arrival."""
        # Use ROS time instead of system time for consistent timing
        current_ros_time = self.get_clock().now().nanoseconds * 1e-9
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.last_image_time is not None:
            interval = (current_ros_time - self.last_image_time) * 1000
            age = (current_ros_time - msg_time) * 1000
            # Much more aggressive filtering - only log severe issues
            if age > 1000000:  # More than 1000 seconds old
                if self.callback_count % 100 == 1:  # Log every 100th callback only
                    self.get_logger().error(f"[TIMESTAMP ERROR] Img age: {age/1000:.1f}s")
            elif interval > 500 or age > 2000:  # Much higher thresholds
                if self.callback_count % 50 == 1:  # Log every 50th occurrence
                    self.get_logger().warn(f"[IMG ISSUE] Interval: {interval:.1f}ms, Age: {age:.1f}ms")
        
        self.last_image_time = current_ros_time
        
    def debug_detection_callback(self, msg):
        """Debug callback to track detection message arrival."""
        # Use ROS time instead of system time for consistent timing
        current_ros_time = self.get_clock().now().nanoseconds * 1e-9
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.last_detection_time is not None:
            interval = (current_ros_time - self.last_detection_time) * 1000
            age = (current_ros_time - msg_time) * 1000
            # Much more aggressive filtering - only log severe issues
            if age > 1000000:  # More than 1000 seconds old
                if self.callback_count % 100 == 1:  # Log every 100th callback only
                    self.get_logger().error(f"[TIMESTAMP ERROR] Det age: {age/1000:.1f}s, Count: {len(msg.detections)}")
            elif interval > 500 or age > 2000:  # Much higher thresholds
                if self.callback_count % 50 == 1:  # Log every 50th occurrence
                    self.get_logger().warn(f"[DET ISSUE] Interval: {interval:.1f}ms, Age: {age:.1f}ms, Count: {len(msg.detections)}")
        
        self.last_detection_time = current_ros_time
        
    def synchronized_callback(self, image_msg, det_msg):
        """Process synchronized image and detection messages.
        
        Args:
            image_msg: ROS image message
            det_msg: Detection array message
        """
        # Use ROS time instead of system time for consistent timing
        callback_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.callback_count += 1
        
        # Calculate message ages and synchronization delay
        img_msg_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        det_msg_time = det_msg.header.stamp.sec + det_msg.header.stamp.nanosec * 1e-9
        img_age = (callback_start_time - img_msg_time) * 1000
        det_age = (callback_start_time - det_msg_time) * 1000
        sync_diff = abs(img_msg_time - det_msg_time) * 1000
        
        # Calculate callback interval
        callback_interval = 0
        if self.last_callback_time is not None:
            callback_interval = (callback_start_time - self.last_callback_time) * 1000
        self.last_callback_time = callback_start_time
        
        # Log synchronization issues - much more selective
        if img_age > 1000000 or det_age > 1000000:  # More than 1000 seconds old
            if self.callback_count % 100 == 1:  # Much less frequent
                self.get_logger().error(
                    f"[TIMESTAMP ERROR] Callback #{self.callback_count}: "
                    f"Img_age={img_age/1000:.1f}s, Det_age={det_age/1000:.1f}s (CLOCK ISSUE!)"
                )
        elif (callback_interval > 1000 or img_age > 5000 or det_age > 5000 or sync_diff > 1000):  # Much higher thresholds
            self.sync_issues_count += 1  # Track sync issues for summary
            if self.callback_count % 100 == 1:  # Only log every 100th severe issue
                self.get_logger().warn(
                    f"[SYNC ISSUE] Callback #{self.callback_count}: "
                    f"Interval={callback_interval:.1f}ms, "
                    f"Img_age={img_age:.1f}ms, "
                    f"Det_age={det_age:.1f}ms, "
                    f"Sync_diff={sync_diff:.1f}ms"
                )
        
        try:
            # Timing: Image loading and preprocessing
            # Use ROS time for all timing measurements
            img_start_time = self.get_clock().now().nanoseconds * 1e-9
            
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
            
            img_end_time = self.get_clock().now().nanoseconds * 1e-9
            img_processing_time = (img_end_time - img_start_time) * 1000  # Convert to ms
            
            if len(dets):
                # Increment frame counter for cleanup tracking
                self.processed_frames += 1
                
                # Timing: Motion estimation
                motion_start_time = self.get_clock().now().nanoseconds * 1e-9
                
                # Update motion estimation
                if self.use_cuda:
                    motion = self.motion_estimator.estimate_motion_cuda(cv_image, self.prev_frame)
                else:
                    motion = self.motion_estimator.estimate_motion(cv_image, self.prev_frame)

                self.prev_frame = cv_image.copy()
                
                motion_end_time = self.get_clock().now().nanoseconds * 1e-9
                motion_time = (motion_end_time - motion_start_time) * 1000  # Convert to ms
                
                # Timing: Tracking update
                tracking_start_time = self.get_clock().now().nanoseconds * 1e-9
                
                # Limit number of detections to prevent tracker overload
                if len(dets) > 100:  # Arbitrary limit to prevent performance issues
                    # Keep only the highest confidence detections
                    confidence_scores = dets[:, 4]
                    top_indices = np.argsort(confidence_scores)[-100:]  # Top 100
                    dets = dets[top_indices]
                
                # Update tracker
                self.online_targets = self.tracker.update(
                    dets,
                    None,
                    odom_vx=motion.translation_x,
                    odom_vy=motion.translation_y,
                    odom_uncertainty=(motion.uncertainty_x, motion.uncertainty_y)
                )
                
                # Limit number of active tracks to prevent memory accumulation
                if len(self.online_targets) > self.max_tracks:
                    # Keep only the most recent tracks
                    self.online_targets = self.online_targets[-self.max_tracks:]
                    self.get_logger().warn(f"[CLEANUP] Limited tracks to {self.max_tracks}")
                
                tracking_end_time = self.get_clock().now().nanoseconds * 1e-9
                tracking_time = (tracking_end_time - tracking_start_time) * 1000  # Convert to ms
                
                # Timing: Publishing
                publish_start_time = self.get_clock().now().nanoseconds * 1e-9
                
                # Publish tracked results
                self.publish_tracks(self.online_targets, image_msg.header)
                
                if self.visualization:
                    # Draw and publish visualization
                    viz_img = draw_tracks(
                        orig_image, 
                        self.online_targets,
                        draw_history=True,
                        active_color=(0, 255, 0),
                        history_color=(255, 0, 255)
                    )
                    self.publish_visualization(viz_img, image_msg.header)
                
                publish_end_time = self.get_clock().now().nanoseconds * 1e-9
                publish_time = (publish_end_time - publish_start_time) * 1000  # Convert to ms
                
                # Calculate total time
                total_time = (publish_end_time - callback_start_time) * 1000  # Convert to ms
                self.total_processing_time += total_time  # Track for summary
                
                # Log timing summary - much less frequent
                if self.callback_count % 50 == 1 or total_time > 200:  # Every 50th callback or if really slow
                    self.get_logger().warn(
                        f"[TIMING] #{self.callback_count}: "
                        f"Img={img_processing_time:.1f}ms, "
                        f"Motion={motion_time:.1f}ms, "
                        f"Track={tracking_time:.1f}ms, "
                        f"Pub={publish_time:.1f}ms, "
                        f"Total={total_time:.1f}ms "
                        f"({'CUDA' if self.use_cuda else 'CPU'}) "
                        f"Dets={len(dets)}, Tracks={len(self.online_targets)}"
                    )
            else:
                # No detections case - much less frequent logging
                total_time = (self.get_clock().now().nanoseconds * 1e-9 - callback_start_time) * 1000
                if self.callback_count % 100 == 1:  # Every 100th callback
                    self.get_logger().warn(
                        f"[NO DETS] #{self.callback_count}: "
                        f"Img={img_processing_time:.1f}ms, "
                        f"Total={total_time:.1f}ms"
                    )
                    
        except Exception as e:
            total_time = (self.get_clock().now().nanoseconds * 1e-9 - callback_start_time) * 1000
            self.get_logger().error(f"[ERROR] After {total_time:.1f}ms: {str(e)}")

    def synchronized_callback_debug(self, image_msg, det_msg):
        """Synchronized callback for image and detection messages."""
        img_header = image_msg.header
        det_header = det_msg.header
        self.publish_tracks_debug(img_header)
        
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
        
    def publish_tracks_debug(self, header):
        """Publish tracked keypoints (relative to full image).
        Args:
            tracks: List of track objects (coordinates relative to cropped image)
            header: ROS message header
        """
        msg = Keypoint2DArray()
        msg.header = header
          # Example keypoint coordinates in camera image coordinates

        for i, kpt in enumerate(self.kpt_coords):
            det = Detection2D()
            # Fetch keypoints from laser pointer debugger tool
            keypoint = Keypoint2D()
            keypoint.position.x = float(kpt[0]) # x value   in camera image coordinates
            keypoint.position.y = float(kpt[1]) # y value   in camera image coordinates
            keypoint.confidence = 1.0  # Tracked points are considered confident
            
            # Put dummy detection box
            det.bbox.center.position.x = 100.0
            det.bbox.center.position.y = 100.0
            det.bbox.size_x = 10.0
            det.bbox.size_y = 10.0
            det.id = str(i)  # Use the index as the track ID
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
        
    def log_performance_summary(self):
        """Log periodic performance summary instead of spamming individual messages."""
        # Use ROS time for consistent timing
        current_time = self.get_clock().now().nanoseconds * 1e-9
        time_since_last = current_time - self.last_summary_time
        
        if self.callback_count > 0:
            callback_rate = self.callback_count / time_since_last if time_since_last > 0 else 0
            avg_processing_time = self.total_processing_time / self.callback_count if self.callback_count > 0 else 0
            sync_issue_rate = (self.sync_issues_count / self.callback_count * 100) if self.callback_count > 0 else 0
            
            self.get_logger().info(
                f"[SUMMARY] {time_since_last:.1f}s: "
                f"Callbacks: {self.callback_count} ({callback_rate:.1f} Hz), "
                f"Avg processing: {avg_processing_time:.1f}ms, "
                f"Sync issues: {self.sync_issues_count} ({sync_issue_rate:.1f}%), "
                f"Config: CUDA={self.use_cuda}, Queue={self.queue_size}"
            )
        
        # Reset counters for next period
        self.callback_count = 0
        self.sync_issues_count = 0
        self.total_processing_time = 0
        self.last_summary_time = current_time
        
    def periodic_cleanup(self):
        """Perform periodic cleanup to prevent memory accumulation."""
        
        # Get memory usage if psutil is available
        memory_mb = 0
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
            except:
                memory_mb = 0
        
        # Reset tracker periodically to prevent memory accumulation
        if self.processed_frames > self.tracker_reset_interval:
            self.get_logger().warn(f"[CLEANUP] Resetting tracker after {self.processed_frames} frames. Memory: {memory_mb:.1f}MB")
            
            # Reinitialize tracker to clear accumulated state
            self.tracker = BYTETracker(
                track_thresh=self.get_parameter("tracking.track_thresh").value,
                track_buffer=self.get_parameter("tracking.track_buffer").value,
                match_thresh=self.get_parameter("tracking.match_thresh").value,
                frame_rate=self.get_parameter("tracking.frame_rate").value,
                odom_std_weight=self.get_parameter("tracking.odom_std_weight").value
            )
            
            # Reset frame counter
            self.processed_frames = 0
            
            # Force garbage collection
            gc.collect()
            
        # Log memory usage
        if PSUTIL_AVAILABLE:
            if memory_mb > 500:  # Log if using more than 500MB
                self.get_logger().warn(f"[MEMORY] High memory usage: {memory_mb:.1f}MB, Frames: {self.processed_frames}")
            elif self.processed_frames % 100 == 0:  # Log every 100 frames
                self.get_logger().info(f"[MEMORY] Current usage: {memory_mb:.1f}MB, Frames: {self.processed_frames}")
        else:
            if self.processed_frames % 200 == 0:  # Log every 200 frames if no psutil
                self.get_logger().info(f"[MEMORY] psutil not available, Frames: {self.processed_frames}")

def main():
    rclpy.init()
    node = MotionTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
