#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import BoundingBox2D, Detection2D, ObjectHypothesisWithPose
from lalweco_perception_msgs.msg import Keypoint2D, Keypoint2DArray

from ..core.vision.processor import TensorRTProcessor
from ..core.vision.utils import (
    preprocess_image,
    scale_boxes,
    scale_coords,
    xywh2xyxy,
    keypoint_nms,
    remove_overlapping_boxes,
    non_max_suppression
)
from ..utils.visualization import draw_detections

class KeypointDetectorNode(Node):
    """ROS2 node for keypoint detection and tracking."""
    
    def __init__(self):
        """Initialize the node."""
        super().__init__("keypoint_detector_node") # This has to match the namespace in the config/params.yaml
        
        # Declare parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("operation_mode", rclpy.Parameter.Type.STRING),
                ("image_topic", rclpy.Parameter.Type.STRING),
                ("model_precision", rclpy.Parameter.Type.STRING),
                ("model_path", rclpy.Parameter.Type.STRING),
                ("confidence_threshold", rclpy.Parameter.Type.DOUBLE),
                ("iou_threshold", rclpy.Parameter.Type.DOUBLE),
                ("max_detections", rclpy.Parameter.Type.INTEGER),
                ("roi.height", rclpy.Parameter.Type.INTEGER),
                ("roi.x_min", rclpy.Parameter.Type.INTEGER),
                ("roi.x_max", rclpy.Parameter.Type.INTEGER)
            ]
        )
        
        # Get parameters
        self.operation_mode = self.get_parameter("operation_mode").value
        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.conf_thresh = self.get_parameter("confidence_threshold").value
        self.iou_thresh = self.get_parameter("iou_threshold").value
        self.max_det = self.get_parameter("max_detections").value
        self.roi = {
            "height": self.get_parameter("roi.height").value,
            "x_min": self.get_parameter("roi.x_min").value,
            "x_max": self.get_parameter("roi.x_max").value
        }
        
        # Initialize components
        self.bridge = CvBridge()
        self.processor = TensorRTProcessor(
            self.model_path,
            precision=self.get_parameter("model_precision").value
        )
        
        # Set up publishers
        self.det_pub = self.create_publisher(
            Keypoint2DArray,
            "/inference/Keypoint2DDetArray",
            10
        )
        self.viz_pub = self.create_publisher(
            Image,
            "/inference/detection_image",
            10
        )
        
        # Set up subscriber
        topic_type = CompressedImage if "compressed" in self.image_topic else Image
        self.sub = self.create_subscription(
            topic_type,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(
            f"Initialized keypoint detector node in {self.operation_mode} mode"
        )
        self.classes = ["crop", "weed"]
        self.kpt_shape = (1, 3)
        
    def image_callback(self, msg):
        """Process incoming image messages.
        
        Args:
            msg: ROS image message
        """
        try:
            # Convert message to OpenCV image
            if isinstance(msg, CompressedImage):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg)
                
            if cv_image.shape[2] != 3:
                cv_image = cv_image[:, :, :3]
            
            # # Apply gamma correction for contrast enhancement
            # gamma = 1.3  # Adjust gamma value as needed
            # cv_image = np.power(cv_image / 255.0, gamma) * 255.0
            # cv_image = cv_image.astype(np.uint8)

            # Apply ROI cropping
            cv_image = cv_image[:self.roi["height"],
                              self.roi["x_min"]:self.roi["x_max"]]
            
            # Store original image
            orig_image = cv_image.copy()
            
            # Preprocess image
            input_data, pad_shape = preprocess_image(
                cv_image,
                self.processor.input_shape,
                self.processor.dtype
            )
            
            # Run inference
            outputs = self.processor.infer(input_data)
            
            if outputs:
                # Post-process detections
                det = outputs[0]
                preds = self.process_detections(det, pad_shape, orig_image.shape)
                
                # # TODO ONLY FOR DEBUGGING WE REMOVE THE CROP
                # preds = preds[preds[:, 5] != 1]

                # Publish results
                self.publish_detections(preds, msg.header)
                
                if self.operation_mode == "detection":
                    # Draw and publish visualization
                    viz_img = draw_detections(
                        orig_image,
                        preds,
                        preds[:, 6:].reshape(len(preds), *self.kpt_shape),
                        self.classes
                    )
                    self.publish_visualization(viz_img, msg.header)
                    
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            
    def process_detections(
        self,
        det: np.ndarray,
        pad_shape: tuple,
        orig_shape: tuple
    ) -> np.ndarray:
        """Process raw detections from model.
        
        Args:
            det: Raw detection array
            pad_shape: Shape of padded image
            orig_shape: Shape of original image
            
        Returns:
            Processed detections
        """
        # Apply NMS
        preds = det[0]  # First batch item
        preds = non_max_suppression(
            predictions=det,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            max_det=self.max_det,
            #    multi_label=True,
            nc=len(self.classes),
        )[0]
        preds = preds[preds[:, 4] > self.conf_thresh]
        
        # Scale boxes to original image size
        if len(preds):
            preds[:, :4] = scale_boxes(preds[:, :4], pad_shape, orig_shape)
            preds[:, :4] = xywh2xyxy(preds[:, :4])
            
            # Remove overlapping boxes
            keep = remove_overlapping_boxes(preds[:, :4], self.iou_thresh)
            preds = preds[keep]
            
            # Scale keypoints
            kpts = preds[:, 6:].reshape(len(preds), *self.kpt_shape)
            kpts = scale_coords(pad_shape, kpts, orig_shape) # orig_shape is the cropped shape here
            
            # --- Add ROI offset ---
            kpts[:, :, 0] += self.roi["x_min"] # Offset X coordinate
            # ----------------------
            
            preds[:, 6:] = kpts.reshape(len(preds), -1)
            
            # Apply keypoint NMS
            # Use the original keypoint coordinates (before offset) for NMS if needed,
            # or apply NMS before offsetting. Here we use offsetted coords.
            keypoints = kpts[:, 0, :2] 
            confidences = kpts[:, 0, 2]
            keep = keypoint_nms(keypoints, confidences)
            preds = preds[keep]
            
        return preds
        
    def publish_detections(self, preds: np.ndarray, header):
        """Publish detection results.
        
        Args:
            preds: Processed detection array
            header: ROS message header
        """
        msg = Keypoint2DArray()
        msg.header = header
        
        if len(preds):
            for i, pred in enumerate(preds):
                det = Detection2D()
                bbox = BoundingBox2D()
                
                # Set bounding box (apply offset)
                bbox.center.position.x = (pred[0].item() + pred[2].item()) / 2 + self.roi["x_min"] # Center x offset
                bbox.center.position.y = (pred[1].item() + pred[3].item()) / 2  # Center y (no offset if y=0)
                bbox.size_x = pred[2].item() - pred[0].item()  # Width
                bbox.size_y = pred[3].item() - pred[1].item()  # Height
                
                # Set class info
                obj = ObjectHypothesisWithPose()
                obj.hypothesis.class_id = self.classes[int(pred[5])]
                obj.hypothesis.score = float(pred[4])
                
                det.bbox = bbox
                det.results.append(obj)
                det.id = str(i)
                
                # Set keypoint (already offset in process_detections)
                kpt = pred[6:9]  # First keypoint
                keypoint = Keypoint2D()
                keypoint.position.x = float(kpt[0]) # Already offset
                keypoint.position.y = float(kpt[1]) # No y-offset needed if roi starts at y=0
                keypoint.confidence = float(kpt[2])
                
                msg.keypoints.append(keypoint)
                msg.detections.append(det)
                
        self.det_pub.publish(msg)
        
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
    node = KeypointDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
