import os
import time

import cv2
import imutils
import numpy as np
import pycuda.driver as cuda
import rclpy
import tensorrt as trt
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

# message definitions
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    ObjectHypothesisWithPose,
)
from lalweco_perception_msgs.msg import Keypoint2D, Keypoint2DArray
# Remove yolox import and use local bytetracker package
from bytetracker.byte_tracker import BYTETracker

from utils import (
    non_max_suppression_v8,
    plot,
    remove_overlapping_boxes,
    scale_boxes,
    scale_coords,
    xywh2xyxy,
)
from utils.trt_utils import HostDeviceMem, TrtLogger, _cuda_error_check

classes = ["background", "crop", "weed"]
kpt_shape = (1, 3)


class CropKeypointDetector(Node):
    def __init__(self, mode="fp32", topic="/realsenseD435/color/image_raw/compressed"):
        super().__init__("CropKeypointDetector")
        self.declare_parameter("operation_mode", "detection")
        self.operation_mode = (
            self.get_parameter("operation_mode").get_parameter_value().string_value
        )
        self.get_logger().info(f"Operating in {self.operation_mode} mode")
        # if self.operation_mode == 'detection':
        self.publisher_array = self.create_publisher(
            Keypoint2DArray, "/inference/Keypoint2DDetArray", 10
        )
        # elif self.operation_mode == 'image':
        self.publisher_image = self.create_publisher(
            Image, "/inference/detection_image", 10
        )
        if "compressed" in topic:
            self.compressed = True
            self.subscription = self.create_subscription(
                CompressedImage, topic, self.listener_callback, 10
            )
        else:
            self.compressed = False
            self.subscription = self.create_subscription(
                Image, topic, self.listener_callback, 10
            )
        self.get_logger().info("Subscribing to {}".format(topic))
        self.ros_logger = self.get_logger()
        self.ros_logger = self.get_logger()
        self.trt_logger = TrtLogger(self)
        self.class_ids = {0: "weeds", 1: "crop"}
        # Initialize tracker with odometry support
        self.tracker = BYTETracker(
            track_thresh=0.3,
            track_buffer=30,
            match_thresh=0.9,
            frame_rate=5,
            odom_std_weight=1.0/40  # Adjust this based on your odometry noise characteristics
        )
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        # NOTE! self.context is not allowed since the Node parent has a ROS2 related context which cannot be overridden.
        self.trt_context = None
        self.init_model(mode=mode)
        
        # Add previous frame storage
        self.prev_frame = None

    def get_logger(self):
        # Override get_logger to use ROS 2 logger
        return super().get_logger()

    def listener_callback(self, msg):
        if self.compressed:
            self.cv_image = CvBridge().compressed_imgmsg_to_cv2(msg)
        else:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg)
        if self.cv_image.shape[2] != 3:
            self.cv_image = self.cv_image[:, :, :3]
        # Crop the image to the region of interest
        self.cv_image = self.cv_image[:800, 360:1160]
        self.header = msg.header
        # TODO: Remove after DEBUG
        # self.cv_image = cv2.imread('./sample.png')
        # self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.orig = self.cv_image.astype(np.uint8)
        try:
            t1 = time.time()
            self.input_image = self.preprocess_image(self.cv_image)
            t2 = time.time()
            preprocess_time = round((t2 - t1) * 1000, 2)
            # inference
            outputs = self.infer_trt(self.input_image)
            t3 = time.time()
            inference_time = round((t3 - t2) * 1000, 2)
            if outputs is not None:
                self.postprocess_image(outputs)
            t4 = time.time()
            post_process_time = round((t4 - t3) * 1000, 2)
            total = preprocess_time + inference_time + post_process_time
            self.ros_logger.info(
                "Preprocessing: {} ms Inference: {} ms Postprocessing {} ms FPS: {}".format(
                    preprocess_time,
                    inference_time,
                    post_process_time,
                    round(1 / (total / 1000), 2),
                )
            )
        except KeyboardInterrupt:
            self.get_logger().loginfo("Callback interrupted, cleaning up CUDA context")
            self.cuda_ctx.pop()

    def calculate_motion_from_homography(self, curr_frame, prev_frame):
        """Calculate frame-to-frame motion using fast feature detection and matching"""
        if prev_frame is None:
            return 0.0, -40.0, (10.0, 10.0)
            
        # Convert images to grayscale and downscale for speed
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Max image size
        max_size = 512
        scale_factor = max(gray1.shape[0] / max_size, gray1.shape[1] / max_size)
        # Optionally downscale images for even more speed
        gray1 = cv2.resize(gray1, (int(gray1.shape[1] / scale_factor), int(gray1.shape[0] / scale_factor)))
        gray2 = cv2.resize(gray2, (int(gray2.shape[1] / scale_factor), int(gray2.shape[0] / scale_factor)))
        
        # Use FAST feature detector instead of SIFT
        fast = cv2.FastFeatureDetector_create(threshold=20)
        kp1 = fast.detect(gray1, None)
        kp2 = fast.detect(gray2, None)
        
        # Convert keypoints to numpy arrays for faster processing
        if len(kp1) < 10 or len(kp2) < 10:
            return 0.0, -40.0, (10.0, 10.0)
        
        # Use ORB for faster feature description
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.compute(gray1, kp1)
        kp2, des2 = orb.compute(gray2, kp2)
        
        if des1 is None or des2 is None:
            return 0.0, -40.0, (10.0, 10.0)
        
        # Use Brute Force matcher with Hamming distance for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        try:
            matches = bf.match(des1, des2)
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)[:50]  # Keep only top 50 matches
        except Exception as e:
            self.get_logger().warning(f"Matching failed: {e}")
            return 0.0, -40.0, (10.0, 10.0)
        
        if len(matches) >= 10:
            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography matrix with lower precision requirements
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0, maxIters=200)
            
            if H is not None:
                # Extract translation from homography
                tx = H[0,2]
                ty = H[1,2]

                tx = tx * scale_factor
                ty = ty * scale_factor

                # Simple uncertainty based on number of matches
                uncertainty = (20.0 / len(matches), 20.0 / len(matches))
                
                return tx, ty, uncertainty
        
        return 0.0, -40.0, (10.0, 10.0)

    # CUDA VERSION: Uncomment this method and comment the CPU version below
    # def calculate_motion_from_homography(self, curr_frame, prev_frame):
    #     """Calculate frame-to-frame motion using GPU accelerated feature detection and matching"""
    #     if prev_frame is None:
    #         return 0.0, 0.0, (10.0, 10.0)

    #     # Upload frames to GPU
    #     gpu_prev = cv2.cuda_GpuMat()
    #     gpu_curr = cv2.cuda_GpuMat()
    #     gpu_prev.upload(prev_frame)
    #     gpu_curr.upload(curr_frame)
        
    #     # Convert to grayscale on GPU
    #     gpu_gray1 = cv2.cuda.cvtColor(gpu_prev, cv2.COLOR_BGR2GRAY)
    #     gpu_gray2 = cv2.cuda.cvtColor(gpu_curr, cv2.COLOR_BGR2GRAY)

    #     # Optionally resize on GPU
    #     max_size = 512
    #     scale_factor = max(gpu_gray1.size()[1] / max_size, gpu_gray1.size()[0] / max_size)
    #     new_size = (int(gpu_gray1.size()[1] / scale_factor), int(gpu_gray1.size()[0] / scale_factor))
    #     gpu_gray1 = cv2.cuda.resize(gpu_gray1, new_size)
    #     gpu_gray2 = cv2.cuda.resize(gpu_gray2, new_size)

    #     # Detect features using GPU FAST
    #     fast_gpu = cv2.cuda_FastFeatureDetector_create(threshold=20)
    #     kp1_gpu = fast_gpu.detect(gpu_gray1, None)
    #     kp2_gpu = fast_gpu.detect(gpu_gray2, None)

    #     if len(kp1_gpu) < 10 or len(kp2_gpu) < 10:
    #         return 0.0, 0.0, (10.0, 10.0)

    #     # Compute descriptors using GPU ORB
    #     orb_gpu = cv2.cuda_ORB_create(nfeatures=500)
    #     kp1_gpu, des1_gpu = orb_gpu.compute(gpu_gray1, kp1_gpu)
    #     kp2_gpu, des2_gpu = orb_gpu.compute(gpu_gray2, kp2_gpu)
        
    #     if des1_gpu is None or des2_gpu is None:
    #         return 0.0, 0.0, (10.0, 10.0)
        
    #     # Use GPU BFMatcher with Hamming distance
    #     bf_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
    #     matches_gpu = bf_gpu.match(des1_gpu, des2_gpu)
    #     matches = sorted(matches_gpu, key=lambda x: x.distance)[:50]

    #     if len(matches) >= 10:
    #         # Convert keypoints back to CPU arrays for homography computation
    #         src_pts = np.float32([kp1_gpu[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #         dst_pts = np.float32([kp2_gpu[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    #         # Compute homography on CPU (or use GPU-based method if available)
    #         H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0, maxIters=200)
            
    #         if H is not None:
    #             tx = H[0, 2] * scale_factor
    #             ty = H[1, 2] * scale_factor
    #             uncertainty = (20.0 / len(matches), 20.0 / len(matches))
    #             return tx, ty, uncertainty

    #     return 0.0, 0.0, (10.0, 10.0)


    def init_model(self, mode="fp32"):
        """Initialize the model for inference"""
        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        self.engine_path = os.path.join(
            os.getenv('MODEL_PATH', '/home/docker/ros2_ws/src/inference_ros2/model'),
            "yolov8-keypoint-det-cropweed-nuc-{}-23.10.engine".format(mode)
        )
        self.runtime = trt.Runtime(self.trt_logger)
        trt.init_libnvinfer_plugins(None, "")
        assert os.path.exists(
            self.engine_path
        ), f"Engine file not found at path: \n {self.engine_path}"
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if mode == "fp32":
            self.dtype = np.float32
        elif mode == "fp16":
            self.dtype = np.float16
        elif mode == "int8":
            self.dtype = np.int8
        else:
            self.dtype = np.float32
        self.confidence = 0.5
        self.batch_size = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.trt_context = self.engine.create_execution_context()
        assert self.engine
        assert self.trt_context
        self.setup_IO_binding()

    def infer_trt(self, input_image: np.ndarray):
        """
        input_image: A numpy array that is contiguous, preprocessed and in the data type expected by the self.data_type variable
        """
        self.input_image = self.input_image.ravel()
        # NOTE! Copy current image to the host buffer i.e. self.input.host
        np.copyto(self.input.host, self.input_image)
        # Process I/O and execute the network.
        self.cuda_ctx.push()
        # Copy current image from host to device memory ----> part of IO operations
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        # t2 = time.time()
        # io_time = round((t2-t1)*1000, 2)
        err = self.trt_context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        # err = self.trt_context.execute_v2(bindings=self.bindings)
        self.cuda_ctx.pop()
        predictions = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            # out.host = out.host.reshape()
            predictions.append(out.host.reshape(self.output_shape))
        self.stream.synchronize()

        return predictions

    def setup_IO_binding(self):
        self.outputs = []
        self.bindings = []
        self.allocations = []

        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            shape = self.engine.get_tensor_shape(binding_name)
            size = trt.volume(shape)

            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            # Handle input/output separately
            if self.engine.binding_is_input(binding_idx):
                self.input = HostDeviceMem(host_mem, device_mem)
                self.input_shape = shape
                self.input_dtype = dtype
                self.trt_context.set_binding_shape(binding_idx, shape)
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.output_shape = shape

        self.stream = cuda.Stream()
        assert len(self.outputs) > 0
        assert len(self.bindings) > 0

    def preprocess_image(self, image):
        """Preprocess the image for inference"""
        self.orig_shape = image.shape
        # resize and pad to square the resolution
        b, c, h, w = self.input_shape
        img_0 = np.zeros((h, w, 3), dtype=np.float32)
        image = imutils.resize(image, width=w)
        self.p_h, self.p_w = image.shape[0], image.shape[1]
        img_0[0 : image.shape[0], 0 : image.shape[1], :] = image
        image = img_0.copy()
        image = image.transpose(2, 0, 1)  # HWC to CHW
        # normalize the image
        image = image / 255.0
        # add batch dimension
        image = np.expand_dims(image, axis=0)  # BCHW
        image = np.ascontiguousarray(image.astype(np.dtype(self.dtype))).ravel()
        return image

    def postprocess_image(self, output):
        """Postprocess the model output for publishing"""
        det = output[0]
        preds = non_max_suppression_v8(
            prediction=det,
            conf_thres=0.5,
            iou_thres=0.4,
            max_det=200,
            #    multi_label=True,
            nc=len(classes),
        )[0]
        # The preds are in xtl,ytl,w,h format
        remove_indices = np.where(np.logical_or(preds[:, 5] == 1, preds[:, 5] == 0))[
            0
        ]  # TODO remove after DEBUGGING
        preds = np.delete(preds, remove_indices, axis=0)  # TODO remove after DEBUGGING
        preds[:, :4] = scale_boxes(
            preds[:, :4], (self.p_h, self.p_w), self.orig_shape, padded=True
        )
        preds[:, :4] = xywh2xyxy(preds[:, :4])
        keep_indices = remove_overlapping_boxes(preds[:, :4], iou_threshold=0.8)
        preds = preds[keep_indices, :]
        pred_kpts = (
            preds[:, 6:].view(len(preds), *kpt_shape) if len(preds) else preds[:, 6:]
        )  # TODO Fetch keypoint shape from model dynamically
        pred_kpts = scale_coords((self.p_h, self.p_w), pred_kpts, self.orig_shape)
        # The tracker expects them in xtl,ytl,xbr,ybr format.
        
        # Apply NMS to keypoints before tracking
        if preds.shape[0] != 0:
            keypoints = pred_kpts[:, 0, :2]  # N x 2 array of x,y coordinates
            confidences = pred_kpts[:, 0, 2]  # N array of confidence scores
            
            # Apply NMS to keypoints
            keep_indices = keypoint_nms(keypoints, confidences, distance_threshold=30)
            
            # Filter predictions and keypoints based on NMS results
            preds = preds[keep_indices]
            pred_kpts = pred_kpts[keep_indices]

        # Calculate motion from homography
        t_motion_start = time.time()
        odom_vx, odom_vy, odom_uncertainty = self.calculate_motion_from_homography(
            self.cv_image, self.prev_frame
        )
        motion_time = round((time.time() - t_motion_start) * 1000, 2)
        self.get_logger().info(f"Motion calculation time: {motion_time} ms")
        print(f"Motion: {odom_vx}, {odom_vy}, {odom_uncertainty}")
        # Store current frame as previous
        self.prev_frame = self.cv_image.copy()

        # Update tracker with calculated odometry
        self.online_targets = self.tracker.update(
            preds,
            None,
            odom_vx=odom_vx,
            odom_vy=odom_vy,
            odom_uncertainty=odom_uncertainty
        )
        # Original plotting code for detections
        if preds.shape[0] != 0:
            self.orig = plot(preds, pred_kpts, self.orig, mode="det")
            
        # Modified plotting code for tracks
        if len(self.online_targets) != 0:

            # Draw boxes, IDs, Kalman points, and track history
            for track in self.tracker.tracked_stracks:
                if track.is_activated:
                    x1, y1, x2, y2 = track._detection.astype(int)
                    kx, ky = track.keypoint.astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(self.orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw track ID
                    cv2.putText(self.orig, f'ID: {int(track.track_id)}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    # Draw Kalman-filtered point
                    cv2.circle(self.orig, (kx, ky), 4, (255, 0, 255), -1)  # Magenta dot
                    
                    # Draw track history
                    if len(track.track_history) > 1:
                        # Convert history points to integer array
                        points = np.array(track.track_history, dtype=np.int32)
                        # Draw lines connecting history points
                        cv2.polylines(self.orig, [points], False, (255, 0, 255), 2)
                        # Draw history points as small black dots
                        for point in points:
                            cv2.circle(self.orig, tuple(point), 3, (0, 255, 255), -1)

            # Optionally visualize lost tracks with different color
            for track in self.tracker.lost_stracks:
                if len(track.track_history) > 1:
                    points = np.array(track.track_history, dtype=np.int32)
                    # Draw lost track history in a different color (e.g., gray)
                    overlay = self.orig.copy()
                    cv2.polylines(overlay, [points], False, (128, 128, 128), 1)
                    cv2.addWeighted(overlay, 0.3, self.orig, 0.7, 0, self.orig)

        if preds.shape[0] != 0:
            # Extract keypoints and their confidences
            keypoints = pred_kpts[:, 0, :2]  # N x 2 array of x,y coordinates
            confidences = pred_kpts[:, 0, 2]  # N array of confidence scores
            
            # Apply NMS to keypoints
            keep_indices = keypoint_nms(keypoints, confidences, distance_threshold=30)
            
            # Filter predictions and keypoints based on NMS results
            preds = preds[keep_indices]
            pred_kpts = pred_kpts[keep_indices]
            
            keypoint_msg = Keypoint2DArray()
            keypoint_msg.header.stamp = self.header.stamp
            keypoint_msg.header.frame_id = self.header.frame_id
            obj = ObjectHypothesisWithPose()
            for i, kpt_idx in zip(range(preds.shape[0]), range(pred_kpts.shape[0])):
                bbox = BoundingBox2D()
                detection = Detection2D()
                bbox.center.position.x = preds[i, 0].item()
                bbox.center.position.y = preds[i, 1].item()
                bbox.size_x = preds[i, 2].item()
                bbox.size_y = preds[i, 3].item()
                obj = ObjectHypothesisWithPose()
                obj.hypothesis.class_id = classes[int(preds[i, 5].item())]
                obj.hypothesis.score = np.round(preds[i, 4].item(), 2)
                detection.bbox = bbox
                detection.results.append(obj)
                detection.id = str(0)  # TODO add tracking IDs here.
                keypoint = Keypoint2D()
                keypoint.position.x = pred_kpts[kpt_idx, 0, 0].item()
                keypoint.position.y = pred_kpts[kpt_idx, 0, 1].item()
                keypoint.confidence = pred_kpts[kpt_idx, 0, 2].item()
                keypoint_msg.keypoints.append(keypoint)
                keypoint_msg.detections.append(detection)
            self.publisher_array.publish(keypoint_msg)
            preds = []
            det = []
        else:
            keypoint_msg = Keypoint2DArray()
            keypoint_msg.header.stamp = self.header.stamp
            keypoint_msg.header.frame_id = self.header.frame_id
            self.publisher_array.publish(keypoint_msg)
            preds = []
            det = []
        # elif self.operation_mode == 'image':
        processed_image = self.orig
        img_msg = CvBridge().cv2_to_imgmsg(processed_image, encoding="bgr8")
        img_msg.header.stamp = self.header.stamp
        img_msg.header.frame_id = self.header.frame_id
        self.publisher_image.publish(img_msg)


def keypoint_nms(keypoints, confidences, distance_threshold=30):
    """
    Apply non-maximum suppression to keypoints based on their spatial distance and confidence.
    
    Args:
        keypoints: numpy array of shape (N, 2) containing x,y coordinates
        confidences: numpy array of shape (N,) containing confidence scores
        distance_threshold: maximum distance between keypoints to be considered for suppression
    
    Returns:
        keep_indices: indices of keypoints to keep after NMS
    """
    if len(keypoints) == 0:
        return []
        
    keep_indices = []
    
    # Convert to numpy arrays if not already
    keypoints = np.array(keypoints)
    confidences = np.array(confidences)
    
    # Get indices sorted by confidence
    order = confidences.argsort()[::-1]
    
    while order.size > 0:
        # Keep the current highest confidence keypoint
        i = order[0]
        keep_indices.append(i)
        
        if order.size == 1:
            break
            
        # Calculate distances between the current keypoint and all others
        current_point = keypoints[i]
        other_points = keypoints[order[1:]]
        distances = np.sqrt(np.sum((other_points - current_point) ** 2, axis=1))
        
        # Find points that are far enough from the current point
        far_enough = distances > distance_threshold
        
        # Update order by removing nearby points
        order = order[1:][far_enough]
    
    return keep_indices


def main(args=None):
    rclpy.init(args=args)
    node = CropKeypointDetector(
        topic="/sensors/zed_r/zed_node/rgb/image_rect_color",
        mode="fp32",
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
