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
    Keypoint2D,
    Keypoint2DArray,
    ObjectHypothesisWithPose,
)
from yolox.tracker.byte_tracker import BYTETracker

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
        self.tracker = BYTETracker(args=None, class_dict=self.class_ids)
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        # NOTE! self.context is not allowed since the Node parent has a ROS2 related context which cannot be overridden.
        self.trt_context = None
        self.init_model(mode=mode)

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

    def init_model(self, mode="fp32"):
        """Initialize the model for inference"""
        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        self.engine_path = os.path.join(
            "/root/ros2_ws/src/inference_ros2/model/yolov8-keypoint-det-cropweed-nuc-{}-23.10-800.engine".format(
                mode
            )
        )
        # self.logger = trt.Logger(self.trt_logger)
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
        preds[:, :4] = xywh2xyxy(preds[:, :4])
        keep_indices = remove_overlapping_boxes(preds[:, :4], iou_threshold=0.8)
        preds = preds[keep_indices, :]
        pred_kpts = (
            preds[:, 6:].view(len(preds), *kpt_shape) if len(preds) else preds[:, 6:]
        )  # TODO Fetch keypoint shape from model dynamically
        pred_kpts = scale_coords((self.p_h, self.p_w), pred_kpts, self.orig_shape)
        # The tracker expects them in xtl,ytl,xbr,ybr format.
        # self.online_targets = self.tracker.update(preds, self.orig_shape, self.orig_shape)
        if preds.shape[0] != 0:
            self.orig = plot(preds, pred_kpts, self.orig, mode="det")
        # if len(self.online_targets) != 0:
        #     self.orig = plot(self.online_targets, None, self.orig, mode='track')
        # cv2.imwrite('prediction.jpg', self.orig)
        # cv2.imshow('predictions', self.orig)
        # cv2.waitKey(1)
        # plot(preds, pred_kpts, self.cv_image.astype(np.uint8))
        # if self.operation_mode == 'detection':
        if preds.shape[0] != 0:
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


def main(args=None):
    rclpy.init(args=args)
    node = CropKeypointDetector(
        topic="/sensors/zed_laser_module/zed_node/rgb/image_rect_color/compressed",
        mode="fp32",
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
