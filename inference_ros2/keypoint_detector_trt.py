import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils
import os
import time 

# message definitions
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from vision_msgs.msg import Keypoint2DArray, Keypoint2D
from sensor_msgs.msg import CompressedImage, Image

import tensorrt as trt
import pycuda.driver as cuda
from utils.trt_utils import HostDeviceMem, _cuda_error_check, TrtLogger
from utils import non_max_suppression_v8, scale_boxes, scale_coords, plot, xywh2xyxy
from yolox.tracker.byte_tracker import BYTETracker


classes = ['background', 'crop', 'weed']
kpt_shape = (1,3)

class CropKeypointDetector(Node):

    def __init__(self, mode='onnx', topic='/realsenseD435/color/image_raw/compressed'):
        super().__init__('CropKeypointDetector')
        if 'compressed' in topic:
            self.compressed = True
            self.subscription = self.create_subscription(
                CompressedImage,
                topic,
                self.listener_callback,
                10)
        else:
            self.compressed = False
            self.subscription = self.create_subscription(
                Image,
                topic,
                self.listener_callback,
                10)
        self.get_logger().info('Subscribing to {}'.format(topic))
        self.ros_logger = self.get_logger()
        self.trt_logger = TrtLogger(self)
        self.class_ids = {0: 'weeds', 1: 'maize'}
        self.tracker = BYTETracker(args=None, class_dict=self.class_ids)
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        # NOTE! self.context is not allowed since the Node parent has a ROS2 related context which cannot be overridden. 
        self.trt_context = None 
        self.init_model(mode=mode)
        self.publisher = self.create_publisher(Keypoint2DArray, '/cropweed/keypoint_detection', 10)

    def get_logger(self):
        # Override get_logger to use ROS 2 logger
        return super().get_logger()
    
    def listener_callback(self, msg):
        if self.compressed:
            self.cv_image = CvBridge().compressed_imgmsg_to_cv2(msg)
        else:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg)
        if self.cv_image.shape[2] != 3:
            self.cv_image = self.cv_image[:,:, :3]
        # TODO: Remove after DEBUG
        self.header = msg.header
        # self.cv_image = cv2.imread('./sample.png')
        # self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.orig = self.cv_image.astype(np.uint8)
        t1 = time.time()
        self.input_image = self.preprocess_image(self.cv_image)        
        t2 = time.time()
        preprocess_time = round((t2-t1)*1000, 2)
        # inference
        if self.inference_mode == 'onnx':
            outputs = self.model.run(None, {self.input_name: self.input_image})
            if outputs is not None:
                self.postprocess_image(outputs)
        elif self.inference_mode == 'tensorrt':
            try:
                outputs = self.infer_trt(self.input_image)
                t3 = time.time()
                inference_time = round((t3-t2)*1000, 2)
                if outputs is not None:
                    self.postprocess_image(outputs)
                t4 = time.time()
                post_process_time = round((t4-t3)*1000, 2)
                total = preprocess_time + inference_time + post_process_time
                self.ros_logger.info("Preprocessing: {} ms Inference: {} ms Postprocessing {} ms FPS: {}".format(preprocess_time, 
                                                                                                                 inference_time, 
                                                                                                                 post_process_time,
                                                                                                                 round(1/(total / 1000), 2)))
            except KeyboardInterrupt:
                self.get_logger().loginfo("Callback interrupted, cleaning up CUDA context")
                self.cuda_ctx.pop()
        else:
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')

    def init_model(self, mode='onnx'):
        """Initialize the model for inference"""
        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        self.engine_path = os.path.join('/ros2_ws/src/inference_ros2/model/yolov8_trt_23.10_fp32.engine')
        # self.logger = trt.Logger(self.trt_logger)
        self.runtime = trt.Runtime(self.trt_logger)
        trt.init_libnvinfer_plugins(None, "")   
        assert os.path.exists(self.engine_path)          
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if 'fp32' in self.engine_path:
            self.dtype = np.float32
        elif 'fp16' in self.engine_path:
            self.dtype = np.float16
        elif 'int8' in self.engine_path:
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
        '''
            input_image: A numpy array that is contiguous, preprocessed and in the data type expected by the self.data_type variable 
        '''
        self.input_image = self.input_image.ravel()
        # NOTE! Copy current image to the host buffer i.e. self.input.host
        np.copyto(self.input.host, self.input_image)
        # Process I/O and execute the network.
        self.cuda_ctx.push()
        # Copy current image from host to device memory ----> part of IO operations
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        # t2 = time.time()
        # io_time = round((t2-t1)*1000, 2)
        err = self.trt_context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
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
        # Setup I/O bindings
        self.outputs = []
        self.bindings = []
        self.allocations = []
        for binding_name, i in zip(self.engine, range(len(self.engine))):
            is_input = False
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            shape = self.engine.get_tensor_shape(binding_name)
            if self.engine.get_tensor_mode(binding_name).name=='INPUT':
                is_input = True
                self.batch_size = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #self.engine.max_batch_size #shape[0]
                self.input_shape = shape
                self.input_dtype = dtype
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            # Here the size argument is the length of the flattened array and not the byte size
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding_name).name=='INPUT':
                self.input = HostDeviceMem(host_mem, device_mem)
                # TODO Deprecated warning remove in future
                self.trt_context.set_binding_shape(i, [j for j in shape])
                # self.trt_context.set_input_shape(binding_name, [j for j in shape])
            elif self.engine.get_tensor_mode(binding_name).name=='OUTPUT':
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.trt_context.set_binding_shape(i, [j for j in shape])
                # TODO Deprecated warning remove in future
                # self.trt_context.set_input_shape(binding_name, [j for j in shape])
                self.output_shape = shape
            else:
                pass

        # print(self.trt_context.all_binding_shapes_specified)
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
        img_0[0:image.shape[0], 0:image.shape[1], :] = image
        image = img_0.copy()
        image = image.transpose(2, 0, 1)     # HWC to CHW
        # normalize the image
        image = image / 255.0
        # add batch dimension
        image = np.expand_dims(image, axis=0)   # BCHW
        image = np.ascontiguousarray(image.astype(np.dtype(self.dtype))).ravel()
        return image
    
    def postprocess_image(self, output):
        """Postprocess the model output for publishing"""
        det = output[0]
        preds = non_max_suppression_v8(prediction=det, 
                                       conf_thres=0.5, 
                                       iou_thres=0.9,
                                       max_det=200,
                                    #    multi_label=True, 
                                       nc=len(classes))[0] 
        # The preds are in xtl,ytl,w,h format
        preds[:, :4] = scale_boxes(preds[:, :4], (self.p_h, self.p_w), self.orig_shape, padded=True)
        preds[:, :4] = xywh2xyxy(preds[:, :4])
        # keep_indices = remove_overlapping_boxes(preds[:, :4], iou_threshold=0.8)
        # preds = preds[keep_indices, :]
        pred_kpts = preds[:, 6:].view(len(preds), *kpt_shape) if len(preds) else preds[:, 6:] # TODO Fetch keypoint shape from model dynamically
        pred_kpts = scale_coords((self.p_h, self.p_w), pred_kpts, self.orig_shape)
        # The tracker expects them in xtl,ytl,xbr,ybr format.
        self.online_targets = self.tracker.update(preds, self.orig_shape, self.orig_shape)
        # if preds.shape[0] != 0:
        #     self.orig = plot(preds, pred_kpts, self.orig, mode='det')
        # if len(self.online_targets) != 0: 
        #     self.orig = plot(self.online_targets, None, self.orig, mode='track')
        # cv2.imwrite('prediction.jpg', self.orig)
        # cv2.imshow('predictions', self.orig)
        # cv2.waitKey(1)
        # plot(preds, pred_kpts, self.cv_image.astype(np.uint8))
        if preds.shape[0] != 0:
            keypoint_msg = Keypoint2DArray()
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
                detection.id = str(0)                               # TODO add tracking IDs here. 
                keypoint = Keypoint2D()
                keypoint.position.x = pred_kpts[kpt_idx, 0, 0].item()
                keypoint.position.y = pred_kpts[kpt_idx, 0, 1].item()
                keypoint.confidence = pred_kpts[kpt_idx, 0, 2].item()
                keypoint_msg.header = self.header
                keypoint_msg.keypoints.append(keypoint)
                keypoint_msg.detections.append(detection)
            self.publisher.publish(keypoint_msg)
        else:
            keypoint_msg = Keypoint2DArray()
            self.publisher.publish(keypoint_msg)

def main(args=None):
    rclpy.init(args=args)
    subscriber = CropKeypointDetector(topic='/zed/zed_node/rgb/image_rect_color', mode='tensorrt')
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
