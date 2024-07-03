import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils
import os

# message definitions
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from vision_msgs.msg import Keypoint2DArray, Keypoint2D
from sensor_msgs.msg import CompressedImage, Image

import tensorrt as trt
import pycuda.driver as cuda
from utils.trt_utils import HostDeviceMem, _cuda_error_check

classes = ['background', 'crop', 'weed']
kpt_shape = (1,3)
from utils import non_max_suppression_v8, scale_boxes, scale_coords, plot

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
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        self.init_model(mode=mode)
        self.publisher = self.create_publisher(Keypoint2DArray, '/cropweed/keypoint_detection', 10)

    def listener_callback(self, msg):
        if self.compressed:
            self.cv_image = CvBridge().compressed_imgmsg_to_cv2(msg)
        else:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg)
        if self.cv_image.shape[2] != 3:
            self.cv_image = self.cv_image[:,:, :3]
        self.input_image = self.preprocess_image(self.cv_image)
        # inference
        if self.inference_mode == 'onnx':
            outputs = self.model.run(None, {self.input_name: self.input_image})
            if outputs is not None:
                self.postprocess_image(outputs)
        elif self.inference_mode == 'tensorrt':
            pass
        else:
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')

    def init_model(self, mode='onnx'):
        """Initialize the model for inference"""
        if self.inference_mode == 'onnx':
            import onnxruntime as rt
            self.model_path = os.path.join(os.path.abspath('.'),
                    'model/yolov8-keypoint-det-cropweed.onnx')
            self.exec_providers = rt.get_available_providers()
            self.exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in self.exec_providers else ['CPUExecutionProvider']
            self.session_options = rt.SessionOptions()
            # TODO add more session optionss
            self.get_logger().info('Using {} for inference'.format(self.exec_provider))
            self.model = rt.InferenceSession(self.model_path, 
                                             sess_options=self.session_options,
                                             providers=self.exec_provider)
            assert self.model 
            self.get_logger().info('Model loaded successfully in ONNX format')
        elif self.inference_mode == 'tensorrt':
            cuda.init()
            self.device = cuda.Device(0)
            self.cuda_ctx = self.device.make_context()
            self.engine_path = 'model/yolov7-instance-seg-cropweed.engine'
            self.logger = trt.Logger(trt.Logger.ERROR)
            self.runtime = trt.Runtime(self.logger)
            trt.init_libnvinfer_plugins(None, "")   
            assert os.path.exists(self.engine_path)          
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            self.dtype = np.float32
            self.confidence = 0.5
            self.batch_size = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            trt.init_libnvinfer_plugins(self.logger, namespace="")
            self.context = self.engine.create_execution_context()
            assert self.engine
            assert self.context
            self.setup_IO_binding()
    

    def build_engine_onnx(self, model_file):
        def GiB(val):
            return val * 1 << 30
        builder = trt.Builder(self.logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.logger)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1))
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        engine_bytes = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(self.logger)
        return runtime.deserialize_cuda_engine(engine_bytes)

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
                self.context.set_binding_shape(i, [j for j in shape])
                # self.context.set_input_shape(binding_name, [j for j in shape])
            elif self.engine.get_tensor_mode(binding_name).name=='OUTPUT':
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.context.set_binding_shape(i, [j for j in shape])
                # TODO Deprecated warning remove in future
                # self.context.set_input_shape(binding_name, [j for j in shape])
                self.output_shape = shape
            else:
                pass

        # print(self.context.all_binding_shapes_specified)
        self.stream = cuda.Stream()
        assert len(self.outputs) > 0
        assert len(self.bindings) > 0

    def preprocess_image(self, image):
        """Preprocess the image for inference"""
        if self.inference_mode == 'onnx':
            self.orig_shape = image.shape
            # resize and pad to square the resolution
            b, c, h, w = self.model.get_inputs()[0].shape
            self.input_name = self.model.get_inputs()[0].name
            self.input_dtype = self.model.get_inputs()[0].type
            self.input_shape = h, w
            assert c == 3
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
            if self.input_dtype == 'tensor(float)':
                image = image.astype(np.float32)
            assert image.shape == (b, c, h, w)
        elif self.inference_mode == 'tensorrt':
            self.orig_shape = image.shape
            # resize and pad to square the resolution
            b, c, h, w = self.model.get_inputs()[0].shape
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
            image = np.ascontiguousarray(self.in_tensor.astype(np.dtype(self.dtype))).ravel()
        return image
    
    def postprocess_image(self, output):
        """Postprocess the model output for publishing"""
        det = output[0]
        preds = non_max_suppression_v8(prediction=det, 
                                       conf_thres=0.5, 
                                       iou_thres=0.5,
                                       max_det=200,
                                    #    multi_label=True, 
                                       nc=len(classes))[0] 
        preds[:, :4] = scale_boxes(preds[:, :4], (self.p_h, self.p_w), self.orig_shape, padded=True)
        pred_kpts = preds[:, 6:].view(len(preds), *kpt_shape) if len(preds) else preds[:, 6:] # TODO Fetch keypoint shape from model dynamically
        pred_kpts = scale_coords((self.p_h, self.p_w), pred_kpts, self.orig_shape)
        plot(preds, pred_kpts, self.cv_image)
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
                keypoint_msg.keypoints.append(keypoint)
                keypoint_msg.detections.append(detection)
            self.publisher.publish(keypoint_msg)
        else:
            keypoint_msg = Keypoint2DArray()
            self.publisher.publish(keypoint_msg)

def main(args=None):
    rclpy.init(args=args)
    # subscriber = CropKeypointDetector(topic='/zed/zed_node/rgb/image_rect_color')
    subscriber = CropKeypointDetector(topic='/realsenseD435/color/image_raw/compressed', mode='tensorrt')
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
