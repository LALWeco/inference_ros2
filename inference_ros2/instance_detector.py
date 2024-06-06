import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils
import os

# message definitions
from vision_msgs.msg import Detection2D, Detection2DArray, Detection2DArrayMask, BoundingBox2D, ObjectHypothesisWithPose, ObjectHypothesis
from sensor_msgs.msg import CompressedImage, Image

# inference imports
import onnxruntime as rt

classes = ['crop', 'weed']
from utils import non_max_suppression, process_mask
class CropDetector(Node):

    def __init__(self, mode='onnx', topic='/realsenseD435/color/image_raw/compressed'):
        super().__init__('CropDetector')
        self.subscription = self.create_subscription(
            CompressedImage,
            topic,
            self.listener_callback,
            10)
        self.get_logger().info('Subscribing to {}'.format(topic))
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        if self.inference_mode == 'onnx':
            self.init_model(mode=mode)
        elif self.inference_mode == 'tensorrt':
            pass
        else:   
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')
        
        self.publisher = self.create_publisher(Detection2DArrayMask, '/cropweed/instance_seg', 10)

    def listener_callback(self, msg):
        cv_image = CvBridge().compressed_imgmsg_to_cv2(msg)
        cv_image = self.preprocess_image(cv_image)
        # inference
        if self.inference_mode == 'onnx':
            outputs = self.model.run(None, {self.input_name: cv_image})
            if outputs is not None:
                self.postprocess_image(outputs)
        elif self.inference_mode == 'tensorrt':
            pass
        else:
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')

    def init_model(self, mode='onnx'):
        """Initialize the model for inference"""
        if mode == 'onnx':
            self.model_path = os.path.join(os.path.abspath('.'),
                    'src/inference_ros2/model/yolov7-instance-seg-cropweed.onnx')
            self.exec_providers = rt.get_available_providers()
            self.exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in self.exec_providers else ['CPUExecutionProvider']
            self.session_options = rt.SessionOptions()
            # TODO add more session options
            self.get_logger().info('Using {} for inference'.format(self.exec_provider))
            self.model = rt.InferenceSession(self.model_path, 
                                             sess_options=self.session_options,
                                             providers=self.exec_provider)
            assert self.model 
            self.get_logger().info('Model loaded successfully in ONNX format')
        elif mode == 'tensorrt':
            self.model = rt.InferenceSession('model/yolov7-instance-seg-cropweed.engine')
        
    def preprocess_image(self, image):
        """Preprocess the image for inference"""
        self.orig_shape = image.shape
        # resize and pad to square the resolution
        b, c, h, w = self.model.get_inputs()[0].shape
        self.input_name = self.model.get_inputs()[0].name
        self.input_dtype = self.model.get_inputs()[0].type
        assert c == 3
        img_0 = np.zeros((h, w, 3), dtype=np.float32)
        image = imutils.resize(image, width=w)
        p_h, p_w = image.shape[0], image.shape[1]
        img_0[0:image.shape[0], 0:image.shape[1], :] = image
        image = img_0.copy()
        image = image.transpose(2, 0, 1)    # HWC to CHW
        # normalize the image
        image = image / 255.0
        # add batch dimension
        image = np.expand_dims(image, axis=0)   # BCHW
        if self.input_dtype == 'tensor(float)':
            image = image.astype(np.float32)
        assert image.shape == (b, c, h, w)
        return image
    
    def postprocess_image(self, output):
        """Postprocess the model output for publishing"""
        det = output[0]
        masks = output[4]
        preds = non_max_suppression(prediction=det, conf_thres=0.3, iou_thres=0.45, nm=32)[0] #TODO nm dimension to be set?
        if len(preds) != 0:
            masks = process_mask(protos=masks[0], 
                                masks_in=preds[:, 6:], 
                                bboxes=preds[:, :4], 
                                shape=(self.orig_shape[0], self.orig_shape[1]), 
                                upsample=True)
            inst_msg = Detection2DArrayMask()
            # detection_array = Detection2DArray()
            obj = ObjectHypothesisWithPose()
            for i, mask in zip(range(preds.shape[0]), masks):
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
                msg = Image()
                msg.data = mask.numpy().astype(np.uint16).tobytes()  # TODO only 255 instances can be uniquely labelled here. 
                msg.height = mask.shape[0]
                msg.width = mask.shape[1]
                msg.step = mask.shape[1]
                msg.encoding = 'mono'
                inst_msg.masks.append(msg)
                inst_msg.detections.append(detection)
            self.publisher.publish(inst_msg)
        else:
            inst_msg = Detection2DArrayMask()
            self.publisher.publish(inst_msg)

def main(args=None):
    rclpy.init(args=args)
    subscriber = CropDetector()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
