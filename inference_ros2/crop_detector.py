import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils

# message definitions
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import CompressedImage

# inference imports
import onnxruntime as rt

class CropDetector(Node):

    def __init__(self, mode='onnx'):
        super().__init__('CropDetector')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/realsenseD435/color/image_raw/compressed',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        if self.inference_mode == 'onnx':
            self.init_model(mode=mode)
        elif self.inference_mode == 'tensorrt':
            pass
        else:   
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')

    def listener_callback(self, msg):
        cv_image = CvBridge().compressed_imgmsg_to_cv2(msg)
        cv_image = self.preprocess_image(cv_image)
        # inference
        if self.inference_mode == 'onnx':
            outputs = self.model.run(None, {self.input_name: cv_image})
        elif self.inference_mode == 'tensorrt':
            pass
        else:
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')

    def init_model(self, mode='onnx'):
        """Initialize the model for inference"""
        if mode == 'onnx':
            self.model_path = 'model/yolov7-instance-seg-cropweed.onnx'
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

def main(args=None):
    rclpy.init(args=args)

    subscriber = CropDetector()

    rclpy.spin(subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()