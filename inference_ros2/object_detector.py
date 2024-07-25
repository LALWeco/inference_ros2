import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils
import os
import torch

# message definitions
from vision_msgs.msg import Detection2D, Detection2DArray, Detection2DArrayMask, BoundingBox2D, ObjectHypothesisWithPose, ObjectHypothesis
from sensor_msgs.msg import CompressedImage, Image

# inference imports
import onnxruntime as rt
from yolox.tracker.byte_tracker import BYTETracker

classes = ['crop', 'weed']
from utils import non_max_suppression_v8, scale_boxes, scale_coords, plot
class CropDetector(Node):

    def __init__(self, mode='onnx', topic='/realsenseD435/color/image_raw/compressed'):
        super().__init__('CropDetector')
        if 'compressed' in topic:
            self.compressed = True
            self.subscription = self.create_subscription(
                CompressedImage,
                topic,
                self.listener_callback,
                30)
        else:
            self.compressed = False
            self.subscription = self.create_subscription(
                Image,
                topic,
                self.listener_callback,
                30)
        self.get_logger().info('Subscribing to {}'.format(topic))
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        self.tracker = BYTETracker(args=None)
        self.class_ids = {0: 'weeds', 1: 'maize'}
        if self.inference_mode == 'onnx':
            self.init_model(mode=mode)
        elif self.inference_mode == 'tensorrt':
            pass
        else:   
            raise ValueError('Invalid mode. Choose either onnx or tensorrt')
        
        self.publisher = self.create_publisher(Detection2DArrayMask, '/cropweed/instance_seg', 10)

    def listener_callback(self, msg):
        if self.compressed:
            self.cv_image = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.orig = self.cv_image
        else:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.orig = self.cv_image
        self.cv_image = self.preprocess_image(self.cv_image)
        # inference
        if self.inference_mode == 'onnx':
            outputs = self.model.run(None, {self.input_name: self.cv_image})
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
                    'model/rt_detr.onnx')
            self.exec_providers = rt.get_available_providers()
            self.exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in self.exec_providers else ['CPUExecutionProvider']
            # self.exec_provider = ['TensorrtExecutionProvider'] if 'CUDAExecutionProvider' in self.exec_providers else ['CPUExecutionProvider']
            self.session_options = rt.SessionOptions()
            # TODO add more session options
            self.get_logger().info('Using {} for inference'.format(self.exec_provider))
            self.model = rt.InferenceSession(self.model_path, 
                                             sess_options=self.session_options,
                                             providers=self.exec_provider)
            assert self.model 
            self.b, self.c, h, w = self.model.get_inputs()[0].shape
            self.input_name = self.model.get_inputs()[0].name
            self.input_dtype = self.model.get_inputs()[0].type
            self.input_shape = [h, w, self.c]
            assert self.c == 3
            self.get_logger().info('Model loaded successfully in ONNX format')
        elif mode == 'tensorrt':
            self.model = rt.InferenceSession('model/yolov7-instance-seg-cropweed.engine')
        
    def preprocess_image(self, image):
        """Preprocess the image for inference"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        self.orig_shape = image.shape
        # resize and pad to square the resolution
        img_0 = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        image = imutils.resize(image, width=self.input_shape[1])
        self.p_h, self.p_w = image.shape[0], image.shape[1]
        img_0[0:image.shape[0], 0:image.shape[1], :] = image
        image = img_0.copy()
        # self.p_image = cv2.cvtColor(image.copy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        # self.orig = img_0.copy().astype(np.uint8)
        image = image.transpose(2, 0, 1)    # HWC to CHW
        # normalize the image
        image = image / 255.0
        # add batch dimension
        image = np.expand_dims(image, axis=0)   # BCHW
        if self.input_dtype == 'tensor(float)':
            image = image.astype(np.float32)
        elif self.input_dtype == 'tensor(float16)':
            image = image.astype(np.float16)
        assert image.shape == (self.b, self.c, self.input_shape[0], self.input_shape[1])
        return np.ascontiguousarray(image)
    
    def scale_boxes(self, box, source_dim=(640,640), orig_size=(760, 1280), padded=False):
        '''
        box: Bounding box generated for the model input size (e.g. 512 x 512) which was fed to the model. 
        source_dim : The size of input image fed to the model. 
        orig_size : The size of the input image fetched from the ROS publisher. 
        padded: If True, source_dim must be changed to p_h, p_w of image i.e. the image dimensions without the padded dimensions. 
        return: Scaled bounding box according to the input image from the ROS topic.
        '''
        # if padded:
        #     xtl, xbr = box[:, 0] * (orig_size[1] / source_dim[1]), \
        #                box[:, 2] * (orig_size[1] / source_dim[1])
        #     ytl, ybr = box[:, 1] * (orig_size[0] / source_dim[0]), \
        #                box[:, 3] * (orig_size[0] / source_dim[0])
        # else:
        #     xtl, xbr = box[:, 0] * (orig_size[1] / source_dim[1]), \
        #                box[:, 2] * (orig_size[1] / source_dim[1])
        #     ytl, ybr = box[:, 1] * orig_size[0] / self.input_size[0], \
        #                box[:, 3] * orig_size[0] / self.input_size[0]
        xtl, xbr = box[:, 0] * (orig_size[1] / source_dim[1]), \
                       box[:, 2] * (orig_size[1] / source_dim[1])
        ytl, ybr = box[:, 1] * (orig_size[0] / source_dim[0]), \
                    box[:, 3] * (orig_size[0] / source_dim[0])
        xtl = np.reshape(xtl, (len(xtl), 1))
        xbr = np.reshape(xbr, (len(xbr), 1))

        ytl = np.reshape(ytl, (len(ytl), 1))
        ybr = np.reshape(ybr, (len(ybr), 1))

        return torch.concatenate((xtl, ytl, xbr, ybr, box[:, 4:]), axis=1)
    
    def cxcywh2xyxy(self, boxes):
        c_x = boxes[:, 0]
        c_y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x_min = c_x - (w / 2)
        x_max = c_x + (w / 2)
        y_min = c_y - (h / 2)
        y_max = c_y + (h / 2)

        return torch.concatenate((x_min[:,None], y_min[:,None], x_max[:,None], y_max[:,None], boxes[:, 4:]), axis=1)
    
    def postprocess_image(self, output):
        """Postprocess the model output for publishing"""
        if False:
            preds = non_max_suppression_v8(prediction=output[0], 
                                           conf_thres=0.3, 
                                           iou_thres=0.5,
                                           max_det=200, 
                                        #    multi_label=True, 
                                           nc=len(classes))[0] 
            preds[:, :4] = scale_boxes(preds[:, :4], (self.p_h, self.p_w), self.orig_shape, padded=True)
        else:
            tmp = torch.empty((0, 6), dtype=torch.float32)
            nd = output[0].shape[-1]
            preds = torch.tensor(output[0], device='cpu')
            batch_data, scores = preds.split((4, nd - 4), dim=-1)
            for batch_index, bbox in enumerate(batch_data):  # (300, 4)
                bbox = self.cxcywh2xyxy(bbox)
                score, cls = scores[batch_index].max(-1, keepdim=True)  # (300, 1)
                idx = score.squeeze(-1) > 0.4  # (300, )
                pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
                # These pixel locations are based on padded image input to the network
                pred[..., [0, 2]] *= self.input_shape[1]
                pred[..., [1, 3]] *= self.input_shape[0]
                tmp = torch.concatenate([tmp, pred], dim=0)
            preds = tmp
            preds[:, :4] = scale_boxes(preds[:, :4], (self.p_h, self.p_w), self.orig_shape, padded=True)
            tmp = []      
        self.online_targets = self.tracker.update(preds, self.orig_shape, self.orig_shape)
        if len(self.online_targets) != 0: 
            self.orig = plot(self.online_targets, None, self.orig, mode='track')
        # if preds.shape[0] != 0:
        #     self.orig = plot(preds, None, self.orig, mode='det')
        preds = []
        cv2.imshow('prediction', self.orig)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    subscriber = CropDetector(topic='/oak1_3/oak_rgb_publisher/compressed')
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
