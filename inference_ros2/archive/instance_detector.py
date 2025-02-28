import os

import cv2
import imutils
import numpy as np

# inference imports
import onnxruntime as rt
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

# message definitions
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    Detection2DArrayMask,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from yolox.tracker.byte_tracker import BYTETracker

classes = ["crop", "weed"]
from utils import non_max_suppression, process_mask


from utils import non_max_suppression, process_mask


class CropDetector(Node):
    def __init__(self, mode="onnx", topic="/realsenseD435/color/image_raw/compressed"):
        super().__init__("CropInstanceDetector")
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
        self.subscription  # prevent unused variable warning
        self.inference_mode = mode
        self.tracker = BYTETracker(args=None)
        self.class_ids = {0: "weeds", 1: "maize"}
        if self.inference_mode == "onnx":
            self.init_model(mode=mode)
        elif self.inference_mode == "tensorrt":
            pass
        else:
            raise ValueError("Invalid mode. Choose either onnx or tensorrt")

        self.publisher = self.create_publisher(
            Detection2DArrayMask, "/cropweed/instance_seg", 10
        )

    def listener_callback(self, msg):
        if self.compressed:
            self.cv_image = CvBridge().compressed_imgmsg_to_cv2(msg)
        else:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg)
            self.orig = self.cv_image
        self.cv_image = self.preprocess_image(self.cv_image)
        # inference
        if self.inference_mode == "onnx":
            outputs = self.model.run(None, {self.input_name: self.cv_image})
            if outputs is not None:
                self.postprocess_image(outputs)
        elif self.inference_mode == "tensorrt":
            pass
        else:
            raise ValueError("Invalid mode. Choose either onnx or tensorrt")

    def init_model(self, mode="onnx"):
        """Initialize the model for inference"""
        if mode == "onnx":
            self.model_path = os.path.join(
                os.path.abspath("."), "model/yolov7-instance-seg-cropweed.onnx"
            )
            self.exec_providers = rt.get_available_providers()
            self.exec_provider = (
                ["CUDAExecutionProvider"]
                if "CUDAExecutionProvider" in self.exec_providers
                else ["CPUExecutionProvider"]
            )
            # self.exec_provider = ['TensorrtExecutionProvider'] if 'CUDAExecutionProvider' in self.exec_providers else ['CPUExecutionProvider']
            self.session_options = rt.SessionOptions()
            # TODO add more session options
            self.get_logger().info("Using {} for inference".format(self.exec_provider))
            self.model = rt.InferenceSession(
                self.model_path,
                sess_options=self.session_options,
                providers=self.exec_provider,
            )
            assert self.model
            self.get_logger().info("Model loaded successfully in ONNX format")
        elif mode == "tensorrt":
            self.model = rt.InferenceSession(
                "model/yolov7-instance-seg-cropweed.engine"
            )

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
        self.p_h, self.p_w = image.shape[0], image.shape[1]
        img_0[0 : image.shape[0], 0 : image.shape[1], :] = image
        image = img_0.copy()
        # self.orig = img_0.copy().astype(np.uint8)
        image = image.transpose(2, 0, 1)  # HWC to CHW
        # normalize the image
        image = image / 255.0
        # add batch dimension
        image = np.expand_dims(image, axis=0)  # BCHW
        if self.input_dtype == "tensor(float)":
            image = image.astype(np.float32)
        assert image.shape == (b, c, h, w)
        return image

    def scale_boxes(
        self, box, source_dim=(640, 640), orig_size=(760, 1280), padded=False
    ):
        """
        box: Bounding box generated for the model input size (e.g. 512 x 512) which was fed to the model.
        source_dim : The size of input image fed to the model.
        orig_size : The size of the input image fetched from the ROS publisher.
        padded: If True, source_dim must be changed to p_h, p_w of image i.e. the image dimensions without the padded dimensions.
        return: Scaled bounding box according to the input image from the ROS topic.
        """
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
        xtl, xbr = (
            box[:, 0] * (orig_size[1] / source_dim[1]),
            box[:, 2] * (orig_size[1] / source_dim[1]),
        )
        ytl, ybr = (
            box[:, 1] * (orig_size[0] / source_dim[0]),
            box[:, 3] * (orig_size[0] / source_dim[0]),
        )
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
        # x_min = c_x
        # x_max = c_x + w
        # y_min = c_y
        # y_max = c_y + h

        return torch.concatenate(
            (
                x_min[:, None],
                y_min[:, None],
                x_max[:, None],
                y_max[:, None],
                boxes[:, 4:],
            ),
            axis=1,
        )

    def postprocess_image(self, output):
        """Postprocess the model output for publishing"""
        det = output[0]
        masks = output[4]
        preds = non_max_suppression(
            prediction=det, conf_thres=0.3, iou_thres=0.45, nm=32
        )[0]  # TODO nm dimension to be set?
        if len(preds) != 0:
            masks = process_mask(
                protos=masks[0],
                masks_in=preds[:, 6:],
                bboxes=preds[:, :4],
                shape=(self.orig_shape[0], self.orig_shape[1]),
                upsample=True,
            )
            online_targets = self.tracker.update(
                preds, self.orig_shape, self.orig_shape
            )
            preds = self.cxcywh2xyxy(preds)
            preds = self.scale_boxes(preds, source_dim=(self.p_h, self.p_w))
            # inst_msg = Detection2DArrayMask()
            for obj in range(preds.shape[0]):
                box = preds[obj, :4]
                if preds[obj, 5] == 0:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(
                    self.orig,
                    pt1=(int(box[0]), int(box[1])),
                    pt2=(int(box[2]), int(box[3])),
                    color=color,
                    thickness=2,
                )
                # cv2.putText(self.orig,
                #             '{:.2f} {}'.format(preds[obj, 4], self.class_ids[preds[obj, 5].item()]),
                #             org=(int(box[0]), int(box[1] - 10)),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5,
                #             thickness=2,
                #             color=color)
            cv2.imshow("prediction", self.orig)
            cv2.waitKey(0)
            # detection_array = Detection2DArray()
        #     obj = ObjectHypothesisWithPose()
        #     for i, mask in zip(range(preds.shape[0]), masks):
        #         bbox = BoundingBox2D()
        #         detection = Detection2D()
        #         bbox.center.position.x = preds[i, 0].item()
        #         bbox.center.position.y = preds[i, 1].item()
        #         bbox.size_x = preds[i, 2].item()
        #         bbox.size_y = preds[i, 3].item()
        #         obj = ObjectHypothesisWithPose()
        #         obj.hypothesis.class_id = classes[int(preds[i, 5].item())]
        #         obj.hypothesis.score = np.round(preds[i, 4].item(), 2)
        #         detection.bbox = bbox
        #         detection.results.append(obj)
        #         detection.id = str(0)                               # TODO add tracking IDs here.
        #         msg = Image()
        #         msg.data = mask.numpy().astype(np.uint16).tobytes()  # TODO only 255 instances can be uniquely labelled here.
        #         msg.height = mask.shape[0]
        #         msg.width = mask.shape[1]
        #         msg.step = mask.shape[1]
        #         msg.encoding = 'mono'
        #         inst_msg.masks.append(msg)
        #         inst_msg.detections.append(detection)
        #     self.publisher.publish(inst_msg)
        # else:
        #     inst_msg = Detection2DArrayMask()
        #     self.publisher.publish(inst_msg)


def main(args=None):
    rclpy.init(args=args)
    subscriber = CropDetector(topic="/realsenseD435/color/image_raw")
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
