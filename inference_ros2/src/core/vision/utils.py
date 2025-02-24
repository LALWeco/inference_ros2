import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
import imutils
import torch
import torchvision

def preprocess_image(
    image: np.ndarray,
    input_shape: Tuple[int, int, int, int],
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Preprocess image for model inference.
    
    Args:
        image: Input image in BGR format
        input_shape: Model input shape (batch, channels, height, width)
        dtype: Data type for the preprocessed image
        
    Returns:
        Preprocessed image ready for inference
    """
    b, c, h, w = input_shape
    
    # Create zero-padded image
    padded_img = np.zeros((h, w, 3), dtype=np.float32)
    
    # Resize image preserving aspect ratio
    resized = imutils.resize(image, width=w)
    p_h, p_w = resized.shape[0], resized.shape[1]
    
    # Copy resized image into padded image
    padded_img[0:p_h, 0:p_w, :] = resized
    
    # Transpose to CHW format
    preprocessed = padded_img.transpose(2, 0, 1)
    
    # Normalize
    preprocessed = preprocessed / 255.0
    
    # Add batch dimension
    preprocessed = np.expand_dims(preprocessed, axis=0)
    
    # Convert to desired dtype
    preprocessed = np.ascontiguousarray(preprocessed.astype(dtype))
    
    return preprocessed, (p_h, p_w)

def scale_boxes(
    boxes: np.ndarray,
    padded_shape: Tuple[int, int],
    original_shape: Tuple[int, int, int],
    padded: bool = True
) -> np.ndarray:
    """Scale bounding boxes from padded dimensions to original image dimensions.
    
    Args:
        boxes: Array of bounding boxes
        padded_shape: Shape of padded image (height, width)
        original_shape: Shape of original image (height, width, channels)
        padded: Whether boxes are in padded image coordinates
        
    Returns:
        Scaled bounding boxes
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    
    shape = original_shape[:2]  # height, width
    ratio = min(padded_shape[0] / shape[0], padded_shape[1] / shape[1])
    
    if padded:
        boxes[:, [0, 2]] -= (padded_shape[1] - ratio * shape[1]) / 2
        boxes[:, [1, 3]] -= (padded_shape[0] - ratio * shape[0]) / 2
    
    boxes[:, :4] /= ratio
    
    # Clip boxes to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    
    return boxes

def scale_coords(
    padded_shape: Tuple[int, int],
    coords: np.ndarray,
    original_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Scale coordinates (keypoints) from padded to original dimensions.
    
    Args:
        padded_shape: Shape of padded image (height, width)
        coords: Array of coordinates
        original_shape: Shape of original image (height, width, channels)
        
    Returns:
        Scaled coordinates
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    
    shape = original_shape[:2]
    ratio = min(padded_shape[0] / shape[0], padded_shape[1] / shape[1])
    
    coords[:, :, 0] -= (padded_shape[1] - ratio * shape[1]) / 2
    coords[:, :, 1] -= (padded_shape[0] - ratio * shape[0]) / 2
    coords[:, :, :2] /= ratio
    
    # Clip to image bounds
    coords[:, :, 0] = coords[:, :, 0].clip(0, shape[1])
    coords[:, :, 1] = coords[:, :, 1].clip(0, shape[0])
    
    return coords

def xywh2xyxy(x):
    # Convert nx4 boxes from [cx, cy, w, h] to [xtl, ytl, xbr, ybr] where tl=top-left, br=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [cx, cy, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def keypoint_nms(
    keypoints: np.ndarray, 
    confidences: np.ndarray, 
    distance_threshold: float = 30.0
) -> List[int]:
    """Apply non-maximum suppression to keypoints based on distance and confidence.
    
    Args:
        keypoints: Array of keypoint coordinates (N, 2)
        confidences: Array of confidence scores (N,)
        distance_threshold: Maximum distance between keypoints for suppression
        
    Returns:
        Indices of keypoints to keep
    """
    if len(keypoints) == 0:
        return []
    
    # Ensure numpy arrays
    keypoints = np.array(keypoints)
    confidences = np.array(confidences)
    
    # Sort by confidence
    order = confidences.argsort()[::-1]
    keep_indices = []
    
    while order.size > 0:
        # Keep highest confidence keypoint
        i = order[0]
        keep_indices.append(i)
        
        if order.size == 1:
            break
        
        # Calculate distances to remaining keypoints
        current_point = keypoints[i]
        other_points = keypoints[order[1:]]
        distances = np.sqrt(np.sum((other_points - current_point) ** 2, axis=1))
        
        # Keep points beyond threshold
        far_enough = distances > distance_threshold
        order = order[1:][far_enough]
    
    return keep_indices

def remove_overlapping_boxes(
    boxes: np.ndarray, 
    iou_threshold: float = 0.8
) -> List[int]:
    """Remove overlapping bounding boxes using NMS.
    
    Args:
        boxes: Array of boxes in (x1,y1,x2,y2) format
        iou_threshold: IoU threshold for overlap
        
    Returns:
        Indices of boxes to keep
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by bottom-right y-coordinate
    order = y2.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Calculate IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

# source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py TODO check license!!!!
def non_max_suppression(
    predictions,
    conf_thresh=0.25,
    iou_thresh=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    predictions = torch.tensor(predictions, device="cpu")
    assert (
        0 <= conf_thresh <= 1
    ), f"Invalid Confidence threshold {conf_thresh}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thresh <= 1
    ), f"Invalid IoU {iou_thresh}, valid values are between 0.0 and 1.0"
    if isinstance(
        predictions, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        predictions = predictions[0]  # select only inference output

    device = predictions.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        predictions = predictions.cpu()
    bs = predictions.shape[0]  # batch size
    nc = nc or (predictions.shape[1] - 4)  # number of classes
    nm = predictions.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = predictions[:, 4:mi].amax(1) > conf_thresh  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    predictions = predictions.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    predictions[..., :4] = xywh2xyxy(predictions[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=predictions.device)] * bs
    for xi, x in enumerate(predictions):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thresh)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thresh]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thresh)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
    # Convert to numpy if torch
    if not mps:
        output = [x.cpu().numpy() for x
                  in output]
    return output
