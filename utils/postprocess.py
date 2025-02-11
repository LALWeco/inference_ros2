import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

class_dict = {0: "background", 1: "crop", 2: "weed"}


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def crop(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[
        None, None, :
    ]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[
        None, :, None
    ]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Crop after upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[
        0
    ]  # CHW
    masks = crop(masks, bboxes)  # CHW
    return masks.gt_(0.5)


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


# def cxcywh2xyxy(boxes):
#     c_x = boxes[:, 0]
#     c_y = boxes[:, 1]
#     w = boxes[:, 2]
#     h = boxes[:, 3]

#     x_min = c_x - (w / 2)
#     x_max = c_x + (w / 2)
#     y_min = c_y - (h / 2)
#     y_max = c_y + (h / 2)

#     return torch.concatenate((x_min[:,None], y_min[:,None], x_max[:,None], y_max[:,None], boxes[:, 4:]), axis=1)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """
    if len(bboxes) == 0:
        return torch.zeros(shape, dtype=torch.bool)
    else:
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        protos = torch.tensor(protos, dtype=torch.float32)
        masks = (
            (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
        )  # CHW

        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = crop(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = F.interpolate(
                masks[None], shape, mode="bilinear", align_corners=False
            )[0]  # CHW
        return masks.gt_(0.5)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    prediction = torch.tensor(prediction)
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        output[xi][:, :4] = xyxy2xywh(output[xi][:, :4])
    return output


# source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py TODO check license!!!!
def non_max_suppression_v8(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
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
    prediction = torch.tensor(prediction, device="cpu")
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
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
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

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
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
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

    return output


def remove_overlapping_boxes(boxes, iou_threshold):
    """
    Remove overlapping bounding boxes based on IoU threshold.

    Args:
    boxes (np.array): Array of bounding boxes in format [xtl, ytl, xbr, ybr]
    iou_threshold (float): IoU threshold for considering boxes as overlapping

    Returns:
    list: List of boxes to keep
    """

    def calculate_iou(box1, box2):
        """Calculate IoU of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / (area1 + area2 - intersection)
        return iou

    if len(boxes) == 0:
        return np.array([])

    # Sort boxes by area (largest first)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = np.argsort(areas).tolist()
    sorted_indices.reverse()  # Reverse the order to get largest first

    keep = []

    for i in range(len(boxes)):
        should_keep = True
        for j in keep:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)

    return keep


def scale_boxes(box, source_dim=(512, 512), orig_size=(760, 1280), padded=False):
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

    return torch.concatenate((xtl, ytl, xbr, ybr), axis=1)


def remove_overlapping_boxes(boxes, iou_threshold):
    """
    Remove overlapping bounding boxes based on IoU threshold.

    Args:
    boxes (np.array): Array of bounding boxes in format [xtl, ytl, xbr, ybr]
    iou_threshold (float): IoU threshold for considering boxes as overlapping

    Returns:
    list: List of boxes to keep
    """

    def calculate_iou(box1, box2):
        """Calculate IoU of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / (area1 + area2 - intersection)
        return iou

    if len(boxes) == 0:
        return np.array([])

    # Sort boxes by area (largest first)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = np.argsort(areas).tolist()
    sorted_indices.reverse()  # Reverse the order to get largest first

    keep = []

    for i in range(len(boxes)):
        should_keep = True
        for j in keep:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)

    return keep


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(
    img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True
):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
    """
    if isinstance(coords, torch.Tensor):  # faster individually
        coords[..., 0].clamp_(0, shape[1])  # x
        coords[..., 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y


def plot(boxes, keypoints, cv_image, mode="track"):
    cv_image = plot_boxes(boxes, cv_image, mode=mode)
    if keypoints is not None:
        cv_image = plot_kpts(keypoints, cv_image)
    # cv2.imshow('prediction', cv_image)
    # cv2.waitKey(1)
    return cv_image


def plot_boxes(boxes, cv_image, mode="det"):
    """
    boxes : [xtl, ytl, xbr, ybr]
    cv_image : cv image in HWC format on which cv functions can operate.
    mode = 'det' or 'track'
    """
    if mode == "track":
        tracks = boxes.copy()
        boxes = np.array([track.tlbr for track in boxes])
        # boxes = xywh2xyxy(boxes) This is not needed and only kept for debugging
        for box_idx in range(boxes.shape[0]):
            box = boxes[box_idx, :]
            # if tracklet.score < 1.0:
            #     color = (0, 0, 255)
            # else:
            #     color = (255, 0, 0)
            color = (0, 0, 255)
            text = "{} {}".format(
                "canola", int(tracks[box_idx].track_id)
            )  # TODO dynamically assign classes in a multi class scenarios
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            if (
                int(box[1] - 25) < 10
            ):  # if box is at the top then put label at bottom right corner.
                cv2.rectangle(
                    cv_image,
                    pt1=(int(box[2] - tw), int(box[3])),
                    pt2=(int(box[2]), int(box[3] + 25)),
                    color=color,
                    thickness=-1,
                )  # solid background rectangle
                cv2.putText(
                    cv_image,
                    text,
                    org=(int(box[2] - tw + 1), int(box[3] + 20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    thickness=2,
                    color=(255, 255, 255),
                )
            else:
                cv2.rectangle(
                    cv_image,
                    pt1=(int(box[0]), int(box[1] - 25)),
                    pt2=(int(box[0] + tw), int(box[1])),
                    color=color,
                    thickness=-1,
                )  # solid background rectangle
                cv2.putText(
                    cv_image,
                    text,
                    org=(int(box[0]), int(box[1] - 5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    thickness=2,
                    color=(255, 255, 255),
                )
            cv2.rectangle(
                cv_image,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])),
                color=color,
                thickness=2,
            )

            # cv2.rectangle(cv_image, (50, 50), (400, 400), (0, 255, 0), 2)
    else:
        # boxes = xywh2xyxy(boxes)
        color = (255, 0, 0)
        for obj in range(boxes.shape[0]):
            box = boxes[obj, :]
            # if boxes[obj, 5] == 2:
            #     color = (0, 0, 255)
            # else:
            #     color = (255, 0, 0)
            cv2.rectangle(
                cv_image,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])),
                color=color,
                thickness=2,
            )
            text = "{} {:.2f}".format(
                class_dict[int(box[5].item())], box[4].item()
            )  # TODO dynamically assign classes in a multi class scenarios
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(
                cv_image,
                pt1=(int(box[0]), int(box[1] - 25)),
                pt2=(int(box[0] + tw), int(box[1])),
                color=color,
                thickness=-1,
            )  # solid background rectangle
            cv2.rectangle(
                cv_image,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])),
                color=color,
                thickness=2,
            )
            cv2.putText(
                cv_image,
                text,
                org=(int(box[0]), int(box[1] - 5)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                thickness=2,
                color=(255, 255, 255),
            )
    return cv_image


def plot_kpts(kpts, cv_image):
    kpts = np.reshape(kpts, (kpts.shape[0], kpts.shape[2]))
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3
    radius = 5
    # kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    for i, k in enumerate(kpts):
        color_k = (255, 0, 0)
        x_coord, y_coord = k[0], k[1]
        # if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
        if len(k) == 3:
            conf = k[2]
            if conf < 0.5:
                continue
        cv2.circle(
            cv_image,
            (int(x_coord), int(y_coord)),
            radius,
            color_k,
            -1,
            lineType=cv2.LINE_AA,
        )
    # This will not be triggered for a single keypoint
    # if kpt_line:
    #     ndim = kpts.shape[-1]
    #     for i, sk in enumerate(self.skeleton):
    #         pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
    #         pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
    #         if ndim == 3:
    #             conf1 = kpts[(sk[0] - 1), 2]
    #             conf2 = kpts[(sk[1] - 1), 2]
    #             if conf1 < 0.5 or conf2 < 0.5:
    #                 continue
    #         if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
    #             continue
    #         if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
    #             continue
    #         cv2.line(cv_image, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

    return cv_image
