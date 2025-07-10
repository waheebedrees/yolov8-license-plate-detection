import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union
import torch.nn as nn
import torchvision
import math

def xywh2xyxy(xywh: Union[np.ndarray, torch.Tensor]):
    """
    Convert bounding box coordinates from (xywh) to (xyxy)
    """
    if isinstance(xywh, np.ndarray):
        xy, wh = np.split(xywh, 2, axis=-1)
        return np.concatenate((xy - wh / 2, xy + wh / 2), axis=-1)
    else:
        xy, wh = torch.chunk(xywh, 2, dim=-1)
        return torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)

def xyxy2xywh(xyxy: Union[np.ndarray, torch.Tensor]):
    """
    Convert bounding box coordinates from (xyxy) to (xywh)
    """
    if isinstance(xyxy, np.ndarray):
        xy_lt, xy_rb = np.split(xyxy, 2, axis=-1)
        return np.concatenate(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), axis=-1)
    else:
        xy_lt, xy_rb = torch.chunk(xyxy, 2, dim=-1)
        return torch.cat(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), dim=-1)

def pad_to(x: torch.Tensor, stride: int = None, shape: Tuple[int, int] = None):
    """
    Pads an image with zeros to make it divisible by stride
    (Pads both top/bottom and left/right evenly) or pads to
    specified shape.

    Args:
        x (Tensor): image tensor of shape (..., h, w)
        stride (optional, int): stride of model
        shape (optional, Tuple[int,int]): shape to pad image to
    """
    h, w = x.shape[-2:]

    if stride is not None:
        h_new = h if h % stride == 0 else h + stride - h % stride
        w_new = w if w % stride == 0 else w + stride - w % stride
    elif shape is not None:
        h_new, w_new = shape

    t, b = int((h_new - h) / 2), int(h_new - h) - int((h_new - h) / 2)
    l, r = int((w_new - w) / 2), int(w_new - w) - int((w_new - w) / 2)
    pads = (l, r, t, b)

    x_padded = F.pad(x, pads, "constant", 0)

    return x_padded, pads

def unpad(x: torch.Tensor, pads: tuple):
    l, r, t, b = pads
    return x[..., t:-b, l:-r]

def pad_xyxy(xyxy: Union[np.ndarray, torch.Tensor], pads: Tuple[int, int, int, int], im_size: Tuple[int, int] = None, return_norm: bool = False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xyxy: The bounding boxes in the format of `(x_min, y_min, x_max, y_max)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")

    if im_size is not None:
        h, w = im_size
        hpad, wpad = h + b + t, w + l + r

    if isinstance(xyxy, np.ndarray):
        xyxy_unnorm = xyxy * np.array([w, h, w, h], dtype=xyxy.dtype) if im_size else xyxy
        padded = xyxy_unnorm + np.array([l, t, l, t], dtype=xyxy.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xyxy.dtype)
        return padded

    xyxy_unnorm = xyxy * torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device) if im_size else xyxy
    padded = xyxy_unnorm + torch.tensor([l, t, l, t], dtype=xyxy.dtype, device=xyxy.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad], dtype=xyxy.dtype, device=xyxy.device)
    return padded

def pad_xywh(xywh: Union[np.ndarray, torch.Tensor], pads: Tuple[int, int, int, int], im_size: Tuple[int, int] = None, return_norm: bool = False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xywh: The bounding boxes in the format of `(x, y, w, h)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")

    if im_size is not None:
        h, w = im_size
        hpad, wpad = h + b + t, w + l + r

    if isinstance(xywh, np.ndarray):
        xywh_unnorm = xywh * np.array([w, h, w, h], dtype=xywh.dtype) if im_size else xywh
        padded = xywh_unnorm + np.array([l, t, 0, 0], dtype=xywh.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xywh.dtype)
        return padded

    xywh_unnorm = xywh * torch.tensor([w, h, w, h], dtype=xywh.dtype, device=xywh.device) if im_size else xywh
    padded = xywh_unnorm + torch.tensor([l, t, 0, 0], dtype=xywh.dtype, device=xywh.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad], dtype=xywh.dtype, device=xywh.device)
    return padded

def unpad_xyxy(xyxy: Union[np.ndarray, torch.Tensor], pads: Tuple[int, int, int, int]):
    """
    Remove padding from the bounding boxes based on image padding

    Args:
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
    """
    l, r, t, b = pads
    if isinstance(xyxy, np.ndarray):
        return xyxy - np.array([l, t, l, t], dtype=xyxy.dtype)
    return xyxy - torch.tensor([l, t, l, t], dtype=xyxy.dtype, device=xyxy.device)

def autopad(kernel_size: int, padding: int = None):
    """
    Calculate padding size automatically
    """
    if padding is None:
        return kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding

def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1):
    """
    Transform distance in (ltrb) to bounding box (xywh) or (xyxy)
    """
    lt, rb = torch.chunk(distance, 2, dim=dim)
    xy_lt = anchor_points - lt
    xy_rb = anchor_points + rb

    if xywh:
        center = (xy_lt + xy_rb) / 2
        wh = xy_rb - xy_lt
        return torch.cat((center, wh), dim=dim)

    return torch.cat((xy_lt, xy_rb), dim=dim)

def bbox2dist(bbox: torch.Tensor, anchor_points: torch.Tensor, reg_max: int):
    """
    Transform bounding box (xyxy) to distance (ltrb)
    """
    xy_lt, xy_rb = torch.chunk(bbox, 2, dim=-1)
    lt = anchor_points - xy_lt
    rb = xy_rb - anchor_points
    return torch.cat((lt, rb), dim=-1).clamp(max=reg_max - 0.01)


def init_weights(model:  nn.Module):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in (nn.ReLU, nn.SiLU):
            m.inplace = True

def make_anchors(feats: torch.Tensor, strides: torch.Tensor):
    anchor_points = []
    stride_tensor = []

    device = feats[0].device
    dtype = feats[0].dtype

    for i, stride in enumerate(strides):
        h, w = feats[i].shape[-2:]
        yv, xv = torch.meshgrid(torch.arange(h).to(device=device, dtype=dtype) + 0.5,
                                torch.arange(w).to(device=device, dtype=dtype) + 0.5)

        # (x,y) coordinates of center of each cell in grid
        anchor_points.append(torch.stack((xv, yv), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride).to(device=device, dtype=dtype))

    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)

    return anchor_points, stride_tensor

def nms(preds: torch.Tensor, confidence_thresh: float = 0.25, iou_thresh: float = 0.45) -> list[torch.Tensor]:
    """
    Non-Maximum Suppression for predicted boxes and classes

    Args:
        preds (Tensor): Predictions from model of shape (bs, 4 + num_classes, num_boxes)
        confidence_thresh (float, optional): Confidence threshold. Defaults to 0.25
        iou_thresh (float, optional): IoU threshold. Defaults to 0.45

    Returns:
        List[Tensor]: list of tensors of shape (num_boxes, 6) containing boxes with
            (x1, y1, x2, y2, confidence, class)
    """
    b, nc, _ = preds.shape
    nc -= 4

    # max confidence score among boxes
    xc = preds[:, 4:].amax(dim=1) > confidence_thresh

    # (b, 4+nc, a) -> (b, a, 4+nc)
    preds = preds.transpose(-1, -2)

    preds[..., :4] = xywh2xyxy(preds[..., :4])

    out = [torch.zeros((0, 6), device=preds.device)] * b

    for i, x in enumerate(preds):
        # take max cls confidence score
        # only consider predictions with confidence > confidence_thresh
        x = x[xc[i]]

        # If there are no remaining predictions, move to next image
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), dim=1)

        confidences, cls_idxs = cls.max(dim=1, keepdim=True)
        x = torch.cat((box, confidences, cls_idxs.float()), dim=1)

        keep_idxs = torchvision.ops.nms(x[:, :4], x[:, 4], iou_thresh)

        out[i] = x[keep_idxs]

    return out

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-10, iou_mode: str = "iou"):
    """
    Calculate IoU between two bounding boxes

    Args:
        box1: (Tensor) with shape (..., 1 or n, 4)
        box2: (Tensor) with shape (..., n, 4)
        xywh: (bool) True if box coordinates are in (xywh) else (xyxy)
        eps: (float) epsilon to prevent division by zero
        iou_mode: (str) IoU mode to use (\"iou\", \"giou\", \"diou\", \"ciou\")

    Returns:
        iou: (Tensor) with IoU
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    else:
        (b1_x1, b1_y1, b1_x2, b1_y2), (b2_x1, b2_y1, b2_x2, b2_y2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    intersection = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0) * \
                   (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)

    # Union Area
    union = w1 * h1 + w2 * h2 - intersection + eps

    # IoU
    iou = intersection / union

    if iou_mode == "giou":
        # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
        return iou - (c_area - union) / c_area

    elif iou_mode == "diou":
        # Distance IoU https://arxiv.org/pdf/1911.08287.pdf
        center_b1_x = (b1_x1 + b1_x2) / 2
        center_b1_y = (b1_y1 + b1_y2) / 2
        center_b2_x = (b2_x1 + b2_x2) / 2
        center_b2_y = (b2_y1 + b2_y2) / 2

        # Euclidean distance between centers
        rho2 = ((center_b1_x - center_b2_x) ** 2) + ((center_b1_y - center_b2_y) ** 2)

        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)

        # Diagonal distance of the smallest enclosing box
        c2 = ((c_x2 - c_x1) ** 2) + ((c_y2 - c_y1) ** 2) + eps
        return iou - rho2 / c2

    elif iou_mode == "ciou":
        # Complete IoU https://arxiv.org/pdf/1911.08287.pdf
        center_b1_x = (b1_x1 + b1_x2) / 2
        center_b1_y = (b1_y1 + b1_y2) / 2
        center_b2_x = (b2_x1 + b2_x2) / 2
        center_b2_y = (b2_y1 + b2_y2) / 2

        rho2 = ((center_b1_x - center_b2_x) ** 2) + ((center_b1_y - center_b2_y) ** 2)

        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)

        c2 = ((c_x2 - c_x1) ** 2) + ((c_y2 - c_y1) ** 2) + eps

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1) - torch.atan(w2 / h2), 2)
        alpha = v / (1 - iou + v + eps)
        return iou - (rho2 / c2 + alpha * v)

    return iou


def df_loss(pred_box_dist: torch.Tensor, targets: torch.Tensor):
    """
    Sum of left and right DFL losses
    """
    target_left = targets.long()
    target_right = target_left + 1
    weight_left = target_right - targets
    weight_right = 1 - weight_left

    dfl_left = F.cross_entropy(pred_box_dist, target_left.view(-1),
                               reduction='none').view(target_left.shape) * weight_left
    dfl_right = F.cross_entropy(pred_box_dist, target_right.view(-1),
                                reduction='none').view(target_right.shape) * weight_right

    return torch.mean(dfl_left + dfl_right, dim=-1, keepdim=True)




def anchors_in_gt_boxes(anchor_points: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-8):
    """
    Returns mask for positive anchor centers that are in GT boxes

    Args:
        anchor_points (Tensor): Anchor points of shape (n_anchors, 2)
        gt_boxes (Tensor): GT boxes of shape (batch_size, n_boxes, 4)

    Returns:
        mask (Tensor): Mask of shape (batch_size, n_boxes, n_anchors)
    """
    n_anchors = anchor_points.shape[0]
    batch_size, n_boxes, _ = gt_boxes.shape
    lt, rb = gt_boxes.view(-1, 1, 4).chunk(chunks=2, dim=2)
    box_deltas = torch.cat((anchor_points.unsqueeze(0) - lt, rb - anchor_points.unsqueeze(0)), dim=2).view(batch_size, n_boxes, n_anchors, -1)
    return torch.amin(box_deltas, dim=3) > eps

def select_highest_iou(mask: torch.Tensor, ious: torch.Tensor, num_max_boxes: int):
    """
    Select GT box with highest IoU for each anchor

    Args:
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors)
        ious (Tensor): IoU of shape (batch_size, num_max_boxes, n_anchors)

    Returns:
        target_gt_box_idxs (Tensor): Index of GT box with highest IoU for each anchor of shape (batch_size, n_anchors)
        fg_mask (Tensor): Mask of shape (batch_size, n_anchors) where 1 indicates positive anchor
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors) where 1 indicates positive anchor
    """
    # sum over n_max_boxes dim to get num GT boxes assigned to each anchor
    # (batch_size, num_max_boxes, n_anchors) -> (batch_size, n_anchors)
    fg_mask = mask.sum(dim=1)

    if fg_mask.max() > 1:
        # If 1 anchor assigned to more than one GT box, select the one with highest IoU
        max_iou_idx = ious.argmax(dim=1)  # (batch_size, n_anchors)

        # mask for where there are more than one GT box assigned to anchor
        multi_gt_mask = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (batch_size, num_max_boxes, n_anchors)

        # mask for GT box with highest IoU
        max_iou_mask = torch.zeros_like(mask, dtype=torch.bool)
        max_iou_mask.scatter_(dim=1, index=max_iou_idx.unsqueeze(1), value=1)

        mask = torch.where(multi_gt_mask, max_iou_mask, mask)
        fg_mask = mask.sum(dim=1)

    target_gt_box_idxs = mask.argmax(dim=1)  # (batch_size, n_anchors)
    return target_gt_box_idxs, fg_mask, mask


