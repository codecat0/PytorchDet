#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :bbox_utils.py
@Author :CodeCat
@Date   :2025/11/10 15:26
"""
import math
import numpy as np
import torch


def bbox2delta(src_boxes, tgt_boxes, weights=[1.0, 1.0, 1.0, 1.0]):
    """Encode bboxes to deltas.
    Args:
        src_boxes (torch.Tensor): Source bounding boxes, shape (N, 4).
        tgt_boxes (torch.Tensor): Target bounding boxes, shape (N, 4).
        weights (list or torch.Tensor): Weights for the deltas (dx, dy, dw, dh).
                                        Defaults to [1.0, 1.0, 1.0, 1.0].
    Returns:
        torch.Tensor: Delta values, shape (N, 4).
    """
    # Ensure weights is a tensor for multiplication
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=src_boxes.dtype, device=src_boxes.device)

    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * torch.log(tgt_w / src_w)
    dh = wh * torch.log(tgt_h / src_h)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas


def delta2bbox(deltas, boxes, weights=[1.0, 1.0, 1.0, 1.0], max_shape=None):
    """Decode deltas to boxes. Used in RCNNBox,CascadeHead,RCNNHead,RetinaHead.
    Note: return tensor shape [n,1,4]
        If you want to add a reshape, please add after the calling code instead of here.
    """
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=clip_scale)
    dh = torch.clamp(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = torch.stack(pred_boxes, dim=-1)

    if max_shape is not None:
        pred_boxes[..., 0::2] = torch.clamp(pred_boxes[..., 0::2], min=0, max=max_shape[1])
        pred_boxes[..., 1::2] = torch.clamp(pred_boxes[..., 1::2], min=0, max=max_shape[0])
    return pred_boxes


def bbox2delta_v2(src_boxes,
                  tgt_boxes,
                  delta_mean=[0.0, 0.0, 0.0, 0.0],
                  delta_std=[1.0, 1.0, 1.0, 1.0]):
    """Encode bboxes to deltas.
    Modified from bbox2delta() which just use weight parameters to multiply deltas.
    Args:
        src_boxes (torch.Tensor): Source bounding boxes, shape (N, 4).
        tgt_boxes (torch.Tensor): Target bounding boxes, shape (N, 4).
        delta_mean (list or torch.Tensor): Mean values for delta normalization.
                                           Defaults to [0.0, 0.0, 0.0, 0.0].
        delta_std (list or torch.Tensor): Standard deviation values for delta normalization.
                                          Defaults to [1.0, 1.0, 1.0, 1.0].
    Returns:
        torch.Tensor: Normalized delta values, shape (N, 4).
    """
    # Ensure mean and std are tensors for the normalization operation
    if not isinstance(delta_mean, torch.Tensor):
        delta_mean = torch.tensor(delta_mean, dtype=src_boxes.dtype, device=src_boxes.device)
    if not isinstance(delta_std, torch.Tensor):
        delta_std = torch.tensor(delta_std, dtype=src_boxes.dtype, device=src_boxes.device)

    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    dx = (tgt_ctr_x - src_ctr_x) / src_w
    dy = (tgt_ctr_y - src_ctr_y) / src_h
    dw = torch.log(tgt_w / src_w)
    dh = torch.log(tgt_h / src_h)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    # Normalize the deltas using mean and standard deviation
    deltas = (deltas - delta_mean) / delta_std
    return deltas


def delta2bbox_v2(deltas,
                  boxes,
                  delta_mean=[0.0, 0.0, 0.0, 0.0],
                  delta_std=[1.0, 1.0, 1.0, 1.0],
                  max_shape=None,
                  ctr_clip=32.0):
    """Decode deltas to bboxes.
    Modified from delta2bbox() which just use weight parameters to be divided by deltas.
    Used in YOLOFHead.
    Note: return tensor shape [n,1,4]
        If you want to add a reshape, please add after the calling code instead of here.
    """
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # Denormalize the deltas using mean and standard deviation
    deltas = deltas * torch.tensor(delta_std, dtype=deltas.dtype, device=deltas.device) + torch.tensor(delta_mean,
                                                                                                       dtype=deltas.dtype,
                                                                                                       device=deltas.device)
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # Prevent sending too large values into torch.exp()
    dx = dx * widths.unsqueeze(1)
    dy = dy * heights.unsqueeze(1)
    if ctr_clip is not None:
        dx = torch.clamp(dx, min=-ctr_clip, max=ctr_clip)
        dy = torch.clamp(dy, min=-ctr_clip, max=ctr_clip)
        dw = torch.clamp(dw, max=clip_scale)
        dh = torch.clamp(dh, max=clip_scale)
    else:
        # Note: The original code had dw.clip(min=-clip_scale, max=clip_scale) etc.
        # This is now applied to dw and dh regardless of ctr_clip being None.
        dw = torch.clamp(dw, min=-clip_scale, max=clip_scale)
        dh = torch.clamp(dh, min=-clip_scale, max=clip_scale)

    pred_ctr_x = dx + ctr_x.unsqueeze(1)
    pred_ctr_y = dy + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = torch.stack(pred_boxes, dim=-1)

    if max_shape is not None:
        pred_boxes[..., 0::2] = torch.clamp(pred_boxes[..., 0::2], min=0, max=max_shape[1])
        pred_boxes[..., 1::2] = torch.clamp(pred_boxes[..., 1::2], min=0, max=max_shape[0])
    return pred_boxes


def expand_bbox(bboxes, scale):
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    bboxes_exp = np.zeros(bboxes.shape, dtype=np.float32)
    bboxes_exp[:, 0] = x_c - w_half
    bboxes_exp[:, 2] = x_c + w_half
    bboxes_exp[:, 1] = y_c - h_half
    bboxes_exp[:, 3] = y_c + h_half

    return bboxes_exp


def clip_bbox(boxes, im_shape):
    h, w = im_shape[0], im_shape[1]
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    """
    Find bounding boxes that are non-empty (width and height greater than min_size).

    Args:
        boxes (torch.Tensor): Bounding boxes, shape (N, 4) in [x1, y1, x2, y2] format.
        min_size (float or int): Minimum size for width and height. Defaults to 0.
        return_mask (bool): If True, returns a boolean mask; if False, returns indices. Defaults to False.

    Returns:
        torch.Tensor: Boolean mask (if return_mask=True) or indices (if return_mask=False).
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    # Use torch.logical_and for element-wise AND operation
    mask = torch.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    # Use torch.nonzero and flatten to get indices
    keep = torch.nonzero(mask).flatten()
    return keep


def bbox_area(boxes):
    """Calculate the area of bounding boxes.
    Args:
        boxes (torch.Tensor): Bounding boxes, shape (N, 4) in [x1, y1, x2, y2] format.
    Returns:
        torch.Tensor: Areas of the boxes, shape (N,).
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    return areas


def bbox_overlaps(boxes1, boxes2):
    """
    Calculate overlaps (IoU) between boxes1 and boxes2.

    Args:
        boxes1 (torch.Tensor): boxes with shape [M, 4]
        boxes2 (torch.Tensor): boxes with shape [N, 4]

    Return:
        overlaps (torch.Tensor): overlaps between boxes1 and boxes2 with shape [M, N]
    """
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    if M * N == 0:
        # Return zeros with correct dtype and device
        return torch.zeros((M, N), dtype=torch.float32, device=boxes1.device)

    area1 = bbox_area(boxes1)  # Shape: [M]
    area2 = bbox_area(boxes2)  # Shape: [N]

    # Expand boxes1 to [M, 1, 4] and boxes2 to [N, 4]
    # Then broadcast for min/max calculation
    # For intersection coordinates:
    # - Max of top-left corners ([:, :, :2] for x1, y1)
    # - Min of bottom-right corners ([:, :, 2:] for x2, y2)
    xy_max = torch.minimum(
        torch.unsqueeze(boxes1, 1)[:, :, 2:],  # [M, 1, 2] (x2, y2 of boxes1)
        boxes2[:, 2:]  # [N, 2] (x2, y2 of boxes2)
    )  # Result: [M, N, 2] (x2_inter, y2_inter)
    xy_min = torch.maximum(
        torch.unsqueeze(boxes1, 1)[:, :, :2],  # [M, 1, 2] (x1, y1 of boxes1)
        boxes2[:, :2]  # [N, 2] (x1, y1 of boxes2)
    )  # Result: [M, N, 2] (x1_inter, y1_inter)

    width_height = xy_max - xy_min  # [M, N, 2] (width_inter, height_inter)
    width_height = torch.clamp(width_height, min=0)  # [M, N, 2], clamped to >= 0
    inter = width_height.prod(dim=2)  # [M, N], intersection area

    # Calculate IoU: intersection / (area1 + area2 - intersection)
    # area1.unsqueeze(1) -> [M, 1], area2 -> [N]
    # Broadcasting: [M, 1] + [N] - [M, N] -> [M, N]
    union = torch.unsqueeze(area1, 1) + area2 - inter  # [M, N], union area
    overlaps = torch.where(
        inter > 0,
        inter / union,  # Calculate IoU where intersection > 0
        torch.zeros_like(inter)  # Return 0 where intersection <= 0
    )
    return overlaps


def batch_bbox_overlaps(bboxes1,
                        bboxes2,
                        mode='iou',
                        is_aligned=False,
                        eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground) or "giou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (B, m, n) if ``is_aligned `` is False else shape (B, m,)
    """
    assert mode in ['iou', 'iof', 'giou'], 'Unsupported mode {}'.format(mode)
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    # Note: This assertion might need adjustment based on how empty tensors are handled
    # assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0) # This check might be too strict for batched dims
    # assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0) # This check might be too strict for batched dims
    # A more robust check considering batch dimensions:
    assert bboxes1.ndim >= 2 and bboxes2.ndim >= 2
    assert bboxes1.shape[-1] == 4 and bboxes2.shape[-1] == 4

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn) -> (B,)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    # Handle case where m or n could be 0 for the non-batch dimensions
    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return torch.full(batch_shape + (rows,), 1.0, dtype=torch.float32, device=bboxes1.device)
        else:
            return torch.full(batch_shape + (rows, cols), 1.0, dtype=torch.float32, device=bboxes1.device)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])  # [B, rows] or [rows] if no batch
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])  # [B, cols] or [cols] if no batch

    if is_aligned:
        # bboxes1: [B, rows, 4], bboxes2: [B, cols, 4] -> [B, rows, 4] (assuming cols == rows)
        # For is_aligned, we assume the last dimension before coordinates is the same (rows==cols)
        lt = torch.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = torch.clamp((rb - lt), min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]  # [B, rows]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap  # [B, rows]
        else:  # mode == 'iof'
            union = area1  # [B, rows]
        if mode == 'giou':
            enclosed_lt = torch.minimum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
            enclosed_rb = torch.maximum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]
    else:
        # bboxes1: [B, rows, 4], bboxes2: [B, cols, 4]
        # Use unsqueeze to add dimensions for broadcasting: [B, rows, 1, 2] vs [B, 1, cols, 2]
        lt = torch.maximum(bboxes1[..., :2].unsqueeze(-2), bboxes2[..., :2].unsqueeze(-3))  # [B, rows, cols, 2]
        rb = torch.minimum(bboxes1[..., 2:].unsqueeze(-2), bboxes2[..., 2:].unsqueeze(-3))  # [B, rows, cols, 2]

        wh = torch.clamp((rb - lt), min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]  # [B, rows, cols]

        if mode in ['iou', 'giou']:
            # Broadcasting: [B, rows, 1] + [B, 1, cols] -> [B, rows, cols]
            union = area1.unsqueeze(-1) + area2.unsqueeze(-2) - overlap
        else:  # mode == 'iof'
            # Broadcasting: [B, rows, 1] -> [B, rows, cols]
            union = area1.unsqueeze(-1)
        if mode == 'giou':
            enclosed_lt = torch.minimum(bboxes1[..., :2].unsqueeze(-2),
                                        bboxes2[..., :2].unsqueeze(-3))  # [B, rows, cols, 2]
            enclosed_rb = torch.maximum(bboxes1[..., 2:].unsqueeze(-2),
                                        bboxes2[..., 2:].unsqueeze(-3))  # [B, rows, cols, 2]

    eps_tensor = torch.tensor([eps], dtype=torch.float32, device=bboxes1.device)
    union = torch.maximum(union, eps_tensor)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious  # Shape: [B, rows, cols] or [B, rows] if is_aligned

    # Calculate gious
    enclose_wh = torch.clamp((enclosed_rb - enclosed_lt), min=0)  # [B, rows, cols, 2] or [B, rows, 2] if is_aligned
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]  # [B, rows, cols] or [B, rows] if is_aligned
    enclose_area = torch.maximum(enclose_area, eps_tensor)
    gious = ious - (enclose_area - union) / enclose_area
    return 1 - gious  # Return 1 - GIoU for consistency if needed by the caller, or just gious


def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return [x1, y1, x2, y2]


def make_grid(h, w, dtype):
    yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    # Stack along the last dimension (dim=2)
    grid = torch.stack((xv, yv), dim=2)
    # Cast to the specified dtype
    return grid.to(dtype)


def decode_yolo(box, anchor, downsample_ratio):
    """decode yolo box

    Args:
        box (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        anchor (list or torch.Tensor): anchor with the shape [na, 2]
        downsample_ratio (int): downsample ratio, default 32

    Return:
        box (list): decoded box, [x, y, w, h], all have the shape [b, na, h, w, 1]
    """
    x, y, w, h = box
    # Get dimensions from the input tensors
    na, grid_h, grid_w = x.shape[1:4]

    # Create the grid using the helper function (assumed to be converted to PyTorch)
    # Shape: [grid_h, grid_w, 2] -> [1, 1, grid_h, grid_w, 2]
    grid = make_grid(grid_h, grid_w, x.dtype).reshape((1, 1, grid_h, grid_w, 2))

    # Decode x and y coordinates (sigmoid output + grid offset) / grid size
    x1 = (x + grid[:, :, :, :, 0:1]) / grid_w
    y1 = (y + grid[:, :, :, :, 1:2]) / grid_h

    # Convert anchor to tensor and reshape for broadcasting
    # Shape: [na, 2] -> [1, na, 1, 1, 2]
    anchor_tensor = torch.tensor(anchor, dtype=x.dtype, device=x.device)
    anchor_tensor = anchor_tensor.reshape((1, na, 1, 1, 2))

    # Decode w and h (exp of log-space output * anchor) / (downsample_ratio * grid size)
    w1 = torch.exp(w) * anchor_tensor[:, :, :, :, 0:1] / (downsample_ratio * grid_w)
    h1 = torch.exp(h) * anchor_tensor[:, :, :, :, 1:2] / (downsample_ratio * grid_h)

    return [x1, y1, w1, h1]


def batch_iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2 in batch

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    # Use torch.maximum and torch.minimum
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    # Use torch.clamp instead of .clip
    overlap = (x2y2 - x1y1).clamp(min=0).prod(dim=-1)
    area1 = (px2y2 - px1y1).clamp(min=0).prod(dim=-1)
    area2 = (gx2y2 - gx1y1).clamp(min=0).prod(dim=-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x1, y1, x2, y2], all have the shape [b, na, h, w, 1] or broadcastable shapes
        box2 (list): [x1, y1, x2, y2], all have the shape [b, na, h, w, 1] or broadcastable shapes
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box2, with the shape [b, na, h, w, 1] or broadcasted shape
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2

    # Calculate intersection coordinates
    x1 = torch.maximum(px1, gx1)
    y1 = torch.maximum(py1, gy1)
    x2 = torch.minimum(px2, gx2)
    y2 = torch.minimum(py2, gy2)

    # Calculate overlap area
    overlap = ((x2 - x1).clamp(min=0)) * ((y2 - y1).clamp(min=0))

    # Calculate areas of the two boxes
    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clamp(min=0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clamp(min=0)

    # Calculate union area
    union = area1 + area2 - overlap + eps
    # Calculate IoU
    iou = overlap / union

    # GIoU, DIoU, CIoU calculations
    if giou or ciou or diou:
        # Calculate convex hull width and height (cw, ch)
        cw = torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
        ch = torch.maximum(py2, gy2) - torch.minimum(py1, gy1)

        if giou:
            # Calculate GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # Calculate convex diagonal squared (c2)
            c2 = cw ** 2 + ch ** 2 + eps
            # Calculate center distance squared (rho2)
            rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4

            if diou:
                # Calculate DIoU
                return iou - rho2 / c2
            else:  # ciou
                # Calculate aspect ratio factor (v) and alpha
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                # Ensure h1 and h2 are not zero before division, though eps should handle it
                # atan2 might be more robust than atan, but atan is fine here as w, h are positive
                delta = torch.atan(w1 / h1) - torch.atan(w2 / h2)
                v = (4 / (math.pi ** 2)) * torch.pow(delta, 2)
                # Calculate alpha
                alpha = v / (1 + eps - iou + v)
                # In PyTorch, alpha is a tensor that will be used in the final calculation.
                # Its gradients will be computed automatically if its inputs (delta, v, iou) require gradients.
                # If you specifically want alpha not to require gradients (equivalent to stop_gradient),
                # you could use: alpha = alpha.detach()
                # However, for CIoU loss, it's usually fine to let it compute gradients.
                # If needed: alpha = alpha.detach().requires_grad_(False) # This detaches and sets requires_grad=False
                # For standard usage, just calculate and use:
                return iou - (rho2 / c2 + v * alpha)
    else:
        # Return standard IoU
        return iou


def bbox_iou_np_expand(box1, box2, x1y1x2y2=True, eps=1e-16):
    """
    Calculate the iou of box1 and box2 with numpy.

    Args:
        box1 (ndarray): [N, 4]
        box2 (ndarray): [M, 4], usually N != M
        x1y1x2y2 (bool): whether in x1y1x2y2 stype, default True
        eps (float): epsilon to avoid divide by zero
    Return:
        iou (ndarray): iou of box1 and box2, [N, M]
    """
    N, M = len(box1), len(box2)  # usually N != M
    if x1y1x2y2:
        b1_x1, b1_y1 = box1[:, 0], box1[:, 1]
        b1_x2, b1_y2 = box1[:, 2], box1[:, 3]
        b2_x1, b2_y1 = box2[:, 0], box2[:, 1]
        b2_x2, b2_y2 = box2[:, 2], box2[:, 3]
    else:
        # cxcywh style
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.zeros((N, M), dtype=np.float32)
    inter_rect_y1 = np.zeros((N, M), dtype=np.float32)
    inter_rect_x2 = np.zeros((N, M), dtype=np.float32)
    inter_rect_y2 = np.zeros((N, M), dtype=np.float32)
    for i in range(len(box2)):
        inter_rect_x1[:, i] = np.maximum(b1_x1, b2_x1[i])
        inter_rect_y1[:, i] = np.maximum(b1_y1, b2_y1[i])
        inter_rect_x2[:, i] = np.minimum(b1_x2, b2_x2[i])
        inter_rect_y2[:, i] = np.minimum(b1_y2, b2_y2[i])
    # Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(
        inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = np.repeat(
        ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).reshape(-1, 1), M, axis=-1)
    b2_area = np.repeat(
        ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).reshape(1, -1), N, axis=0)

    ious = inter_area / (b1_area + b2_area - inter_area + eps)
    return ious


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=
    Returns:
        Tensor: Decoded distances, shape (n, 4) [left, top, right, bottom].
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], dim=-1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (n, 4).
        max_shape (tuple): Shape of the image (height, width).
    Returns:
        Tensor: Decoded bboxes, shape (n, 4) [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])  # Clamp x1 to [0, width-1]
        y1 = y1.clamp(min=0, max=max_shape[0])  # Clamp y1 to [0, height-1]
        x2 = x2.clamp(min=0, max=max_shape[1])  # Clamp x2 to [0, width-1]
        y2 = y2.clamp(min=0, max=max_shape[0])  # Clamp y2 to [0, height-1]
    return torch.stack([x1, y1, x2, y2], dim=-1)

def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)

def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    # Split distance tensor along the last dimension (-1) into 2 parts
    lt, rb = torch.split(distance, 2, dim=-1) # dim=-1 specifies the last dimension
    # Calculate x1y1 and x2y2
    x1y1 = -lt + points
    x2y2 = rb + points
    # Concatenate x1y1 and x2y2 along the last dimension (-1)
    out_bbox = torch.cat([x1y1, x2y2], dim=-1) # dim=-1 specifies the last dimension

    if max_shapes is not None:
        # Flip max_shapes along the last dimension to get [w, h]
        max_shapes = torch.flip(max_shapes, [-1])
        # Repeat max_shapes along the last dimension to get [w, h, w, h]
        max_shapes = max_shapes.repeat([1, 2])
        # Calculate the difference in dimensions
        delta_dim = out_bbox.dim() - max_shapes.dim() # Use .dim() instead of .ndim
        # Expand max_shapes dimensions by adding size-1 dimensions at the beginning
        # Use unsqueeze in a loop, as unsqueeze is not in-place
        for _ in range(delta_dim):
            max_shapes = max_shapes.unsqueeze(1) # unsqueeze returns a new tensor
        # Clamp out_bbox values using torch.where
        # First, ensure values are not greater than max_shapes
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        # Then, ensure values are not less than 0
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))

    return out_bbox


def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    # Use torch.maximum and torch.minimum
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    # Use torch.clamp instead of .clip
    overlap = (x2y2 - x1y1).clamp(min=0).prod(dim=-1)
    area1 = (px2y2 - px1y1).clamp(min=0).prod(dim=-1)
    area2 = (gx2y2 - gx1y1).clamp(min=0).prod(dim=-1)
    union = area1 + area2 - overlap + eps
    return overlap / union