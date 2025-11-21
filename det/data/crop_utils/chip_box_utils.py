#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :chip_box_utils.py
@Author :CodeCat
@Date   :2025/11/21 15:39
"""
import numpy as np


def bbox_area(boxes):
    """
    计算边界框的面积

    Args:
        boxes (np.ndarray or torch.Tensor): 边界框坐标，形状为 (N, 4)
                                           格式为 [x1, y1, x2, y2]

    Returns:
        计算出的面积，形状为 (N,)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection_over_box(chips, boxes):
    """
    计算chips与boxes的交集面积相对于box面积的比例 (intersection over box)

    Args:
        chips (np.ndarray or torch.Tensor): 切片边界框，形状为 (C, 4)，格式为 [x1, y1, x2, y2]
        boxes (np.ndarray or torch.Tensor): 原始边界框，形状为 (B, 4)，格式为 [x1, y1, x2, y2]

    Returns:
        np.ndarray: IOB矩阵，形状为 (C, B)，其中 iob[c, b] 表示 chips[c] 与 boxes[b] 的交集面积除以 boxes[b] 的面积
    """
    M = chips.shape[0]  # 切片数量
    N = boxes.shape[0]  # 原始框数量
    if M * N == 0:
        return np.zeros([M, N], dtype='float32')

    # 计算原始框的面积
    box_area = bbox_area(boxes)  # 形状为 (B,)

    # 计算交集区域的右下角坐标
    inter_x2y2 = np.minimum(np.expand_dims(chips, 1)[:, :, 2:],
                            boxes[:, 2:])  # 形状为 (C, B, 2)
    # 计算交集区域的左上角坐标
    inter_x1y1 = np.maximum(np.expand_dims(chips, 1)[:, :, :2],
                            boxes[:, :2])  # 形状为 (C, B, 2)
    # 计算交集区域的宽度和高度
    inter_wh = inter_x2y2 - inter_x1y1
    inter_wh = np.clip(inter_wh, a_min=0, a_max=None)  # 确保宽高不为负数
    # 计算交集面积
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # 形状为 (C, B)

    # 计算IOB：交集面积除以box面积
    iob = inter_area / np.expand_dims(box_area, 0)  # 将box_area扩展为 (1, B) 以便广播
    return iob


def clip_boxes(boxes, im_shape):
    """
    将边界框裁剪到图像边界内

    Args:
        boxes (np.ndarray or torch.Tensor): 边界框坐标，形状为 [N, 4]
                                           格式为 [x1, y1, x2, y2]
        im_shape (tuple): 图像形状，格式为 (h, w)

    Returns:
        裁剪后的边界框，形状为 [N, 4]
    """
    # x1 >= 0
    boxes[:, 0] = np.clip(boxes[:, 0], 0, im_shape[1] - 1)
    # y1 >= 0
    boxes[:, 1] = np.clip(boxes[:, 1], 0, im_shape[0] - 1)
    # x2 < im_shape[1]
    boxes[:, 2] = np.clip(boxes[:, 2], 1, im_shape[1])
    # y2 < im_shape[0]
    boxes[:, 3] = np.clip(boxes[:, 3], 1, im_shape[0])
    return boxes


def transform_chip_box(gt_bbox: 'Gx4', boxes_idx: 'B', chip: '4'):
    """
    将原始图像中的边界框坐标转换到图像切片（chip）的局部坐标系，并过滤有效框。

    Args:
        gt_bbox (np.ndarray): 原始图像中的所有地面真值边界框，形状为 (G, 4)，格式为 [x1, y1, x2, y2]。
        boxes_idx (list or np.ndarray): 需要转换的边界框索引列表，形状为 (B,)。
        chip (list or tuple): 图像切片在原始图像中的坐标，格式为 [x1, y1, x2, y2]。

    Returns:
        tuple: 包含两个元素:
               - np.ndarray: 转换并裁剪后的有效边界框，形状为 (V, 4)，格式为 [x1, y1, x2, y2]，其中 V <= B。
               - np.ndarray: 对应的有效边界框在原始 gt_bbox 中的索引，形状为 (V,)。
    """
    boxes_idx = np.array(boxes_idx)
    # 根据索引提取对应的边界框，并创建副本以避免修改原始数据
    cur_gt_bbox = gt_bbox[boxes_idx].copy()  # 形状为 (B, 4)

    # 从切片坐标中提取左上角和右下角坐标
    x1, y1, x2, y2 = chip

    # 将边界框坐标从原始图像坐标系转换到切片的局部坐标系
    cur_gt_bbox[:, 0] -= x1  # x1
    cur_gt_bbox[:, 1] -= y1  # y1
    cur_gt_bbox[:, 2] -= x1  # x2
    cur_gt_bbox[:, 3] -= y1  # y2

    # 获取切片的高度和宽度
    h = y2 - y1
    w = x2 - x1

    # 将边界框裁剪到切片边界内
    cur_gt_bbox = clip_boxes(cur_gt_bbox, (h, w))

    # 计算裁剪后边界框的宽度和高度
    ws = (cur_gt_bbox[:, 2] - cur_gt_bbox[:, 0]).astype(np.int32)
    hs = (cur_gt_bbox[:, 3] - cur_gt_bbox[:, 1]).astype(np.int32)

    # 过滤掉宽度或高度小于2像素的边界框
    valid_idx = (ws >= 2) & (hs >= 2)

    # 返回有效的边界框及其原始索引
    return cur_gt_bbox[valid_idx], boxes_idx[valid_idx]


def find_chips_to_cover_overlaped_boxes(iob, overlap_threshold):
    """
    根据IOB（Intersection over Box）矩阵和重叠阈值，选择最少的chips来覆盖所有重叠的boxes。

    该函数使用贪心算法：在每一步中，选择能覆盖最多剩余boxes的chip，直到所有重叠的boxes都被覆盖。

    Args:
        iob (np.ndarray): IOB矩阵，形状为 (C, B)，其中 C 是chip数量，B 是box数量。
                          iob[c, b] 表示chip c与box b的交集面积相对于box b面积的比例。
        overlap_threshold (float): 重叠阈值，只有当 iob[c, b] >= threshold 时，
                                   才认为chip c与box b有重叠。

    Returns:
        tuple: 包含两个元素:
               - list: 被选择的chip ID列表。
               - np.ndarray: 一维数组，长度为C，表示每个chip与多少个boxes的重叠度超过了阈值。
    """
    # 找到所有重叠度超过阈值的chip和box对的索引
    chip_ids, box_ids = np.nonzero(iob >= overlap_threshold)
    # 统计每个chip与多少个boxes有重叠
    chip_id2overlap_box_num = np.bincount(chip_ids)  # 1d array, 长度为 max(chip_ids)+1
    # 将统计结果填充到与iob中chip数量相同的长度（len(iob)），缺失部分填充0
    chip_id2overlap_box_num = np.pad(
        chip_id2overlap_box_num,
        (0, len(iob) - len(chip_id2overlap_box_num)), # (pad_before, pad_after)
        constant_values=0)

    chosen_chip_ids = []
    # 贪心选择：每次选择能覆盖最多剩余boxes的chip
    while len(box_ids) > 0:
        # 统计当前剩余的chip_ids中，每个chip_id出现的次数（即能覆盖多少个剩余的boxes）
        value_counts = np.bincount(chip_ids)  # 1d array, 长度为 max(chip_ids)+1
        # 找到能覆盖最多剩余boxes的chip_id
        max_count_chip_id = np.argmax(value_counts)
        # 确保该chip_id未被选择过（虽然argmax通常会选择最小的索引，但逻辑上需要保证）
        assert max_count_chip_id not in chosen_chip_ids
        chosen_chip_ids.append(max_count_chip_id)

        # 找到当前选择的chip覆盖的所有boxes
        box_ids_in_cur_chip = box_ids[chip_ids == max_count_chip_id]
        # 创建掩码，标记哪些 (chip_id, box_id) 对不属于当前选择的chip覆盖的boxes
        ids_not_in_cur_boxes_mask = np.logical_not(
            np.isin(box_ids, box_ids_in_cur_chip)) # 长度为当前剩余 (chip_id, box_id) 对的数量
        # 过滤掉已被当前chip覆盖的boxes，更新剩余的 (chip_id, box_id) 对
        chip_ids = chip_ids[ids_not_in_cur_boxes_mask]
        box_ids = box_ids[ids_not_in_cur_boxes_mask]
    return chosen_chip_ids, chip_id2overlap_box_num


def transform_chip_boxes2image_boxes(chip_boxes, chip, img_h, img_w):
    """
    将图像切片（chip）上的边界框坐标转换回原始图像坐标系，并进行裁剪。

    Args:
        chip_boxes (list or np.ndarray): 切片上的边界框列表或数组，形状为 (N, 6) 或类似格式。
                                         假设格式为 [x1, y1, x2, y2, score, class] 或 [x1, y1, w, h, score, class]。
                                         函数会根据后四个值确定边界框坐标。
        chip (list or tuple): 切片在原始图像中的坐标，格式为 [x1, y1, x2, y2]。
        img_h (int): 原始图像的高度。
        img_w (int): 原始图像的宽度。

    Returns:
        np.ndarray: 转换并裁剪后的边界框数组，形状为 (N, 6) 或相应格式，
                    坐标已在原始图像坐标系内，并且被裁剪到图像边界内。
    """
    # 将边界框列表转换为numpy数组并按y1坐标降序排序
    chip_boxes = np.array(sorted(chip_boxes, key=lambda item: -item[1]))

    # 获取切片的左上角坐标
    xmin, ymin, _, _ = chip

    chip_boxes[:, 2] += xmin
    chip_boxes[:, 4] += xmin
    chip_boxes[:, 3] += ymin
    chip_boxes[:, 5] += ymin

    # 将边界框裁剪到原始图像边界内
    chip_boxes = clip_boxes(chip_boxes, (img_h, img_w))
    return chip_boxes


def nms(dets, thresh):
    """
    应用经典的DPM风格贪婪非极大值抑制(NMS)。

    Args:
        dets (np.ndarray): 检测框数组，形状为 (N, 6)，格式为 [class_id, score, x1, y1, x2, y2]
        thresh (float): IOU阈值，用于判断重叠程度

    Returns:
        np.ndarray: 经过NMS处理后的检测框数组
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 1]
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)

    # nominal indices
    # _i, _j
    # sorted indices
    # i, j
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2
    # w, h
    # inter, ovr

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets


