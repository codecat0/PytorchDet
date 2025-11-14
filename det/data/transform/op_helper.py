#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :op_helper.py
@Author :CodeCat
@Date   :2025/11/6 11:57
"""

import numpy as np
import random
import math
import cv2


def is_poly(segm):
    """
    判断输入是否为多边形（列表类型）。

    Args:
        segm: 输入数据，应为列表或字典类型。若非这两种类型会触发断言错误。

    Returns:
        bool: 如果输入是列表类型返回True，否则返回False。

    Raises:
        AssertionError: 当输入类型不是list或dict时触发。
    """
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


def generate_sample_bbox(sampler):
    """
    根据采样器参数生成一个随机的边界框(bbox)。

    该函数通过随机采样尺度(scale)和宽高比(aspect_ratio)来生成bbox，
    并确保宽高比在合理范围内，最终返回bbox的坐标[xmin, ymin, xmax, ymax]。

    Args:
        sampler: 一个包含采样参数的列表或元组，其中：
            - sampler[2]: 尺度的最小值
            - sampler[3]: 尺度的最大值
            - sampler[4]: 宽高比的最小值
            - sampler[5]: 宽高比的最大值

    Returns:
        list: 包含四个浮点数的列表，表示bbox的坐标[xmin, ymin, xmax, ymax]。
    """
    scale = np.random.uniform(sampler[2], sampler[3])
    aspect_ratio = np.random.uniform(sampler[4], sampler[5])
    aspect_ratio = max(aspect_ratio, (scale ** 2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale ** 2.0))
    bbox_width = scale * (aspect_ratio ** 0.5)
    bbox_height = scale / (aspect_ratio ** 0.5)
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = [xmin, ymin, xmax, ymax]
    return sampled_bbox


def bbox_area(src_bbox):
    """
    计算边界框的面积。

    如果边界框的右边界小于左边界或下边界小于上边界，则返回0（无效边界框）。
    否则返回宽度乘以高度的面积值。

    Args:
        src_bbox: 包含边界框坐标的列表或数组，格式为[x1, y1, x2, y2]，
                 其中(x1,y1)是左上角坐标，(x2,y2)是右下角坐标。

    Returns:
        float: 边界框的面积，无效边界框返回0.0。
    """
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    """
    计算两个边界框之间的Jaccard重叠系数（交并比）。

    首先检查两个边界框是否有重叠区域，若无则直接返回0。
    若有重叠则计算交集面积和并集面积，最后返回交并比。

    Args:
        sample_bbox: 样本边界框，格式为[xmin, ymin, xmax, ymax]
        object_bbox: 目标边界框，格式为[xmin, ymin, xmax, ymax]

    Returns:
        float: 两个边界框的Jaccard重叠系数，范围在[0,1]之间
    """
    if sample_bbox[0] >= object_bbox[2] or \
            sample_bbox[2] <= object_bbox[0] or \
            sample_bbox[1] >= object_bbox[3] or \
            sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
            intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
            sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler,
                              sample_bbox,
                              gt_bboxes,
                              satisfy_all=False):
    """
    检查采样框是否满足给定的约束条件。

    Args:
        sampler: 采样器参数列表，其中第6和第7个元素分别表示
                重叠率的最小阈值和最大阈值（0表示不限制）。
        sample_bbox: 待检查的采样框坐标 [x1, y1, x2, y2]。
        gt_bboxes: 真实框坐标列表，每个元素格式为 [x1, y1, x2, y2]。
        satisfy_all: 是否要求所有真实框都满足约束条件（True）或
                    只要有一个满足即可（False）。

    Returns:
        bool: 如果满足约束条件返回True，否则返回False。
              当sampler[6]和sampler[7]都为0时直接返回True。
    """
    if sampler[6] == 0 and sampler[7] == 0:
        return True
    satisfied = []
    for i in range(len(gt_bboxes)):
        object_bbox = [
            gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]
        ]
        overlap = jaccard_overlap(sample_bbox, object_bbox)
        if sampler[6] != 0 and \
                overlap < sampler[6]:
            satisfied.append(False)
            continue
        if sampler[7] != 0 and \
                overlap > sampler[7]:
            satisfied.append(False)
            continue
        satisfied.append(True)
        if not satisfy_all:
            return True

    if satisfy_all:
        return np.all(satisfied)
    else:
        return False


def clip_bbox(src_bbox):
    """
    裁剪边界框坐标到[0.0, 1.0]范围内。

    Args:
        src_bbox (list): 包含4个元素的列表，表示边界框的坐标[x1, y1, x2, y2]。

    Returns:
        list: 裁剪后的边界框坐标列表，所有值都在[0.0, 1.0]范围内。
    """
    src_bbox[0] = max(min(src_bbox[0], 1.0), 0.0)
    src_bbox[1] = max(min(src_bbox[1], 1.0), 0.0)
    src_bbox[2] = max(min(src_bbox[2], 1.0), 0.0)
    src_bbox[3] = max(min(src_bbox[3], 1.0), 0.0)
    return src_bbox


def meet_emit_constraint(src_bbox, sample_bbox):
    """
    判断源边界框的中心点是否位于采样边界框内。

    Args:
        src_bbox: 源边界框坐标，格式为[x_min, y_min, x_max, y_max]。
        sample_bbox: 采样边界框坐标，格式为[x_min, y_min, x_max, y_max]。

    Returns:
        bool: 如果源边界框的中心点位于采样边界框内则返回True，否则返回False。
    """
    center_x = (src_bbox[2] + src_bbox[0]) / 2
    center_y = (src_bbox[3] + src_bbox[1]) / 2
    if center_x >= sample_bbox[0] and \
            center_x <= sample_bbox[2] and \
            center_y >= sample_bbox[1] and \
            center_y <= sample_bbox[3]:
        return True
    return False


def is_overlap(object_bbox, sample_bbox):
    """
    判断两个边界框是否存在重叠区域。

    边界框格式为[x_min, y_min, x_max, y_max]，其中：
    - x_min/y_min 表示边界框左下角坐标
    - x_max/y_max 表示边界框右上角坐标

    Args:
        object_bbox: 目标边界框坐标列表 [x_min, y_min, x_max, y_max]
        sample_bbox: 样本边界框坐标列表 [x_min, y_min, x_max, y_max]

    Returns:
        bool: 如果两个边界框存在重叠区域返回True，否则返回False
    """
    if object_bbox[0] >= sample_bbox[2] or \
            object_bbox[2] <= sample_bbox[0] or \
            object_bbox[1] >= sample_bbox[3] or \
            object_bbox[3] <= sample_bbox[1]:
        return False
    else:
        return True


def filter_and_process(sample_bbox, bboxes, labels, scores=None,
                       keypoints=None):
    """
    根据样本边界框过滤并处理目标边界框、标签、分数和关键点。

    该函数会过滤掉不满足发射约束或不与样本边界框重叠的目标，
    并将符合条件的边界框坐标归一化到样本边界框的相对坐标系中。
    同时对关键点坐标进行相应的归一化处理。

    Args:
        sample_bbox: 样本边界框，格式为[x_min, y_min, x_max, y_max]
        bboxes: 待处理的目标边界框列表，每个元素格式为[x_min, y_min, x_max, y_max]
        labels: 目标标签列表，每个元素是一个列表
        scores: 可选参数，目标分数列表，每个元素是一个列表
        keypoints: 可选参数，包含关键点坐标和忽略标志的元组
                   (每个目标的关键点坐标列表, 每个目标的忽略标志列表)

    Returns:
        根据输入参数返回处理后的结果：
        - 如果keypoints为None，返回(bboxes, labels, scores)
        - 否则返回(bboxes, labels, scores, (processed_keypoints, ignore_flags))
        所有返回的边界框和坐标都已归一化到[0,1]范围
    """
    new_bboxes = []
    new_labels = []
    new_scores = []
    new_keypoints = []
    new_kp_ignore = []
    for i in range(len(bboxes)):
        new_bbox = [0, 0, 0, 0]
        obj_bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if not meet_emit_constraint(obj_bbox, sample_bbox):
            continue
        if not is_overlap(obj_bbox, sample_bbox):
            continue
        sample_width = sample_bbox[2] - sample_bbox[0]
        sample_height = sample_bbox[3] - sample_bbox[1]
        new_bbox[0] = (obj_bbox[0] - sample_bbox[0]) / sample_width
        new_bbox[1] = (obj_bbox[1] - sample_bbox[1]) / sample_height
        new_bbox[2] = (obj_bbox[2] - sample_bbox[0]) / sample_width
        new_bbox[3] = (obj_bbox[3] - sample_bbox[1]) / sample_height
        new_bbox = clip_bbox(new_bbox)
        if bbox_area(new_bbox) > 0:
            new_bboxes.append(new_bbox)
            new_labels.append([labels[i][0]])
            if scores is not None:
                new_scores.append([scores[i][0]])
            if keypoints is not None:
                sample_keypoint = keypoints[0][i]
                for j in range(len(sample_keypoint)):
                    kp_len = sample_height if j % 2 else sample_width
                    sample_coord = sample_bbox[1] if j % 2 else sample_bbox[0]
                    sample_keypoint[j] = (
                                                 sample_keypoint[j] - sample_coord) / kp_len
                    sample_keypoint[j] = max(min(sample_keypoint[j], 1.0), 0.0)
                new_keypoints.append(sample_keypoint)
                new_kp_ignore.append(keypoints[1][i])

    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    if keypoints is not None:
        keypoints = np.array(new_keypoints)
        new_kp_ignore = np.array(new_kp_ignore)
        return bboxes, labels, scores, (keypoints, new_kp_ignore)
    return bboxes, labels, scores


def data_anchor_sampling(bbox_labels, image_width, image_height, scale_array,
                         resize_width):
    """
    根据锚框数据采样方法生成采样边界框。

    该函数从给定的边界框标签中随机选择一个，根据其面积大小在预设的比例数组中选择合适的缩放比例，
    然后生成一个随机大小的采样边界框，确保采样框不会超出图像范围。

    Args:
        bbox_labels (list): 归一化的边界框标签列表，每个元素为[xmin, ymin, xmax, ymax]格式。
        image_width (int): 原始图像的宽度。
        image_height (int): 原始图像的高度。
        scale_array (list): 预设的比例数组，用于确定采样框的大小范围。
        resize_width (int): 调整大小后的目标宽度。

    Returns:
        list: 采样后的归一化边界框坐标[xmin, ymin, xmax, ymax]，若没有边界框则返回0。
    """
    num_gt = len(bbox_labels)
    # np.random.randint range: [low, high)
    rand_idx = np.random.randint(0, num_gt) if num_gt != 0 else 0

    if num_gt != 0:
        norm_xmin = bbox_labels[rand_idx][0]
        norm_ymin = bbox_labels[rand_idx][1]
        norm_xmax = bbox_labels[rand_idx][2]
        norm_ymax = bbox_labels[rand_idx][3]

        xmin = norm_xmin * image_width
        ymin = norm_ymin * image_height
        wid = image_width * (norm_xmax - norm_xmin)
        hei = image_height * (norm_ymax - norm_ymin)
        range_size = 0

        area = wid * hei
        for scale_ind in range(0, len(scale_array) - 1):
            if area > scale_array[scale_ind] ** 2 and area < \
                    scale_array[scale_ind + 1] ** 2:
                range_size = scale_ind + 1
                break

        if area > scale_array[len(scale_array) - 2] ** 2:
            range_size = len(scale_array) - 2

        scale_choose = 0.0
        if range_size == 0:
            rand_idx_size = 0
        else:
            # np.random.randint range: [low, high)
            rng_rand_size = np.random.randint(0, range_size + 1)
            rand_idx_size = rng_rand_size % (range_size + 1)

        if rand_idx_size == range_size:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = min(2.0 * scale_array[rand_idx_size],
                                 2 * math.sqrt(wid * hei))
            scale_choose = random.uniform(min_resize_val, max_resize_val)
        else:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = 2.0 * scale_array[rand_idx_size]
            scale_choose = random.uniform(min_resize_val, max_resize_val)

        sample_bbox_size = wid * resize_width / scale_choose

        w_off_orig = 0.0
        h_off_orig = 0.0
        if sample_bbox_size < max(image_height, image_width):
            if wid <= sample_bbox_size:
                w_off_orig = np.random.uniform(xmin + wid - sample_bbox_size,
                                               xmin)
            else:
                w_off_orig = np.random.uniform(xmin,
                                               xmin + wid - sample_bbox_size)

            if hei <= sample_bbox_size:
                h_off_orig = np.random.uniform(ymin + hei - sample_bbox_size,
                                               ymin)
            else:
                h_off_orig = np.random.uniform(ymin,
                                               ymin + hei - sample_bbox_size)

        else:
            w_off_orig = np.random.uniform(image_width - sample_bbox_size, 0.0)
            h_off_orig = np.random.uniform(image_height - sample_bbox_size, 0.0)

        w_off_orig = math.floor(w_off_orig)
        h_off_orig = math.floor(h_off_orig)

        # Figure out top left coordinates.
        w_off = float(w_off_orig / image_width)
        h_off = float(h_off_orig / image_height)

        sampled_bbox = [
            w_off, h_off, w_off + float(sample_bbox_size / image_width),
                          h_off + float(sample_bbox_size / image_height)
        ]
        return sampled_bbox
    else:
        return 0


def intersect_bbox(bbox1, bbox2):
    """
    计算两个边界框的交集区域。

    如果两个边界框没有交集，则返回全零的边界框[0.0, 0.0, 0.0, 0.0]。
    否则返回两个边界框的交集区域，格式为[x_min, y_min, x_max, y_max]。

    Args:
        bbox1: 第一个边界框，格式为[x_min, y_min, x_max, y_max]。
        bbox2: 第二个边界框，格式为[x_min, y_min, x_max, y_max]。

    Returns:
        list: 两个边界框的交集区域，格式为[x_min, y_min, x_max, y_max]。
              如果没有交集则返回[0.0, 0.0, 0.0, 0.0]。
    """
    if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or \
        bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1]:
        intersection_box = [0.0, 0.0, 0.0, 0.0]
    else:
        intersection_box = [
            max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
            min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        ]
    return intersection_box


def bbox_coverage(bbox1, bbox2):
    """
    计算第二个边界框与第一个边界框的重叠覆盖率（交集面积/第一个边界框面积）

    Args:
        bbox1: 第一个边界框，格式为[x1, y1, x2, y2]
        bbox2: 第二个边界框，格式为[x1, y1, x2, y2]

    Returns:
        float: 重叠覆盖率，范围在[0,1]之间。如果没有重叠则返回0
    """
    inter_box = intersect_bbox(bbox1, bbox2)
    intersect_size = bbox_area(inter_box)

    if intersect_size > 0:
        bbox1_size = bbox_area(bbox1)
        return intersect_size / bbox1_size
    else:
        return 0.


def satisfy_sample_constraint_coverage(sampler, sample_bbox, gt_bboxes):
    """
    检查采样边界框是否满足给定的约束覆盖条件。

    根据采样器参数判断是否需要检查Jaccard重叠度或目标覆盖度，
    然后验证采样边界框与所有真实边界框的对应指标是否满足约束条件。

    Args:
        sampler: 采样器参数列表，其中：
            - sampler[6]: 最小Jaccard重叠度阈值（0表示不检查）
            - sampler[7]: 最大Jaccard重叠度阈值（0表示不检查）
            - sampler[8]: 最小目标覆盖度阈值（0表示不检查）
            - sampler[9]: 最大目标覆盖度阈值（0表示不检查）
        sample_bbox: 待验证的采样边界框 [x1, y1, x2, y2]
        gt_bboxes: 真实边界框列表，每个元素为 [x1, y1, x2, y2]

    Returns:
        bool: 如果采样边界框满足所有非零约束条件则返回True，否则返回False
    """
    if sampler[6] == 0 and sampler[7] == 0:
        has_jaccard_overlap = False
    else:
        has_jaccard_overlap = True
    if sampler[8] == 0 and sampler[9] == 0:
        has_object_coverage = False
    else:
        has_object_coverage = True

    if not has_jaccard_overlap and not has_object_coverage:
        return True
    found = False
    for i in range(len(gt_bboxes)):
        object_bbox = [
            gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]
        ]
        if has_jaccard_overlap:
            overlap = jaccard_overlap(sample_bbox, object_bbox)
            if sampler[6] != 0 and \
                    overlap < sampler[6]:
                continue
            if sampler[7] != 0 and \
                    overlap > sampler[7]:
                continue
            found = True
        if has_object_coverage:
            object_coverage = bbox_coverage(object_bbox, sample_bbox)
            if sampler[8] != 0 and \
                    object_coverage < sampler[8]:
                continue
            if sampler[9] != 0 and \
                    object_coverage > sampler[9]:
                continue
            found = True
        if found:
            return True
    return found


def crop_image_sampling(img, sample_bbox, image_width, image_height,
                        target_size):
    """
    根据边界框对图像进行裁剪和采样。

    该函数首先根据归一化的边界框坐标计算实际像素坐标，然后处理边界越界情况，
    最后将裁剪后的区域调整为目标大小。

    Args:
        img: 输入图像(numpy数组)，格式为HWC(高度,宽度,通道)。
        sample_bbox: 归一化的边界框坐标，格式为[xmin,ymin,xmax,ymax]，值在[0,1]范围内。
        image_width: 输入图像的宽度(像素)。
        image_height: 输入图像的高度(像素)。
        target_size: 目标输出图像的尺寸(正方形边长)。

    Returns:
        裁剪并调整大小后的图像(numpy数组)，尺寸为(target_size,target_size,3)。
    """
    # no clipping here
    xmin = int(sample_bbox[0] * image_width)
    xmax = int(sample_bbox[2] * image_width)
    ymin = int(sample_bbox[1] * image_height)
    ymax = int(sample_bbox[3] * image_height)

    w_off = xmin
    h_off = ymin
    width = xmax - xmin
    height = ymax - ymin
    cross_xmin = max(0.0, float(w_off))
    cross_ymin = max(0.0, float(h_off))
    cross_xmax = min(float(w_off + width - 1.0), float(image_width))
    cross_ymax = min(float(h_off + height - 1.0), float(image_height))
    cross_width = cross_xmax - cross_xmin
    cross_height = cross_ymax - cross_ymin

    roi_xmin = 0 if w_off >= 0 else abs(w_off)
    roi_ymin = 0 if h_off >= 0 else abs(h_off)
    roi_width = cross_width
    roi_height = cross_height

    roi_y1 = int(roi_ymin)
    roi_y2 = int(roi_ymin + roi_height)
    roi_x1 = int(roi_xmin)
    roi_x2 = int(roi_xmin + roi_width)

    cross_y1 = int(cross_ymin)
    cross_y2 = int(cross_ymin + cross_height)
    cross_x1 = int(cross_xmin)
    cross_x2 = int(cross_xmin + cross_width)

    sample_img = np.zeros((height, width, 3))
    sample_img[roi_y1: roi_y2, roi_x1: roi_x2] = \
        img[cross_y1: cross_y2, cross_x1: cross_x2]

    sample_img = cv2.resize(
        sample_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return sample_img


def generate_sample_bbox_square(sampler, image_width, image_height):
    """
    生成一个随机样本边界框（正方形或近似正方形）。

    该函数根据给定的采样器参数（缩放比例和宽高比范围）生成一个随机边界框，
    并确保边界框适应图像尺寸，同时保持坐标在[0,1]范围内。

    Args:
        sampler: 采样器参数列表，包含：
            - sampler[2]: 最小缩放比例
            - sampler[3]: 最大缩放比例
            - sampler[4]: 最小宽高比
            - sampler[5]: 最大宽高比
        image_width: 图像宽度（像素）
        image_height: 图像高度（像素）

    Returns:
        list: 包含四个元素的列表[xmin, ymin, xmax, ymax]，表示边界框的归一化坐标
    """
    scale = np.random.uniform(sampler[2], sampler[3])
    aspect_ratio = np.random.uniform(sampler[4], sampler[5])
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))
    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    if image_height < image_width:
        bbox_width = bbox_height * image_height / image_width
    else:
        bbox_height = bbox_width * image_width / image_height
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = [xmin, ymin, xmax, ymax]
    return sampled_bbox


def bbox_area_sampling(bboxes, labels, scores, target_size, min_size):
    """
    根据边界框面积进行采样，过滤掉面积过小的边界框

    Args:
        bboxes: 边界框列表，每个边界框格式为[x1, y1, x2, y2]
        labels: 边界框对应的标签列表
        scores: 边界框对应的置信度分数列表（可选）
        target_size: 目标尺寸，用于计算边界框的实际宽高
        min_size: 最小允许的边界框边长（以target_size为基准）

    Returns:
        tuple: 包含三个元素的元组，分别是过滤后的边界框、标签和分数列表
    """
    new_bboxes = []
    new_labels = []
    new_scores = []
    for i, bbox in enumerate(bboxes):
        w = float((bbox[2] - bbox[0]) * target_size)
        h = float((bbox[3] - bbox[1]) * target_size)
        if w * h < float(min_size * min_size):
            continue
        else:
            new_bboxes.append(bbox)
            new_labels.append(labels[i])
            if scores is not None and scores.size != 0:
                new_scores.append(scores[i])
    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    return bboxes, labels, scores


def get_border(border, size):
    """
    根据给定的边界值和大小，计算并返回调整后的边界值。

    该函数通过不断将i乘以2，直到满足size - border//i > border//i的条件，
    然后返回border//i作为结果。

    Args:
        border: 初始边界值。
        size: 目标大小值。

    Returns:
        调整后的边界值，即border//i。
    """
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y *
                                                            sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(bbox_size, min_overlap):
    height, width = bbox_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    radius1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    radius2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    radius3 = (b3 + sq3) / 2
    return min(radius1, radius2, radius3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    draw_umich_gaussian, refer to https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L126
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D(
        (diameter, diameter), sigma_x=diameter / 6, sigma_y=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:
                               radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
