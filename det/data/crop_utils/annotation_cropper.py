#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :annotation_cropper.py
@Author :CodeCat
@Date   :2025/11/21 15:57
"""
import copy
import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from collections import defaultdict

from det.data.crop_utils.chip_box_utils import nms, transform_chip_box, transform_chip_boxes2image_boxes, \
    find_chips_to_cover_overlaped_boxes, intersection_over_box


class AnnoCropper(object):
    def __init__(self,
                 image_target_sizes: List[int],
                 valid_box_ratio_ranges: List[List[float]],
                 chip_target_size: int,
                 chip_target_stride: int,
                 use_neg_chip: bool = False,
                 max_neg_num_per_im: int = 8,
                 max_per_img: int = -1,
                 nms_thresh: float = 0.5):
        """
        根据chip_target_size和chip_target_stride生成图像切片（chips）的工具类。
        这两个参数类似于CNN中的kernel_size和stride。

        每张图像都有其原始尺寸。调整大小后，得到目标尺寸。
        调整比例 = 目标尺寸 / 原始尺寸。
        图像的切片也是如此。
        box_ratio = box_raw_size / image_raw_size = box_target_size / image_target_size
        上述提到的'size'是指图像、边界框或切片的长边尺寸。

        Args:
            image_target_sizes: 不同尺度的目标尺寸列表，例如 [2000, 1000]
            valid_box_ratio_ranges: 每个尺度下有效边界框比例范围列表，例如 [[-1, 0.1],[0.08, -1]]
            chip_target_size: 切片的目标尺寸
            chip_target_stride: 切片的步长
            use_neg_chip: 是否使用负样本切片
            max_neg_num_per_im: 每张图像最大负样本切片数量
            max_per_img: 每张图像最大检测数量，-1表示无限制
            nms_thresh: NMS阈值
        """
        self.target_sizes = image_target_sizes
        self.valid_box_ratio_ranges = valid_box_ratio_ranges
        assert len(self.target_sizes) == len(
            self.valid_box_ratio_ranges), "The lengths of target_sizes and valid_box_ratio_ranges must be the same"
        self.scale_num = len(self.target_sizes)
        self.chip_target_size = chip_target_size  # 目标尺寸
        self.chip_target_stride = chip_target_stride  # 目标步长
        self.use_neg_chip = use_neg_chip
        self.max_neg_num_per_im = max_neg_num_per_im
        self.max_per_img = max_per_img
        self.nms_thresh = nms_thresh

    def crop_anno_records(self, records: List[dict]):
        """
        主要逻辑:
        # 对于每个记录(图像):
        #   对于每个尺度:
        #     1 根据切片大小和步长为每个尺度生成切片
        #     2 获取正样本切片
        #     - 验证边界框: 当前尺度; h,w >= 1
        #     - 通过每个尺度中有效的gt边界框贪婪地找到正样本切片
        #     - 对于每个有效的gt边界框，在每个尺度中找到其对应的正样本切片
        #     3 获取负样本切片
        #     - 如果给定提议，找到不在正样本切片中的负样本边界框
        #     - 如果在上一步中得到负样本边界框，我们找到负样本切片并将负样本边界框分配给负样本切片，如2.
        # 4 如果每张图像的负样本切片太多，则进行采样
        #   将此图像-尺度注释转换为切片(正样本切片&负样本切片)注释

        Args:
            records: 标准的coco_record，但带有额外的键`proposals`(Px4)，这些是由stage1
                     模型预测的，可能在其中包含负样本边界框。

        Returns:
            new_records: 字典列表，格式如下
            {
                'im_file': 'fake_image1.jpg',
                'im_id': np.array([1]),  # 以im_id作为新的_global_chip_id
                'h': h,  # 切片高度
                'w': w,  # 切片宽度
                'is_crowd': is_crowd,  # Nx1 -> Mx1
                'gt_class': gt_class,  # Nx1 -> Mx1
                'gt_bbox': gt_bbox,  # Nx4 -> Mx4, 4表示 [x1,y1,x2,y2]
                'gt_poly': gt_poly,  # [None]xN -> [None]xM
                'chip': [x1, y1, x2, y2]  # 新增
            }

        注意:
        ------------------------------>x
        |
        |    (x1,y1)------
        |       |        |
        |       |        |
        |       |        |
        |       |        |
        |       |        |
        |       ----------
        |                 (x2,y2)
        |
        ↓
        y

        如果我们使用 [x1, y1, x2, y2] 来表示边界框或切片，
        (x1,y1) 是在边界框内的左上角点，
        但 (x2,y2) 是不在边界框内的右下角点。
        所以 x1 在 [0, w-1] 范围内，x2 在 [1, w] 范围内，y1 在 [0, h-1] 范围内，y2 在 [1,h] 范围内。
        你可以使用 x2-x1 来获取宽度，也可以使用 image[y1:y2, x1:x2] 来获取边界框区域。
        """

        self.chip_records = []
        self._global_chip_id = 1
        for r in records:
            self._cur_im_pos_chips = [
            ]  # 元素: (chip, boxes_idx), chip是 [x1, y1, x2, y2], boxes_ids是List[int]
            self._cur_im_neg_chips = []  # 元素: (chip, neg_box_num)
            for scale_i in range(self.scale_num):
                self._get_current_scale_parameters(scale_i, r)

                # Cx4
                chips = self._create_chips(r['h'], r['w'], self._cur_scale)

                # # dict: chipid->[box_id, ...]
                pos_chip2boxes_idx = self._get_valid_boxes_and_pos_chips(
                    r['gt_bbox'], chips)

                # dict: chipid->neg_box_num
                neg_chip2box_num = self._get_neg_boxes_and_chips(
                    chips,
                    list(pos_chip2boxes_idx.keys()), r.get('proposals', None))

                self._add_to_cur_im_chips(chips, pos_chip2boxes_idx,
                                          neg_chip2box_num)

            cur_image_records = self._trans_all_chips2annotations(r)
            self.chip_records.extend(cur_image_records)
        return self.chip_records

    def _add_to_cur_im_chips(self, chips, pos_chip2boxes_idx, neg_chip2box_num):
        """
        将正样本和负样本切片添加到当前图像的列表中

        Args:
            chips: 切片列表
            pos_chip2boxes_idx: 正样本切片到边界框索引的映射
            neg_chip2box_num: 负样本切片到边界框数量的映射
        """
        for pos_chipid, boxes_idx in pos_chip2boxes_idx.items():
            chip = np.array(chips[pos_chipid])  # 复制切片切片
            self._cur_im_pos_chips.append((chip, boxes_idx))

        if neg_chip2box_num is None:
            return

        for neg_chipid, neg_box_num in neg_chip2box_num.items():
            chip = np.array(chips[neg_chipid])
            self._cur_im_neg_chips.append((chip, neg_box_num))

    def _trans_all_chips2annotations(self, r):
        """
        将所有切片转换为注释格式

        Args:
            r: 原始记录

        Returns:
            切片注释列表
        """
        gt_bbox = r['gt_bbox']
        im_file = r['im_file']
        is_crowd = r['is_crowd']
        gt_class = r['gt_class']
        # gt_poly = r['gt_poly']   # [None]xN
        # 剩余键: im_id, h, w
        chip_records = self._trans_pos_chips2annotations(im_file, gt_bbox,
                                                         is_crowd, gt_class)

        if not self.use_neg_chip:
            return chip_records

        sampled_neg_chips = self._sample_neg_chips()
        neg_chip_records = self._trans_neg_chips2annotations(im_file,
                                                             sampled_neg_chips)
        chip_records.extend(neg_chip_records)
        return chip_records

    def _trans_pos_chips2annotations(self, im_file, gt_bbox, is_crowd,
                                     gt_class):
        """
        将正样本切片转换为注释格式

        Args:
            im_file: 图像文件路径
            gt_bbox: 地面真值边界框
            is_crowd: 是否为人群标记
            gt_class: 地面真值类别

        Returns:
            正样本切片注释列表
        """
        chip_records = []
        for chip, boxes_idx in self._cur_im_pos_chips:
            chip_bbox, final_boxes_idx = transform_chip_box(gt_bbox, boxes_idx,
                                                            chip)
            x1, y1, x2, y2 = chip
            chip_h = y2 - y1
            chip_w = x2 - x1
            rec = {
                'im_file': im_file,
                'im_id': np.array([self._global_chip_id]),
                'h': chip_h,
                'w': chip_w,
                'gt_bbox': chip_bbox,
                'is_crowd': is_crowd[final_boxes_idx].copy(),
                'gt_class': gt_class[final_boxes_idx].copy(),
                # 'gt_poly': [None] * len(final_boxes_idx),
                'chip': chip
            }
            self._global_chip_id += 1
            chip_records.append(rec)
        return chip_records

    def _sample_neg_chips(self):
        """
        采样负样本切片

        Returns:
            采样后的负样本切片列表
        """
        pos_num = len(self._cur_im_pos_chips)
        neg_num = len(self._cur_im_neg_chips)
        sample_num = min(pos_num + 2, self.max_neg_num_per_im)
        assert sample_num >= 1
        if neg_num <= sample_num:
            return self._cur_im_neg_chips

        candidate_num = int(sample_num * 1.5)
        candidate_neg_chips = sorted(
            self._cur_im_neg_chips, key=lambda x: -x[1])[:candidate_num]
        random.shuffle(candidate_neg_chips)
        sampled_neg_chips = candidate_neg_chips[:sample_num]
        return sampled_neg_chips

    def _trans_neg_chips2annotations(self,
                                     im_file: str,
                                     sampled_neg_chips: List[Tuple]):
        """
        将负样本切片转换为注释格式

        Args:
            im_file: 图像文件路径
            sampled_neg_chips: 采样的负样本切片列表

        Returns:
            负样本切片注释列表
        """
        chip_records = []
        for chip, neg_box_num in sampled_neg_chips:
            x1, y1, x2, y2 = chip
            chip_h = y2 - y1
            chip_w = x2 - x1
            rec = {
                'im_file': im_file,
                'im_id': np.array([self._global_chip_id]),
                'h': chip_h,
                'w': chip_w,
                'gt_bbox': np.zeros(
                    (0, 4), dtype=np.float32),
                'is_crowd': np.zeros(
                    (0, 1), dtype=np.int32),
                'gt_class': np.zeros(
                    (0, 1), dtype=np.int32),
                # 'gt_poly': [],
                'chip': chip
            }
            self._global_chip_id += 1
            chip_records.append(rec)
        return chip_records

    def _get_current_scale_parameters(self, scale_i, r):
        """
        获取当前尺度的参数

        Args:
            scale_i: 尺度索引
            r: 记录
        """
        im_size = max(r['h'], r['w'])
        im_target_size = self.target_sizes[scale_i]
        self._cur_im_size, self._cur_im_target_size = im_size, im_target_size
        self._cur_scale = self._get_current_scale(im_target_size, im_size)
        self._cur_valid_ratio_range = self.valid_box_ratio_ranges[scale_i]

    @staticmethod
    def _get_current_scale(im_target_size, im_size):
        """
        获取当前尺度

        Args:
            im_target_size: 目标图像尺寸
            im_size: 原始图像尺寸

        Returns:
            当前尺度
        """
        return im_target_size / im_size

    def _create_chips(self, h: int, w: int, scale: float):
        """
        根据chip_target_size和chip_target_stride生成切片。
        这两个参数类似于CNN中的kernel_size和stride。

        Args:
            h: 图像高度
            w: 图像宽度
            scale: 缩放比例

        Returns:
            chips: 切片列表，形状为Cx4，xy在原始尺寸维度
        """
        chip_size = self.chip_target_size  # 为简化省略target
        stride = self.chip_target_stride
        width = int(scale * w)
        height = int(scale * h)
        min_chip_location_diff = 20  # 在目标尺寸中

        assert chip_size >= stride
        chip_overlap = chip_size - stride
        if (width - chip_overlap
        ) % stride > min_chip_location_diff:  # 不能被stride整除的部分比较大，则保留
            w_steps = max(1, int(math.ceil((width - chip_overlap) / stride)))
        else:  # 不能被stride整除的部分比较小，则丢弃
            w_steps = max(1, int(math.floor((width - chip_overlap) / stride)))
        if (height - chip_overlap) % stride > min_chip_location_diff:
            h_steps = max(1, int(math.ceil((height - chip_overlap) / stride)))
        else:
            h_steps = max(1, int(math.floor((height - chip_overlap) / stride)))

        chips = list()
        for j in range(h_steps):
            for i in range(w_steps):
                x1 = i * stride
                y1 = j * stride
                x2 = min(x1 + chip_size, width)
                y2 = min(y1 + chip_size, height)
                chips.append([x1, y1, x2, y2])

        # 检查切片尺寸
        for item in chips:
            if item[2] - item[0] > chip_size * 1.1 or item[3] - item[1] > chip_size * 1.1:
                raise ValueError(item)
        chips = np.array(chips, dtype=np.float32)

        raw_size_chips = chips / scale
        return raw_size_chips

    def _get_valid_boxes_and_pos_chips(self, gt_bbox, chips):
        """
        获取有效边界框和正样本切片

        Args:
            gt_bbox: 地面真值边界框
            chips: 切片列表

        Returns:
            pos_chip2boxes_idx: 正样本切片到边界框索引的映射
        """
        valid_ratio_range = self._cur_valid_ratio_range
        im_size = self._cur_im_size
        scale = self._cur_scale
        #   Nx4            N
        valid_boxes, valid_boxes_idx = self._validate_boxes(
            valid_ratio_range, im_size, gt_bbox, scale)
        # dict: chipid->[box_id, ...]
        pos_chip2boxes_idx = self._find_pos_chips(chips, valid_boxes,
                                                  valid_boxes_idx)
        return pos_chip2boxes_idx

    @staticmethod
    def _validate_boxes(valid_ratio_range: List[float],
                        im_size: int,
                        gt_boxes: 'np.array of Nx4',
                        scale: float):
        """
        验证边界框是否有效

        Args:
            valid_ratio_range: 有效比例范围
            im_size: 图像尺寸
            gt_boxes: 地面真值边界框
            scale: 缩放比例

        Returns:
            valid_boxes: 有效边界框
            valid_boxes_idx: 有效边界框索引
        """
        ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
        hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
        maxs = np.maximum(ws, hs)
        box_ratio = maxs / im_size
        mins = np.minimum(ws, hs)
        target_mins = mins * scale

        low = valid_ratio_range[0] if valid_ratio_range[0] > 0 else 0
        high = valid_ratio_range[1] if valid_ratio_range[1] > 0 else np.finfo(
            np.float32).max

        valid_boxes_idx = np.nonzero((low <= box_ratio) & (box_ratio < high) & (
                target_mins >= 2))[0]
        valid_boxes = gt_boxes[valid_boxes_idx]
        return valid_boxes, valid_boxes_idx

    def _find_pos_chips(self,
                        chips: 'Cx4',
                        valid_boxes: 'Bx4',
                        valid_boxes_idx: 'B'):
        """
        找到正样本切片

        Args:
            chips: 切片列表
            valid_boxes: 有效边界框
            valid_boxes_idx: 有效边界框索引

        Returns:
            pos_chip2boxes_idx: 正样本切片到边界框索引的映射
        """
        iob = intersection_over_box(chips, valid_boxes)  # 重叠度, CxB

        iob_threshold_to_find_chips = 1.
        pos_chip_ids, _ = self._find_chips_to_cover_overlaped_boxes(
            iob, iob_threshold_to_find_chips)
        pos_chip_ids = set(pos_chip_ids)

        iob_threshold_to_assign_box = 0.5
        pos_chip2boxes_idx = self._assign_boxes_to_pos_chips(
            iob, iob_threshold_to_assign_box, pos_chip_ids, valid_boxes_idx)
        return pos_chip2boxes_idx

    @staticmethod
    def _find_chips_to_cover_overlaped_boxes(iob, overlap_threshold):
        """
        找到覆盖重叠边界框的切片

        Args:
            iob: 交集与边界框面积比矩阵
            overlap_threshold: 重叠阈值

        Returns:
            找到的切片ID和切片重叠框数量
        """
        return find_chips_to_cover_overlaped_boxes(iob, overlap_threshold)

    @staticmethod
    def _assign_boxes_to_pos_chips(iob, overlap_threshold, pos_chip_ids,
                                   valid_boxes_idx):
        """
        将边界框分配给正样本切片

        Args:
            iob: 交集与边界框面积比矩阵
            overlap_threshold: 重叠阈值
            pos_chip_ids: 正样本切片ID
            valid_boxes_idx: 有效边界框索引

        Returns:
            pos_chip2boxes_idx: 正样本切片到边界框索引的映射
        """
        chip_ids, box_ids = np.nonzero(iob >= overlap_threshold)
        pos_chip2boxes_idx = defaultdict(list)
        for chip_id, box_id in zip(chip_ids, box_ids):
            if chip_id not in pos_chip_ids:
                continue
            raw_gt_box_idx = valid_boxes_idx[box_id]
            pos_chip2boxes_idx[chip_id].append(raw_gt_box_idx)
        return pos_chip2boxes_idx

    def _get_neg_boxes_and_chips(self,
                                 chips: 'Cx4',
                                 pos_chip_ids: 'D',
                                 proposals: 'Px4'):
        """
        获取负样本边界框和切片

        Args:
            chips: 切片列表
            pos_chip_ids: 正样本切片ID
            proposals: 提议边界框

        Returns:
            neg_chip2box_num: 负样本切片到边界框数量的映射，None或dict: chipid->neg_box_num
        """
        if not self.use_neg_chip:
            return None

        # 训练提议可能为None
        if proposals is None or len(proposals) < 1:
            return None

        valid_ratio_range = self._cur_valid_ratio_range
        im_size = self._cur_im_size
        scale = self._cur_scale

        valid_props, _ = self._validate_boxes(valid_ratio_range, im_size,
                                              proposals, scale)
        neg_boxes = self._find_neg_boxes(chips, pos_chip_ids, valid_props)
        neg_chip2box_num = self._find_neg_chips(chips, pos_chip_ids, neg_boxes)
        return neg_chip2box_num

    @staticmethod
    def _find_neg_boxes(chips: 'Cx4',
                        pos_chip_ids: 'D',
                        valid_props: 'Px4'):
        """
        找到负样本边界框

        Args:
            chips: 切片列表
            pos_chip_ids: 正样本切片ID
            valid_props: 有效提议

        Returns:
            neg_boxes: 负样本边界框
        """
        if len(pos_chip_ids) == 0:
            return valid_props

        pos_chips = chips[pos_chip_ids]
        iob = intersection_over_box(pos_chips, valid_props)
        overlap_per_prop = np.max(iob, axis=0)
        non_overlap_props_idx = overlap_per_prop < 0.5
        neg_boxes = valid_props[non_overlap_props_idx]
        return neg_boxes

    def _find_neg_chips(self, chips: 'Cx4', pos_chip_ids: 'D',
                        neg_boxes: 'Nx4'):
        """
        找到负样本切片

        Args:
            chips: 切片列表
            pos_chip_ids: 正样本切片ID
            neg_boxes: 负样本边界框

        Returns:
            neg_chipid2box_num: 负样本切片ID到边界框数量的映射
        """
        neg_chip_ids = np.setdiff1d(np.arange(len(chips)), pos_chip_ids)
        neg_chips = chips[neg_chip_ids]

        iob = intersection_over_box(neg_chips, neg_boxes)
        iob_threshold_to_find_chips = 0.7
        chosen_neg_chip_ids, chip_id2overlap_box_num = \
            self._find_chips_to_cover_overlaped_boxes(iob, iob_threshold_to_find_chips)

        neg_chipid2box_num = {}
        for cid in chosen_neg_chip_ids:
            box_num = chip_id2overlap_box_num[cid]
            raw_chip_id = neg_chip_ids[cid]
            neg_chipid2box_num[raw_chip_id] = box_num
        return neg_chipid2box_num

    def crop_infer_anno_records(self, records: List[dict]):
        """
        将图像记录转换为切片记录

        Args:
            records: 记录列表

        Returns:
            new_records: 字典列表，格式如下
            {
                'im_file': 'fake_image1.jpg',
                'im_id': np.array([1]),  # 以im_id作为新的_global_chip_id
                'h': h,  # 切片高度
                'w': w,  # 切片宽度
                'chip': [x1, y1, x2, y2]  # 新增
                'ori_im_h': ori_im_h  # 新增, 原始图像高度
                'ori_im_w': ori_im_w  # 新增, 原始图像宽度
                'scale_i': 0  # 新增,
            }
        """
        self.chip_records = []
        self._global_chip_id = 1  # im_id从1开始
        self._global_chip_id2img_id = {}

        for r in records:
            for scale_i in range(self.scale_num):
                self._get_current_scale_parameters(scale_i, r)
                # Cx4
                chips = self._create_chips(r['h'], r['w'], self._cur_scale)
                cur_img_chip_record = self._get_chips_records(r, chips, scale_i)
                self.chip_records.extend(cur_img_chip_record)

        return self.chip_records

    def _get_chips_records(self, rec, chips, scale_i):
        """
        获取切片记录

        Args:
            rec: 记录
            chips: 切片列表
            scale_i: 尺度索引

        Returns:
            当前图像切片记录列表
        """
        cur_img_chip_records = []
        ori_im_h = rec["h"]
        ori_im_w = rec["w"]
        im_file = rec["im_file"]
        ori_im_id = rec["im_id"]
        for id, chip in enumerate(chips):
            chip_rec = {}
            x1, y1, x2, y2 = chip
            chip_h = y2 - y1
            chip_w = x2 - x1
            chip_rec["im_file"] = im_file
            chip_rec["im_id"] = self._global_chip_id
            chip_rec["h"] = chip_h
            chip_rec["w"] = chip_w
            chip_rec["chip"] = chip
            chip_rec["ori_im_h"] = ori_im_h
            chip_rec["ori_im_w"] = ori_im_w
            chip_rec["scale_i"] = scale_i

            self._global_chip_id2img_id[self._global_chip_id] = int(ori_im_id)
            self._global_chip_id += 1
            cur_img_chip_records.append(chip_rec)

        return cur_img_chip_records

    def aggregate_chips_detections(self, results, records=None):
        """
        # 1. 将切片检测转换为图像检测
        # 2. 对每张图像进行NMS;
        # 3. 格式化输出结果
        Args:
            results: 检测结果
            records: 记录列表

        Returns:
            聚合结果
        """
        results = deepcopy(results)
        records = records if records else self.chip_records
        img_id2bbox = self._transform_chip2image_bboxes(results, records)
        nms_img_id2bbox = self._nms_dets(img_id2bbox)
        aggregate_results = self._reformat_results(nms_img_id2bbox)
        return aggregate_results

    def _transform_chip2image_bboxes(self, results, records):
        """
        将切片边界框转换为图像边界框

        Args:
            results: 检测结果
            records: 记录列表

        Returns:
            图像ID到边界框的映射
        """
        # 1. 将切片检测转换为图像检测;
        # 2. 过滤有效范围;
        # 3. 重新格式化并聚合切片检测以获得scale_cls_dets
        img_id2bbox = defaultdict(list)
        for result in results:
            bbox_locs = result['bbox']
            bbox_nums = result['bbox_num']
            if len(bbox_locs) == 1 and bbox_locs[0][
                0] == -1:  # 当前批次没有检测
                # bbox_locs = array([[-1.]], dtype=float32); bbox_nums = [[1]]
                # MultiClassNMS输出: 如果所有图像都没有检测到边界框，lod将被设置为{1}，Out只包含一个值，即-1。
                continue
            im_ids = result['im_id']  # 替换为range(len(bbox_nums))

            last_bbox_num = 0
            for idx, im_id in enumerate(im_ids):

                cur_bbox_len = bbox_nums[idx]
                bboxes = bbox_locs[last_bbox_num:last_bbox_num + cur_bbox_len]
                last_bbox_num += cur_bbox_len
                # box: [num_id, score, xmin, ymin, xmax, ymax]
                if len(bboxes) == 0:  # 当前图像没有检测
                    continue

                chip_rec = records[int(im_id) -
                                   1]  # im_id从1开始，类型是np.int64
                image_size = max(chip_rec["ori_im_h"], chip_rec["ori_im_w"])

                bboxes = transform_chip_boxes2image_boxes(
                    bboxes, chip_rec["chip"], chip_rec["ori_im_h"],
                    chip_rec["ori_im_w"])

                scale_i = chip_rec["scale_i"]
                cur_scale = self._get_current_scale(self.target_sizes[scale_i],
                                                    image_size)
                _, valid_boxes_idx = self._validate_boxes(
                    self.valid_box_ratio_ranges[scale_i], image_size,
                    bboxes[:, 2:], cur_scale)
                ori_img_id = self._global_chip_id2img_id[int(im_id)]

                img_id2bbox[ori_img_id].append(bboxes[valid_boxes_idx])

        return img_id2bbox

    def _nms_dets(self, img_id2bbox):
        """
        对检测结果进行NMS

        Args:
            img_id2bbox: 图像ID到边界框的映射

        Returns:
            经过NMS处理的图像ID到边界框的映射
        """
        # 1. 对每张图像-类进行NMS
        # 2. 如果请求，限制检测数量为MAX_PER_IMAGE
        max_per_img = self.max_per_img
        nms_thresh = self.nms_thresh

        for img_id in img_id2bbox:
            box = img_id2bbox[
                img_id]  # np.array列表，形状为[N, 6]，6是[label, score, x1, y1, x2, y2]
            box = np.concatenate(box, axis=0)
            nms_dets = nms(box, nms_thresh)
            if max_per_img > 0:
                if len(nms_dets) > max_per_img:
                    keep = np.argsort(-nms_dets[:, 1])[:max_per_img]
                    nms_dets = nms_dets[keep]

            img_id2bbox[img_id] = nms_dets

        return img_id2bbox

    @staticmethod
    def _reformat_results(img_id2bbox):
        """
        重新格式化结果

        Args:
            img_id2bbox: 图像ID到边界框的映射

        Returns:
            重新格式化的结果列表
        """
        im_ids = img_id2bbox.keys()
        results = []
        for img_id in im_ids:  # 按原始im_id顺序输出
            if len(img_id2bbox[img_id]) == 0:
                bbox = np.array(
                    [[-1., 0., 0., 0., 0., 0.]])  # 边界情况: 没有检测
                bbox_num = np.array([0])
            else:
                # np.array形状为[N, 6]，6是[label, score, x1, y1, x2, y2]
                bbox = img_id2bbox[img_id]
                bbox_num = np.array([len(bbox)])
            res = dict(im_id=np.array([[img_id]]), bbox=bbox, bbox_num=bbox_num)
            results.append(res)
        return results
