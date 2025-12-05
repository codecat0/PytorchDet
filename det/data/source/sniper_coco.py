#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :sniper_coco.py
@Author :CodeCat
@Date   :2025/11/21 15:37
"""
import os
import cv2
import json
import copy
import numpy as np
from collections.abc import Sequence
from loguru import logger

from det.data.crop_utils.annotation_cropper import AnnoCropper
from det.data.source.coco import COCODataset
from det.data.source.dataset import _make_dataset, _is_valid_file


class SniperCOCODataSet(COCODataset):
    """SniperCOCODataSet

    用于SNIPER方法的数据集类，通过AnnoCropper将大图像切分为小切片（chips），
    以便于检测不同尺度的物体。支持训练时的正负样本切片生成和推理时的切片处理。
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 proposals_file=None,  # 可选的提议文件路径，用于生成负样本切片
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,  # 是否加载人群标注
                 allow_empty=True,  # 是否允许空记录
                 empty_ratio=1.,    # 空记录比例
                 is_trainset=True,  # 是否为训练集
                 image_target_sizes=[2000, 1000],  # 图像的目标尺寸列表，用于多尺度处理
                 valid_box_ratio_ranges=[[-1, 0.1],[0.08, -1]],  # 有效边界框比例范围，与image_target_sizes对应
                 chip_target_size=500,  # 切片的目标尺寸
                 chip_target_stride=200,  # 切片的步长
                 use_neg_chip=False,  # 是否使用负样本切片
                 max_neg_num_per_im=8,  # 每张图像最多负样本切片数量
                 max_per_img=-1,  # 每张图像最大检测数量，-1表示无限制
                 nms_thresh=0.5):  # NMS阈值
        super(SniperCOCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            load_crowd=load_crowd,
            allow_empty=allow_empty,
            empty_ratio=empty_ratio
        )
        self.proposals_file = proposals_file  # 提议文件路径
        self.proposals = None  # 存储解析后的提议
        self.anno_cropper = None  # AnnoCropper实例，用于生成切片
        self.is_trainset = is_trainset  # 是否为训练集
        self.image_target_sizes = image_target_sizes
        self.valid_box_ratio_ranges = valid_box_ratio_ranges
        self.chip_target_size = chip_target_size
        self.chip_target_stride = chip_target_stride
        self.use_neg_chip = use_neg_chip
        self.max_neg_num_per_im = max_neg_num_per_im
        self.max_per_img = max_per_img
        self.nms_thresh = nms_thresh

    def parse_dataset(self):
        """
        解析数据集，如果训练集则合并提议，初始化AnnoCropper并生成切片。
        """
        if not hasattr(self, "roidbs"):
            super(SniperCOCODataSet, self).parse_dataset()
        if self.is_trainset:
            self._parse_proposals()  # 解析提议文件
            self._merge_anno_proposals()  # 将提议合并到注释中
        self.ori_roidbs = copy.deepcopy(self.roidbs)  # 保存原始roidbs
        self.init_anno_cropper()  # 初始化切片生成器
        # 根据是否为训练集生成对应的切片roidbs
        self.roidbs = self.generate_chips_roidbs(self.roidbs, self.is_trainset)

    def set_proposals_file(self, file_path):
        """
        设置提议文件路径。

        Args:
            file_path (str): 提议文件路径
        """
        self.proposals_file = file_path

    def init_anno_cropper(self):
        """
        初始化AnnoCropper实例。
        """
        print("Init AnnoCropper...")
        self.anno_cropper = AnnoCropper(
            image_target_sizes=self.image_target_sizes,
            valid_box_ratio_ranges=self.valid_box_ratio_ranges,
            chip_target_size=self.chip_target_size,
            chip_target_stride=self.chip_target_stride,
            use_neg_chip=self.use_neg_chip,
            max_neg_num_per_im=self.max_neg_num_per_im,
            max_per_img=self.max_per_img,
            nms_thresh=self.nms_thresh
        )

    def generate_chips_roidbs(self, roidbs, is_trainset):
        """
        生成切片的roidbs。

        Args:
            roidbs (list): 原始的roidbs
            is_trainset (bool): 是否为训练集

        Returns:
            list: 切片的roidbs
        """
        if is_trainset:
            # 训练时调用crop_anno_records，会生成正负样本切片
            roidbs = self.anno_cropper.crop_anno_records(roidbs)
        else:
            # 推理时调用crop_infer_anno_records，只生成覆盖全图的切片
            roidbs = self.anno_cropper.crop_infer_anno_records(roidbs)
        return roidbs

    def _parse_proposals(self):
        """
        解析提议文件，将其转换为字典格式。
        提议文件应为JSON格式，包含'image_id'和'bbox'字段。
        """
        if self.proposals_file:
            self.proposals = {}
            print("Parse proposals file:{}".format(self.proposals_file))
            with open(self.proposals_file, 'r') as f:
                proposals = json.load(f)
            for prop in proposals:
                image_id = prop["image_id"]
                if image_id not in self.proposals:
                    self.proposals[image_id] = []
                x, y, w, h = prop["bbox"]  # COCO格式的bbox是[x, y, w, h]
                # 转换为[x1, y1, x2, y2]格式
                self.proposals[image_id].append([x, y, x + w, y + h])

    def _merge_anno_proposals(self):
        """
        将提议合并到原始注释记录中，为生成负样本切片做准备。
        """
        assert self.roidbs
        if self.proposals and len(self.proposals.keys()) > 0:
            print("merge proposals to annos")
            for id, record in enumerate(self.roidbs):
                image_id = int(record["im_id"])
                if image_id not in self.proposals.keys():
                    print("image id :{} no proposals".format(image_id))
                # 将对应图像的提议添加到记录中
                record["proposals"] = np.array(self.proposals.get(image_id, []), dtype=np.float32)
                self.roidbs[id] = record

    def get_ori_roidbs(self):
        """
        获取原始的roidbs（未经过切片处理）。

        Returns:
            list or None: 原始roidbs或None
        """
        if not hasattr(self, "ori_roidbs"):
            return None
        return self.ori_roidbs

    def get_roidbs(self):
        """
        获取roidbs（经过切片处理）。

        Returns:
            list: 切片的roidbs
        """
        if not hasattr(self, "roidbs"):
            self.parse_dataset()
        return self.roidbs

    def set_roidbs(self, roidbs):
        """
        设置roidbs。

        Args:
            roidbs (list): roidbs列表
        """
        self.roidbs = roidbs

    def check_or_download_dataset(self):
        """
        检查或下载数据集（SNIPER数据集不需要此操作）。
        """
        return

    def _parse(self):
        """
        解析图像文件路径。
        """
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        """
        加载图像文件路径列表（用于推理时）。
        """
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            im = cv2.imread(image)
            h, w, c = im.shape
            rec = {'im_id': np.array([ct]), 'im_file': image, "h": h, "w": w}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        """
        获取图像ID到路径的映射。

        Returns:
            dict: 图像ID到路径的映射
        """
        return self._imid2path

    def set_images(self, images):
        """
        设置图像列表并加载为roidbs（用于推理时）。

        Args:
            images (list): 图像路径列表
        """
        self._imid2path = {}
        self.image_dir = images
        self.roidbs = self._load_images()