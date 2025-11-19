#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :voc.py
@Author :CodeCat
@Date   :2025/11/19 17:30
"""
import os
import numpy as np

import xml.etree.ElementTree as ET
from det.data.source.dataset import DetDataset


class VOCDataSet(DetDataset):
    """
    加载PascalVOC格式的数据集

    注意:
    `anno_path` 必须包含xml文件和图像文件路径的注释。

    Args:
        dataset_dir (str): 数据集根目录
        image_dir (str): 图像目录
        anno_path (str): voc注释文件路径
        data_fields (list): 数据字典的键名，至少包含'image'
        sample_num (int): 要加载的样本数，-1表示全部
        label_list (str): 如果use_default_label为False，将加载
            类别与类别索引之间的映射
        allow_empty (bool): 是否加载空条目。默认为False
        empty_ratio (float): 空记录数量与总记录数的比例，
            如果empty_ratio超出[0.,1.)范围，则不采样记录并使用所有空条目。默认为1.
        repeat (int): 数据集重复次数，用于基准测试
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 label_list=None,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(VOCDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat)
        self.label_list = label_list
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        """
        采样空记录

        Args:
            records: 记录列表
            num: 总记录数

        Returns:
            采样后的记录列表
        """
        # 如果empty_ratio超出[0.,1.)范围，则不采样记录
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        """
        解析数据集
        """
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        # 将类别名称映射到类别ID
        # first_class:0, second_class:1, ...
        records = []
        empty_records = []
        ct = 0
        cname2cid = {}
        if self.label_list:
            label_path = os.path.join(self.dataset_dir, self.label_list)
            if not os.path.exists(label_path):
                raise ValueError("label_list {} does not exists".format(
                    label_path))
            with open(label_path, 'r') as fr:
                label_id = 0
                for line in fr.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            cname2cid = pascalvoc_label()

        with open(anno_path, 'r') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                img_file, xml_file = [os.path.join(image_dir, x) \
                                      for x in line.strip().split()[:2]]
                if not os.path.exists(img_file):
                    print(
                        'Illegal image file: {}, and it will be ignored'.format(
                            img_file))
                    continue
                if not os.path.isfile(xml_file):
                    print(
                        'Illegal xml file: {}, and it will be ignored'.format(
                            xml_file))
                    continue
                tree = ET.parse(xml_file)
                if tree.find('id') is None:
                    im_id = np.array([ct])
                else:
                    im_id = np.array([int(tree.find('id').text)])

                objs = tree.findall('object')
                im_w = float(tree.find('size').find('width').text)
                im_h = float(tree.find('size').find('height').text)
                if im_w < 0 or im_h < 0:
                    print(
                        'Illegal width: {} or height: {} in annotation, '
                        'and {} will be ignored'.format(im_w, im_h, xml_file))
                    continue

                num_bbox, i = len(objs), 0
                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_score = np.zeros((num_bbox, 1), dtype=np.float32)
                difficult = np.zeros((num_bbox, 1), dtype=np.int32)
                for obj in objs:
                    cname = obj.find('name').text

                    # 用户数据集可能不包含difficult字段
                    _difficult = obj.find('difficult')
                    _difficult = int(
                        _difficult.text) if _difficult is not None else 0

                    x1 = float(obj.find('bndbox').find('xmin').text)
                    y1 = float(obj.find('bndbox').find('ymin').text)
                    x2 = float(obj.find('bndbox').find('xmax').text)
                    y2 = float(obj.find('bndbox').find('ymax').text)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(im_w - 1, x2)
                    y2 = min(im_h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        gt_bbox[i, :] = [x1, y1, x2, y2]
                        gt_class[i, 0] = cname2cid[cname]
                        gt_score[i, 0] = 1.
                        difficult[i, 0] = _difficult
                        i += 1
                    else:
                        print(
                            'Found an invalid bbox in annotations: xml_file: {}'
                            ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                xml_file, x1, y1, x2, y2))
                gt_bbox = gt_bbox[:i, :]
                gt_class = gt_class[:i, :]
                gt_score = gt_score[:i, :]
                difficult = difficult[:i, :]

                voc_rec = {
                    'im_file': img_file,
                    'im_id': im_id,
                    'h': im_h,
                    'w': im_w
                } if 'image' in self.data_fields else {}

                gt_rec = {
                    'gt_class': gt_class,
                    'gt_score': gt_score,
                    'gt_bbox': gt_bbox,
                    'difficult': difficult
                }
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        voc_rec[k] = v

                if len(objs) == 0:
                    empty_records.append(voc_rec)
                else:
                    records.append(voc_rec)

                ct += 1
                if self.sample_num > 0 and ct >= self.sample_num:
                    break
        assert ct > 0, 'not found any voc record in %s' % (self.anno_path)
        print('{} samples in file {}'.format(ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs, self.cname2cid = records, cname2cid

    def get_label_list(self):
        """
        获取标签列表路径

        Returns:
            标签列表文件路径
        """
        return os.path.join(self.dataset_dir, self.label_list)


def pascalvoc_label():
    labels_map = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }
    return labels_map
