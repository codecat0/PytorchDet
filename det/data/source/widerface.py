#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :widerface.py
@Author :CodeCat
@Date   :2025/11/19 17:37
"""
from collections import defaultdict
import os
import numpy as np
from scipy.io import loadmat

from det.data.source.dataset import DetDataset
from loguru import logger


class WIDERFaceDataSet(DetDataset):
    """
    加载WiderFace数据集记录

    Args:
        dataset_dir (str): 数据集根目录
        image_dir (str): 图像目录
        anno_path (str): WiderFace注释数据
        data_fields (list): 数据字典的键名，至少包含'image'
        sample_num (int): 要加载的样本数，-1表示全部
        with_lmk (bool): 是否加载人脸关键点标签
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 with_lmk=False):
        super(WIDERFaceDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            with_lmk=with_lmk)
        self.anno_path = anno_path
        self.sample_num = sample_num
        self.roidbs = None
        self.cname2cid = None
        self.with_lmk = with_lmk

    def parse_dataset(self):
        """
        解析数据集
        """
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        txt_file = anno_path

        records = []
        ct = 0
        file_lists = self._load_file_list(txt_file)
        cname2cid = widerface_label()

        for item in file_lists:
            im_fname = item[0]  # 图像文件名
            im_id = np.array([ct])  # 图像ID
            # 初始化边界框、类别和关键点标签数组
            gt_bbox = np.zeros((len(item) - 1, 4), dtype=np.float32)
            gt_class = np.zeros((len(item) - 1, 1), dtype=np.int32)
            gt_lmk_labels = np.zeros((len(item) - 1, 10), dtype=np.float32)
            lmk_ignore_flag = np.zeros((len(item) - 1, 1), dtype=np.int32)

            # 处理每个检测框
            for index_box in range(len(item)):
                if index_box < 1:
                    continue
                gt_bbox[index_box - 1] = item[index_box][0]  # 边界框坐标
                if self.with_lmk:
                    gt_lmk_labels[index_box - 1] = item[index_box][1]  # 关键点坐标
                    lmk_ignore_flag[index_box - 1] = item[index_box][2]  # 关键点忽略标志

            # 构建图像文件路径
            im_fname = os.path.join(image_dir,
                                    im_fname) if image_dir else im_fname
            # 创建WIDERFACE记录字典
            widerface_rec = {
                'im_file': im_fname,
                'im_id': im_id,
            } if 'image' in self.data_fields else {}

            gt_rec = {
                'gt_bbox': gt_bbox,
                'gt_class': gt_class,
            }
            # 添加需要的数据字段
            for k, v in gt_rec.items():
                if k in self.data_fields:
                    widerface_rec[k] = v
            # 如果需要关键点，则添加关键点相关字段
            if self.with_lmk:
                widerface_rec['gt_keypoint'] = gt_lmk_labels
                widerface_rec['keypoint_ignore'] = lmk_ignore_flag

            if len(item) != 0:
                records.append(widerface_rec)

            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert len(records) > 0, 'not found any widerface in %s' % (anno_path)
        print('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid

    def _load_file_list(self, input_txt):
        """
        加载文件列表

        Args:
            input_txt: 输入的文本文件路径

        Returns:
            解析后的文件列表
        """
        with open(input_txt, 'r') as f_dir:
            lines_input_txt = f_dir.readlines()

        file_dict = {}
        num_class = 0
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]

        for i in range(len(lines_input_txt)):
            line_txt = lines_input_txt[i].strip('\n\t\r')
            split_str = line_txt.split(' ')
            if len(split_str) == 1:
                # 处理图像文件名行
                img_file_name = os.path.split(split_str[0])[1]
                split_txt = img_file_name.split('.')
                if len(split_txt) < 2:
                    continue
                elif split_txt[-1] in exts:
                    if i != 0:
                        num_class += 1
                    file_dict[num_class] = [line_txt]
            else:
                # 处理标注框行
                if len(line_txt) <= 6:
                    continue
                result_boxs = []
                xmin = float(split_str[0])
                ymin = float(split_str[1])
                w = float(split_str[2])
                h = float(split_str[3])
                # 过滤错误的标签
                if w < 0 or h < 0:
                    print('Illegal box with w: {}, h: {} in '
                          'img: {}, and it will be ignored'.format(
                        w, h, file_dict[num_class][0]))
                    continue
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = xmin + w
                ymax = ymin + h
                gt_bbox = [xmin, ymin, xmax, ymax]
                result_boxs.append(gt_bbox)

                if self.with_lmk:
                    # 确保有足够的字符来解析关键点
                    assert len(split_str) > 18, 'When `with_lmk=True`, the number' \
                                                'of characters per line in the annotation file should' \
                                                'exceed 18.'
                    # 解析5个关键点坐标
                    lmk0_x = float(split_str[5])
                    lmk0_y = float(split_str[6])
                    lmk1_x = float(split_str[8])
                    lmk1_y = float(split_str[9])
                    lmk2_x = float(split_str[11])
                    lmk2_y = float(split_str[12])
                    lmk3_x = float(split_str[14])
                    lmk3_y = float(split_str[15])
                    lmk4_x = float(split_str[17])
                    lmk4_y = float(split_str[18])
                    # 关键点忽略标志
                    lmk_ignore_flag = 0 if lmk0_x == -1 else 1
                    gt_lmk_label = [
                        lmk0_x, lmk0_y, lmk1_x, lmk1_y, lmk2_x, lmk2_y, lmk3_x,
                        lmk3_y, lmk4_x, lmk4_y
                    ]
                    result_boxs.append(gt_lmk_label)
                    result_boxs.append(lmk_ignore_flag)
                file_dict[num_class].append(result_boxs)

        return list(file_dict.values())


def widerface_label():
    labels_map = {'face': 0}
    return labels_map


class WIDERFaceValDataset(WIDERFaceDataSet):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 gt_mat_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 with_lmk=False):
        """
        WIDERFace验证数据集类

        Args:
            dataset_dir: 数据集根目录
            image_dir: 图像目录
            anno_path: 注释文件路径
            gt_mat_path: GT mat文件路径
            data_fields: 数据字段列表
            sample_num: 样本数量
            with_lmk: 是否包含关键点
        """
        super().__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            with_lmk=with_lmk)
        self.gt_mat_path = gt_mat_path
        # 设置各种mat文件路径
        self.val_mat = os.path.join(self.dataset_dir, self.gt_mat_path, 'wider_face_val.mat')
        self.hard_mat_path = os.path.join(self.dataset_dir, self.gt_mat_path, 'wider_hard_val.mat')
        self.medium_mat_path = os.path.join(self.dataset_dir, self.gt_mat_path, 'wider_medium_val.mat')
        self.easy_mat_path = os.path.join(self.dataset_dir, self.gt_mat_path, 'wider_easy_val.mat')

        # 验证mat文件是否存在
        assert os.path.exists(self.val_mat), f'{self.val_mat} not exist'
        assert os.path.exists(self.hard_mat_path), f'{self.hard_mat_path} not exist'
        assert os.path.exists(self.medium_mat_path), f'{self.medium_mat_path} not exist'
        assert os.path.exists(self.easy_mat_path), f'{self.easy_mat_path} not exist'

    def parse_dataset(self):
        """
        解析数据集
        """
        super().parse_dataset()

        box_list, flie_list, event_list, hard_info_list, medium_info_list, \
        easy_info_list = self.get_gt_infos()
        setting_infos = [easy_info_list, medium_info_list, hard_info_list]
        settings = ['easy', 'medium', 'hard']
        info_by_name = defaultdict(dict)

        # 为不同难度设置解析GT信息
        for setting_id in range(3):
            info_list = setting_infos[setting_id]
            setting = settings[setting_id]
            for i in range(len(event_list)):
                img_list = flie_list[i][0]
                gt_box_list = box_list[i][0]
                sub_info_list = info_list[i][0]
                for j in range(len(img_list)):
                    img_name = str(img_list[j][0][0])
                    gt_boxes = gt_box_list[j][0].astype(np.float32)
                    info_by_name[img_name]['gt_ori_bbox'] = gt_boxes

                    keep_index = sub_info_list[j][0]
                    ignore = np.zeros(gt_boxes.shape[0])
                    if len(keep_index) != 0:
                        ignore[keep_index - 1] = 1
                    info_by_name[img_name][f'gt_{setting}_ignore'] = ignore

        # 更新roidb信息
        for roidb in self.roidbs:
            img_file = roidb['im_file'].split('/')[-1]
            img_name = ".".join(img_file.split(".")[:-1])
            roidb.update(info_by_name[img_name])

    def get_gt_infos(self):
        """
        获取GT信息

        Returns:
            边界框列表、文件列表、事件列表、难样本信息列表、中等难度样本信息列表、易样本信息列表
        """
        """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

        val_mat = loadmat(self.val_mat)
        hard_mat = loadmat(self.hard_mat_path)
        medium_mat = loadmat(self.medium_mat_path)
        easy_mat = loadmat(self.easy_mat_path)

        box_list = val_mat['face_bbx_list']
        file_list = val_mat['file_list']
        event_list = val_mat['event_list']

        hard_info_list = hard_mat['gt_list']
        medium_info_list = medium_mat['gt_list']
        easy_info_list = easy_mat['gt_list']

        return box_list, file_list, event_list, hard_info_list, medium_info_list, easy_info_list