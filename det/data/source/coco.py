#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :coco.py
@Author :CodeCat
@Date   :2025/11/5 11:37
"""
import os
import copy
from collections import Sequence
import numpy as np
from loguru import logger

from .dataset import DetDataset


__all__ = ['COCODataset']


class COCODataset(DetDataset):
    """
    初始化COCODataset对象。

    Args:
        dataset_dir (str, optional): 数据集目录路径。默认为None。
        image_dir (str, optional): 图像目录路径。默认为None。
        anno_path (str, optional): 标注文件路径。默认为None。
        data_fields (list, optional): 数据字段列表，指定需要从数据集中加载哪些字段。默认为['image']。
        sample_num (int, optional): 采样数量。默认为-1，表示加载所有数据。
        load_crowd (bool, optional): 是否加载crowd数据。默认为False。
        allow_empty (bool, optional): 是否允许数据集为空。默认为False。
        empty_ratio (float, optional): 数据集为空时的采样比例。默认为1.0。
        repeat (int, optional): 数据重复次数。默认为1。

    """
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.0,
                 repeat=1):
        super(COCODataset, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat,
        )
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        """
        从给定的记录列表中随机采样出指定数量的空记录。
        Args:
            records (list): 包含记录的列表。
            num (int): 需要采样的空记录数量。
        Returns:
            list: 包含采样得到的空记录的列表。
        """
        if self.empty_ratio < 0. or self.empty_ratio > 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        """
        解析数据集
        Args:
            无
        Returns:
            无
        Raises:
            AssertionError: 如果给定的注释文件不是以.json结尾，则抛出异常。
        """
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                    im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(
                                box['segmentation'],
                                dtype=object).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                    if 'track_id' in box:
                        gt_track_id[i][0] = box['track_id']
                        has_track_id = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }
                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records

