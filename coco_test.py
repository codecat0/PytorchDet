#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :coco_test.py
@Author :CodeCat
@Date   :2025/11/4 10:54
"""
import json
from pycocotools.coco import COCO


annotations_file = './annotations/instances_val2017.json'
coco = COCO(annotations_file)
img_ids = coco.getImgIds()
img_anno = coco.loadImgs(img_ids[0])[0]
im_fname = img_anno['file_name']
im_width = float(img_anno['width'])
im_height = float(img_anno['height'])
print(f'Image filename: {im_fname}, width: {im_width}, height: {im_height}')

ins_anno_ids = coco.getAnnIds(imgIds=img_ids[0])
print(f'Number of instances: {len(ins_anno_ids)}')
print(ins_anno_ids)
instances = coco.loadAnns(ins_anno_ids)
print(f'Number of instances: {len(instances)}')
print(instances)