#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :detr_loss.py
@Author :CodeCat
@Date   :2025/11/14 15:29
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from det.modeling.losses.iou_loss import GIoULoss
from det.modeling.transformers.utils import bbox_cxcywh_to_xyxy, sigmoid_focal_loss, varifocal_loss_with_logits
from det.modeling.bbox_utils import bbox_iou


