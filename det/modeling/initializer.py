#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :initializer.py
@Author :CodeCat
@Date   :2025/11/14 10:51
"""
import torch
import math
from torch.nn.init import uniform_

def linear_init_(module):
    """为线性层初始化权重和偏置"""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def conv_init_(module):
    """为卷积层初始化权重和偏置"""
    import numpy as np
    bound = 1 / math.sqrt(float(torch.prod(torch.tensor(module.weight.shape[1:]))))
    uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        uniform_(module.bias, -bound, bound)

