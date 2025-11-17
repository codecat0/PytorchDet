#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :utils.py
@Author :CodeCat
@Date   :2025/11/17 10:29
"""
from torch import nn
from typing import List


def get_bn_running_state_names(model: nn.Module) -> List[str]:
    """
    获取所有BN层的状态完整名称，包括运行均值和方差
    """
    names = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            # 检查模块是否具有运行均值和方差属性
            assert hasattr(m, 'running_mean'), f'assert {m} has running_mean'
            assert hasattr(m, 'running_var'), f'assert {m} has running_var'

            # 构造运行均值和方差的完整名称
            running_mean = f'{n}.running_mean'
            running_var = f'{n}.running_var'
            names.extend([running_mean, running_var])

    return names