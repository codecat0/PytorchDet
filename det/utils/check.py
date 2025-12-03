#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :check.py
@Author :CodeCat
@Date   :2025/12/2 15:25
"""
import sys
import torch
from det.utils.logger import setup_logger

logger = setup_logger(__name__)


def check_gpu(use_gpu):
    """
    检查是否启用了 GPU 但当前 PyTorch 不支持 CUDA（即 CPU-only 版本）。

    Args:
        use_gpu (bool): 配置中是否启用了 GPU
    """
    err = ("Config use_gpu cannot be set as true while you are "
           "using PyTorch CPU-only version!\n"
           "Please try:\n"
           "\t1. Install torch with CUDA support (e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`)\n"
           "\t2. Set use_gpu=False in config file to run model on CPU")

    if use_gpu and not torch.cuda.is_available():
        logger.error(err)
        sys.exit(1)


def check_config(cfg):
    """
    检查配置文件的正确性。如果关键字段缺失，则记录错误并退出程序。

    Args:
        cfg (dict or OmegaConf.DictConfig): 配置对象

    Returns:
        cfg: 修正后的配置（确保包含必要字段）
    """
    err = "'{}' not specified in config file. Please set it in config file."
    check_list = ['num_classes']

    try:
        for var in check_list:
            # 兼容 dict 和 OmegaConf 对象的访问方式
            if var not in cfg:
                logger.error(err.format(var))
                sys.exit(1)
    except Exception as e:
        # 容错：避免因配置格式异常导致崩溃
        pass

    # 设置默认值：若未指定 log_iter，则设为 20
    if 'log_iter' not in cfg:
        cfg['log_iter'] = 20  # 若 cfg 是 dict
        # 如果 cfg 是 OmegaConf.DictConfig，应使用：
        # cfg.log_iter = 20

    return cfg