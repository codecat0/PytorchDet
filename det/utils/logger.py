#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :logger.py
@Author :CodeCat
@Date   :2025/11/24 16:40
"""
import logging
import os
import sys

import torch.distributed as dist

__all__ = ['setup_logger']

logger_initialized = []


def setup_logger(name="det", output=None, log_ranks="0"):
    """
    初始化日志记录器，并将日志级别设置为 INFO。

    Args:
        output (str): 保存日志的文件名或目录。如果为 None，则不保存日志文件。
                      如果以 ".txt" 或 ".log" 结尾，视为文件名；
                      否则日志将保存为 `output/log.txt`。
        name (str): 日志记录器的根模块名称。
        log_ranks (str): 要记录日志的 GPU ID（多卡时用逗号分隔），默认为 "0"。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 定义日志格式
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )

    # 解析 log_ranks 参数
    if isinstance(log_ranks, str):
        log_ranks = [int(i) for i in log_ranks.split(',')]
    elif isinstance(log_ranks, int):
        log_ranks = [log_ranks]

    # 获取当前进程的 rank（支持单机多卡或多机多卡）
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    # 控制台日志：仅指定 rank 输出
    if local_rank in log_ranks:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 文件日志：所有 rank 都输出（避免日志冲突，每个 rank 写独立文件）
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)  # 统一日志格式
        logger.addHandler(fh)

    logger_initialized.append(name)
    return logger