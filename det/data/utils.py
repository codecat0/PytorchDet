#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :utils.py
@Author :CodeCat
@Date   :2025/11/5 16:49
"""
import torch
import numbers
import numpy as np
from collections.abc import Sequence, Mapping


def default_collate_fn(batch):
    """
    对batch数据进行处理，将其整理成统一格式。

    Args:
        batch (list): 包含多个样本的列表，样本类型可以包括tensor、numpy.ndarray、dict、list、number等。

    Returns:
        处理后的batch数据，类型与输入样本一致，但进行了统一整理。

    Raises:
        RuntimeError: 如果一个batch中的样本字段数量不一致，则抛出此异常。
        TypeError: 如果输入样本类型不符合预期（非tensor、numpy.ndarray、dict、list、number），则抛出此异常。

    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return torch.from_numpy(batch)
    elif isinstance(sample, torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(sample, (float, int, bool, np.number)):
        return torch.tensor(batch, dtype=torch.float32 if isinstance(sample, (float, np.floating)) else torch.long)
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: default_collate_fn([d[key] for d in batch])
            for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))