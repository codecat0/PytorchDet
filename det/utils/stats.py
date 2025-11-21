#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :stats.py
@Author :CodeCat
@Date   :2025/11/21 16:13
"""
import collections
import numpy as np


class SmoothedValue(object):
    """跟踪一系列值，并提供对窗口平滑值或全局序列平均值的访问。
    """

    def __init__(self, window_size=20, fmt=None):
        """
        初始化平滑值跟踪器。

        Args:
            window_size (int): 滑动窗口的大小，默认为20。
            fmt (str): 字符串格式化模板，默认为"{median:.4f} ({avg:.4f})"。
        """
        if fmt is None:
            fmt = "{median:.4f} ({avg:.4f})"
        self.deque = collections.deque(maxlen=window_size)  # 滑动窗口，存储最近的值
        self.fmt = fmt  # 字符串格式化模板
        self.total = 0.  # 所有值的总和
        self.count = 0  # 所有值的总数量

    def update(self, value, n=1):
        """
        更新当前值。

        Args:
            value: 新的值
            n: 该值的数量，默认为1
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        """
        获取滑动窗口中值的中位数。

        Returns:
            float: 中位数
        """
        return np.median(self.deque)

    @property
    def avg(self):
        """
        获取滑动窗口中值的平均数。

        Returns:
            float: 平均数
        """
        return np.mean(self.deque)

    @property
    def max(self):
        """
        获取滑动窗口中值的最大值。

        Returns:
            float: 最大值
        """
        return np.max(self.deque)

    @property
    def value(self):
        """
        获取滑动窗口中的最新值。

        Returns:
            float: 最新值
        """
        return self.deque[-1]

    @property
    def global_avg(self):
        """
        获取所有历史值的全局平均值。

        Returns:
            float: 全局平均值
        """
        return self.total / self.count

    def __str__(self):
        """
        返回格式化的字符串表示。

        Returns:
            str: 格式化后的字符串
        """
        return self.fmt.format(
            median=self.median, avg=self.avg, max=self.max, value=self.value)


class TrainingStats(object):
    def __init__(self, window_size, delimiter=' '):
        """
        初始化训练统计信息跟踪器

        Args:
            window_size (int): 平滑值的滑动窗口大小
            delimiter (str): 日志字符串中各项之间的分隔符，默认为空格
        """
        self.meters = None  # 存储各个指标的SmoothedValue对象
        self.window_size = window_size  # 滑动窗口大小
        self.delimiter = delimiter  # 分隔符

    def update(self, stats):
        """
        更新统计信息

        Args:
            stats (dict): 包含当前批次统计信息的字典，如 {'loss': 0.5, 'acc': 0.8}
        """
        if self.meters is None:
            # 首次调用时，为每个统计键创建SmoothedValue实例
            self.meters = {
                k: SmoothedValue(self.window_size)
                for k in stats.keys()
            }
        # 更新每个指标的值
        for k, v in self.meters.items():
            v.update(float(stats[k]))

    def get(self, extras=None):
        """
        获取当前的统计信息（平滑后的中位数）

        Args:
            extras (dict, optional): 额外要包含的键值对，如 {'epoch': 1, 'step': 100}

        Returns:
            collections.OrderedDict: 包含所有统计信息的有序字典，值被格式化为6位小数的字符串
        """
        stats = collections.OrderedDict()
        # 如果有额外信息，先添加
        if extras:
            for k, v in extras.items():
                stats[k] = v
        # 添加平滑后的统计值（使用中位数）
        for k, v in self.meters.items():
            stats[k] = format(v.median, '.6f')

        return stats

    def log(self, extras=None):
        """
        生成格式化的日志字符串

        Args:
            extras (dict, optional): 额外要包含的键值对

        Returns:
            str: 格式化的日志字符串，例如 "loss: 0.123456 acc: 0.789012"
        """
        d = self.get(extras)  # 获取当前统计信息
        strs = []  # 存储 "key: value" 格式的字符串
        for k, v in d.items():
            strs.append("{}: {}".format(k, str(v)))
        # 使用指定分隔符合并所有字符串
        return self.delimiter.join(strs)