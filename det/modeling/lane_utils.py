#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :lane_utils.py
@Author :CodeCat
@Date   :2025/11/21 17:03
"""
import os
import cv2
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        """
        车道线类，用于表示和插值车道线

        Args:
            points (np.ndarray): 车道线上的点坐标，形状为 (N, 2)，每行包含 [x, y] 坐标
            invalid_value (float): 当插值超出有效范围时返回的值
            metadata (dict): 元数据字典
        """
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        # 使用样条插值创建从y坐标到x坐标的函数
        self.function = InterpolatedUnivariateSpline(
            points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        # 定义有效y坐标的范围（稍微扩展以允许一些外推）
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01
        self.metadata = metadata or {}

    def __repr__(self):
        """
        返回车道线的字符串表示

        Returns:
            str: 车道线的字符串表示
        """
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        """
        根据y坐标计算对应的x坐标

        Args:
            lane_ys (np.ndarray or float): y坐标值或数组

        Returns:
            np.ndarray or float: 对应的x坐标值或数组，超出有效范围的值设为invalid_value
        """
        lane_xs = self.function(lane_ys)

        # 将超出有效y范围的x坐标设为无效值
        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y
                                          )] = self.invalid_value
        return lane_xs

    def to_array(self, sample_y_range, img_w, img_h):
        """
        将车道线转换为采样点数组

        Args:
            sample_y_range (tuple): y坐标采样范围 (start, end, step)
            img_w (int): 图像宽度
            img_h (int): 图像高度

        Returns:
            np.ndarray: 车道线采样点数组，形状为 (M, 2)，每行包含 [x, y] 坐标
        """
        self.sample_y = range(sample_y_range[0], sample_y_range[1],
                              sample_y_range[2])
        sample_y = self.sample_y
        img_w, img_h = img_w, img_h
        # 将y坐标归一化到 [0, 1] 范围
        ys = np.array(sample_y) / float(img_h)
        # 计算对应的x坐标
        xs = self(ys)
        # 过滤有效范围内的点（x坐标在 [0, 1) 范围内）
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w  # 反归一化到原始图像坐标
        lane_ys = ys[valid_mask] * img_h  # 反归一化到原始图像坐标
        # 组合成 [x, y] 坐标对
        lane = np.concatenate(
            (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
        return lane

    def __iter__(self):
        """
        返回迭代器对象

        Returns:
            Lane: 自身作为迭代器
        """
        return self

    def __next__(self):
        """
        获取下一个点坐标

        Returns:
            np.ndarray: 下一个点的坐标 [x, y]

        Raises:
            StopIteration: 当所有点都已迭代完时抛出
        """
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def imshow_lanes(img, lanes, show=False, out_file=None, width=4):
    """
    在图像上绘制车道线

    Args:
        img (np.ndarray): 输入图像，形状为 (H, W, C)
        lanes (list): 车道线列表，每个元素是一个Lane对象或包含点坐标的列表
        show (bool): 是否显示图像，默认为False
        out_file (str): 输出文件路径，如果为None则不保存，默认为None
        width (int): 绘制线的宽度，默认为4
    """
    lanes_xys = []
    # 遍历所有车道线，提取有效的坐标点
    for _, lane in enumerate(lanes):
        xys = []
        # 遍历车道线上的每个点
        for x, y in lane:
            # 过滤掉无效坐标（小于等于0的点）
            if x <= 0 or y <= 0:
                continue
            # 将坐标转换为整数
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)

    # 按照每条车道线的第一个点的x坐标进行排序
    lanes_xys.sort(key=lambda xys: xys[0][0] if len(xys) > 0 else 0)

    # 绘制每条车道线
    for idx, xys in enumerate(lanes_xys):
        # 遍历当前车道线上的相邻点对，绘制线段
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)

    # 如果需要显示图像
    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    # 如果需要保存图像
    if out_file:
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        cv2.imwrite(out_file, img)