#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :gridmask_utils.py
@Author :CodeCat
@Date   :2025/11/6 10:32
"""
import numpy as np
from PIL import Image


class Gridmask(object):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 upper_iter=360000):
        """
        初始化 Gridmask 类。
        Args:
            use_h (bool): 是否在高度方向上应用网格掩码。默认为 True。
            use_w (bool): 是否在宽度方向上应用网格掩码。默认为 True。
            rotate (int): 网格掩码的旋转角度。默认为 1。
            offset (bool): 是否在网格掩码上应用偏移。默认为 False。
            ratio (float): 网格掩码占图像的比例。默认为 0.5。
            mode (int): 网格掩码的模式。默认为 1。
            prob (float): 应用网格掩码的概率。默认为 0.7。
            upper_iter (int): 在前upper_iter次迭代中，prob从 0 线性增长到prob。默认为 360000。
        """
        super(Gridmask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.st_prob = prob
        self.upper_iter = upper_iter

    def __call__(self, x, curr_iter):
        # 动态调整应用概率
        self.prob = self.st_prob * min(1, 1.0 * curr_iter / self.upper_iter)
        if np.random.rand() > self.prob:
            return x
        h, w, _ = x.shape
        # 创建更大的画布，防止旋转后出现空白
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)  # 遮挡周期（grid spacing）
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)  # 每个周期内遮挡的宽度
        mask = np.ones((hh, ww), np.float32)
        # 随机起始偏移
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        # 画横向遮挡条（水平方向每隔 d 像素画一条宽 l 的黑线）
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        # 画纵向遮挡条 （垂直方向每隔 d 像素画一条宽 l 的黑线）
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0
        # 随机旋转掩码
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        # 裁剪回原图大小（居中裁剪）
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2
                    + w].astype(np.float32)

        if self.mode == 1:
            mask = 1 - mask  # 反转：现在 1 = 保留，0 = 遮挡
        mask = np.expand_dims(mask, axis=-1)
        if self.offset:
            offset = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            x = (x * mask + offset * (1 - mask)).astype(x.dtype)
        else:
            x = (x * mask).astype(x.dtype)  # 被遮挡区域变黑

        return x