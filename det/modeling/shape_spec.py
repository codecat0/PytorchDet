#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :shape_spec.py
@Author :CodeCat
@Date   :2025/11/12 10:46
"""
# The code is based on:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/shape_spec.py

from collections import namedtuple


class ShapeSpec(
    namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width,
                                             stride)
