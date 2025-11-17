#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :compact.py
@Author :CodeCat
@Date   :2025/11/17 11:41
"""
import PIL

def imagedraw_textsize_c(draw, text, font=None):
    if int(PIL.__version__.split('.')[0]) < 10:
        tw, th = draw.textsize(text, font=font)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top

    return tw, th
