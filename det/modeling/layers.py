#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :layers.py
@Author :CodeCat
@Date   :2025/11/12 15:29
"""
import math
import six
import numpy as np
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from det.modeling.bbox_utils import delta2bbox
from det.modeling import ops
from torch.nn.init import xavier_normal_, constant_, normal_, zeros_
from torch.nn.init import xavier_uniform_


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class AlignConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        """
        初始化AlignConv层，用于实现可变形卷积操作。
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int, optional): 卷积核大小，默认为3。
            groups (int, optional): 分组卷积的组数，默认为1。
        初始化过程包括：
        - 设置卷积核大小
        - 创建DeformConv2d可变形卷积层
        - 使用正态分布初始化卷积权重（均值0.0，标准差0.01）
        """
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.align_conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False
        )
        normal_(self.align_conv.weight, mean=0.0, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """
        计算锚框相对于特征图网格点的偏移量。
        该函数通过考虑锚框的中心坐标、尺寸和角度，计算每个锚框在特征图上的采样位置
        相对于卷积核中心点的偏移量。主要用于可变形卷积或类似操作中的位置偏移计算。
        Args:
            anchors: 锚框坐标，形状为[B, L, 5]，分别表示(xc, yc, w, h, angle)
            featmap_size: 特征图尺寸，格式为(feat_h, feat_w)
            stride: 特征图相对于输入图像的下采样步长
        Returns:
            offset: 计算得到的偏移量，形状为[B, 2 * kernel_size * kernel_size, feat_h, feat_w]
                    其中2表示y和x两个方向的偏移，kernel_size为卷积核大小
        """
        batch = anchors.size(0)
        dtype = anchors.dtype
        device = anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)

        yy, xx = torch.meshgrid(idx, idx, indexing='ij')
        xx = xx.flatten()
        yy = yy.flatten()

        xc = torch.arange(0, feat_w, dtype=dtype, device=device)
        yc = torch.arange(0, feat_h, dtype=dtype, device=device)
        yc_grid, xc_grid = torch.meshgrid(yc, xc, indexing='ij')  # [feat_h, feat_w]

        xc_flat = xc_grid.flatten().unsqueeze(1)  # [feat_h * feat_w, 1]
        yc_flat = yc_grid.flatten().unsqueeze(1)  # [feat_h * feat_w, 1]
        # x_conv, y_conv shapes: [feat_h * feat_w, kernel_size * kernel_size]
        x_conv = xc_flat + xx  # Broadcasting: [feat_h * feat_w, 1] + [kernel_size * kernel_size] -> [feat_h * feat_w, kernel_size * kernel_size]
        y_conv = yc_flat + yy  # Broadcasting: [feat_h * feat_w, 1] + [kernel_size * kernel_size] -> [feat_h * feat_w, kernel_size * kernel_size]

        # Get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.split(anchors, 1, dim=-1)  # Each has shape [B, L, 1]
        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos = torch.cos(a)
        sin = torch.sin(a)
        dw = w_s / self.kernel_size
        dh = h_s / self.kernel_size
        # Calculate anchor-based offsets from kernel center
        # xx, yy shapes: [kernel_size * kernel_size]
        x = dw * xx  # Broadcasting: [B, L, 1] * [kernel_size * kernel_size] -> [B, L, kernel_size * kernel_size]
        y = dh * yy  # Broadcasting: [B, L, 1] * [kernel_size * kernel_size] -> [B, L, kernel_size * kernel_size]
        # Apply rotation
        xr = cos * x - sin * y  # [B, L, kernel_size * kernel_size]
        yr = sin * x + cos * y  # [B, L, kernel_size * kernel_size]
        # Add to anchor center
        # x_ctr, y_ctr shapes: [B, L, 1]
        # Broadcasting: [B, L, 1] + [B, L, kernel_size * kernel_size] -> [B, L, kernel_size * kernel_size]
        x_anchor = xr + x_ctr
        y_anchor = yr + y_ctr

        # Get offset field
        x_conv_expanded = x_conv.unsqueeze(0)  # [1, feat_h * feat_w, kernel_size * kernel_size]
        y_conv_expanded = y_conv.unsqueeze(0)  # [1, feat_h * feat_w, kernel_size * kernel_size]
        # offset_x, offset_y shapes: [B, feat_h * feat_w, kernel_size * kernel_size]
        offset_x = x_anchor - x_conv_expanded
        offset_y = y_anchor - y_conv_expanded
        # Stack offset_x and offset_y along the last dimension
        # Shape: [B, feat_h * feat_w, kernel_size * kernel_size * 2]
        offset = torch.stack([offset_y, offset_x], dim=-1).flatten(start_dim=2)
        # Reshape to [B, feat_h * feat_w, kernel_size * kernel_size * 2]
        offset = offset.reshape(batch, feat_h, feat_w, -1)
        offset = offset.permute(0, 3, 1, 2)  # [B, kernel_size * kernel_size * 2, feat_h, feat_w]

        return offset

    def forward(self, x, refine_anchors, featmap_size, stride):
        # Get the offset based on anchors
        offset = self.get_offset(refine_anchors, featmap_size, stride)

        if self.training:
            # Detach offset during training to prevent gradients flowing back to anchor calculation
            x = self.align_conv(x, offset.detach())
        else:
            # Use offset directly during inference
            x = self.align_conv(x, offset)

        x = F.relu(x)
        return x


class DeformableConvV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=None,
                 dcn_bias_lr_scale=2.):
        """
        初始化可变形卷积模块(DeformableConvV2)
        该类实现了可变形卷积的第二版实现，包含偏移量生成和可变形卷积操作。
        主要包含两个子模块：
        1. conv_offset: 生成偏移量和掩码的卷积层
        2. conv_dcn: 执行实际可变形卷积操作的层
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int, optional): 卷积步长，默认为1
            padding (int, optional): 填充大小，默认为0
            dilation (int, optional): 空洞率，默认为1
            groups (int, optional): 分组数，默认为1
            weight_attr (optional): 权重参数属性
            bias_attr (optional): 偏置参数属性
            lr_scale (float, optional): 学习率缩放因子，默认为1
            regularizer (optional): 正则化器
            skip_quant (bool, optional): 是否跳过量化，默认为False
            dcn_bias_regularizer (optional): 可变形卷积偏置的正则化器
            dcn_bias_lr_scale (float, optional): 可变形卷积偏置的学习率缩放因子，默认为2.0
        """
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2
        self.mask_channel = kernel_size ** 2

        if lr_scale == 1 and regularizer is None:
            offset_bias = True
        else:
            offset_bias = True

        self.conv_offset = nn.Conv2d(
            in_channels,
            3 * kernel_size ** 2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=offset_bias)

        with torch.no_grad():
            nn.init.zeros_(self.conv_offset.weight)
            if offset_bias:
                nn.init.zeros_(self.conv_offset.bias)

        if bias_attr:
            dcn_bias = True
        else:
            dcn_bias = False

        self.conv_dcn = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            bias=dcn_bias)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :self.offset_channel, :, :]
        mask = offset_mask[:, self.offset_channel:self.offset_channel + self.mask_channel, :, :]
        mask = torch.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask)
        return y


class ConvNormLayer(nn.Module):  # Changed from nn.Layer to nn.Module
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=None,
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=None):
        """
        初始化卷积归一化层，支持普通卷积和可变形卷积，并可选择不同的归一化方式。
        Args:
            ch_in: 输入通道数。
            ch_out: 输出通道数。
            filter_size: 卷积核大小。
            stride: 卷积步长。
            groups: 分组卷积的组数，默认为1。
            norm_type: 归一化类型，可选'bn'、'sync_bn'、'gn'或None，默认为'bn'。
            norm_decay: 归一化权重的衰减系数，默认为0。
            norm_groups: 分组归一化的组数，默认为32。
            use_dcn: 是否使用可变形卷积，默认为False。
            bias_on: 卷积是否使用偏置，默认为False。
            lr_scale: 学习率缩放系数，默认为1。
            freeze_norm: 是否冻结归一化层的参数，默认为False。
            initializer: 权重初始化器，默认为None。
            skip_quant: 是否跳过量化，默认为False。
            dcn_lr_scale: 可变形卷积的学习率缩放系数，默认为2。
            dcn_regularizer: 可变形卷积的正则化器，默认为None。
        Returns:
            None
        """
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if not use_dcn:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=bias_on)
        else:
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=None,
                bias_attr=True,
                lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        if initializer is not None:
            initializer(self.conv.weight)
        else:
            nn.init.normal_(self.conv.weight, mean=0, std=0.01)

        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out)
        else:
            self.norm = None

        if freeze_norm and self.norm is not None:
            for param in self.norm.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        return out


class LiteConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 with_act=True,
                 norm_type='sync_bn',
                 name=None):
        """
        LiteConv模块的初始化函数，构建一个轻量级卷积块。
        该模块由4个卷积层和ReLU6激活函数组成，其中包含深度可分离卷积结构。
        具体结构为：5x5深度可分离卷积 -> ReLU6 -> 1x1卷积 -> [可选ReLU6] ->
        1x1卷积 -> ReLU6 -> 5x5深度可分离卷积 -> [可选ReLU6]
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int, optional): 卷积步长，默认为1
            with_act (bool, optional): 是否在中间层添加ReLU6激活，默认为True
            norm_type (str, optional): 归一化类型，默认为'sync_bn'
            name (str, optional): 模块名称，默认为None
        """
        super(LiteConv, self).__init__()

        conv1 = ConvNormLayer(
            ch_in=in_channels,
            ch_out=in_channels,
            filter_size=5,
            stride=stride,
            groups=in_channels,
            norm_type=norm_type,
            initializer=XavierUniform
        )
        conv2 = ConvNormLayer(
            ch_in=in_channels,
            ch_out=out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
            initializer=XavierUniform
        )
        conv3 = ConvNormLayer(
            ch_in=out_channels,
            ch_out=out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
            initializer=XavierUniform
        )
        conv4 = ConvNormLayer(
            ch_in=out_channels,
            ch_out=out_channels,
            filter_size=5,
            stride=stride,
            groups=out_channels,
            norm_type=norm_type,
            initializer=XavierUniform
        )

        layers = [
            ('conv1', conv1),
            ('relu6_1', nn.ReLU6()),
            ('conv2', conv2),
        ]
        if with_act:
            layers.append(('relu6_2', nn.ReLU6()))
        layers.extend([
            ('conv3', conv3),
            ('relu6_3', nn.ReLU6()),
            ('conv4', conv4),
        ])
        if with_act:
            layers.append(('relu6_4', nn.ReLU6()))

        self.lite_conv = nn.Sequential()
        for name, layer in layers:
            self.lite_conv.add_module(name, layer)

    def forward(self, inputs):
        out = self.lite_conv(inputs)
        return out


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name=None, data_format='NCHW'):
        """
        DropBlock层初始化函数，参考论文：https://arxiv.org/abs/1810.12890
        Args:
            block_size (int): 块的大小，用于指定DropBlock的尺寸
            keep_prob (float): 保留概率，取值范围在0到1之间
            name (str): 层的名称
            data_format (str): 数据格式，支持'NCHW'或'NHWC'。此处PyTorch操作仅直接处理'NCHW'格式
        Raises:
            ValueError: 当data_format不是'NCHW'或'NHWC'时抛出
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format
        if data_format not in ['NCHW', 'NHWC']:
            raise ValueError("data_format must be 'NCHW' or 'NHWC'")

    def forward(self, x):
        """
        实现空间dropout（Spatial Dropout）的前向传播。
        该函数根据当前模式（训练/评估）和keep_prob参数决定是否执行dropout操作。
        在训练模式下，会按照block_size大小对特征图进行区域级别的随机丢弃，
        并在丢弃后对保留的特征进行缩放以保持期望值不变。
        Args:
            x: 输入特征图，形状为(N, C, H, W)或(N, H, W, C)取决于data_format
        Returns:
            经过dropout处理后的特征图。在评估模式(training=False)或keep_prob=1.0时，
            直接返回输入；在训练模式下返回经过空间dropout和缩放后的特征图。
        """
        if not self.training or self.keep_prob == 1.0:
            return x
        else:
            gamma = (1.0 - self.keep_prob) / (self.block_size ** 2)

            if self.data_format == 'NCHW':
                h, w = x.shape[2], x.shape[3]
            else:
                h, w = x.shape[1], x.shape[2]

            gamma *= (h * w) / ((h - self.block_size + 1) * (w - self.block_size + 1))

            matrix = torch.rand(x.shape, device=x.device, dtype=x.dtype) < gamma
            matrix = matrix.to(x.dtype)

            if self.data_format == 'NCHW':
                mask_inv = F.max_pool2d(
                    matrix,
                    kernel_size=self.block_size,
                    stride=1,
                    padding=self.block_size // 2,
                )
            else:
                matrix_nhwc = matrix.permute(0, 3, 1, 2).contiguous()
                mask_inv_nhwc = F.max_pool2d(
                    matrix_nhwc,
                    kernel_size=self.block_size,
                    stride=1,
                    padding=self.block_size // 2,
                )
                mask_inv = mask_inv_nhwc.permute(0, 2, 3, 1).contiguous()

            mask = 1.0 - mask_inv
            masked_x = x * mask

            total_elements = mask.numel()
            kept_elements = mask.sum()
            scale_factor = total_elements / kept_elements.clamp(min=1e-12)

            y = masked_x * scale_factor
            return y


class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
                 min_ratio=15,
                 max_ratio=90,
                 base_size=300,
                 min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
                 max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
                 offset=0.5,
                 flip=True,
                 clip=False,
                 min_max_aspect_ratios_order=False):
        """
        初始化锚框生成器的配置参数
        该函数用于设置锚框生成器的各项参数，包括不同特征层的步长、宽高比、
        最小/最大尺寸等。当未显式指定min_sizes和max_sizes时，会根据min_ratio和
        max_ratio自动计算这些值。
        Args:
            steps: 每个特征层对应的步长列表，默认[8, 16, 32, 64, 100, 300]
            aspect_ratios: 每个特征层对应的宽高比列表，默认[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]]
            min_ratio: 最小尺寸比例，默认15
            max_ratio: 最大尺寸比例，默认90
            base_size: 基准尺寸，默认300
            min_sizes: 每个特征层对应的最小尺寸列表，默认[30.0, 60.0, 111.0, 162.0, 213.0, 264.0]
            max_sizes: 每个特征层对应的最大尺寸列表，默认[60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
            offset: 锚框中心偏移量，默认0.5
            flip: 是否翻转宽高比，默认True
            clip: 是否裁剪锚框到图像范围内，默认False
            min_max_aspect_ratios_order: 是否按最小/最大宽高比顺序排列，默认False
        Raises:
            ValueError: 当aspect_ratios长度小于2且min_sizes/max_sizes为空时抛出
        """
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order

        if self.min_sizes == [] and self.max_sizes == []:
            num_layer = len(aspect_ratios)
            if num_layer > 1:
                step = int(
                    math.floor((self.max_ratio - self.min_ratio) / (num_layer - 2)))
                for ratio in range(self.min_ratio, self.max_ratio + 1, step):
                    self.min_sizes.append(self.base_size * ratio / 100.)
                    self.max_sizes.append(self.base_size * (ratio + step) / 100.)
                self.min_sizes = [self.base_size * .10] + self.min_sizes
                self.max_sizes = [self.base_size * .20] + self.max_sizes
            else:
                raise ValueError("Automatic min/max size calculation requires at least 2 layers.")

        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(
                aspect_ratios, self.min_sizes, self.max_sizes):
            if isinstance(min_size, (list, tuple)):
                self.num_priors.append(
                    len(_to_list(min_size)) + len(_to_list(max_size)))
            else:
                self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(
                    _to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, inputs, image):
        """
        为特征图列表生成锚框（anchor boxes）。
        Args:
            inputs (list[Tensor]): 特征图列表，例如 [feat1, feat2, ...]。
                                   每个特征图的形状为 [B, C, H, W]。
            image (Tensor): 原始输入图像，形状为 [B, C, H_orig, W_orig]。
        Returns:
            boxes (list[Tensor]): 每个输入特征图对应的锚框列表。
                                  每个锚框张量的形状为 [num_anchors, 4]，
                                  其中4表示 [x_min, y_min, x_max, y_max]。
        """
        boxes = []
        for input_image, min_size, max_size, aspect_ratio, step in zip(
                inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
                self.steps):
            box, _ = ops.prior_box(
                input_tensor=input_image,
                image_tensor=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                steps=[step, step],
                offset=self.offset,
                flip=self.flip,
                clip=self.clip,
                min_max_aspect_ratios_order=self.min_max_aspect_ratios_order
            )
            boxes.append(box.view(-1, 4))
        return boxes


class RCNNBox(object):
    def __init__(self,
                 prior_box_var=[10., 10., 5., 5.],
                 code_type="decode_center_size",
                 box_normalized=False,
                 num_classes=80,
                 export_onnx=False):
        self.prior_box_var = prior_box_var
        self.weights = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.num_classes = num_classes
        self.export_onnx = export_onnx

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred = bbox_head_out[0]
        cls_prob = bbox_head_out[1]
        roi = rois[0]
        rois_num = roi[1]

        if self.export_onnx:
            onnx_rois_num_per_im = rois_num[0].item()
            # Expand im_shape[0] to match the number of RoIs for the single image
            origin_shape = im_shape[0, :].unsqueeze(0).expand(onnx_rois_num_per_im, -1)
        else:
            # Handle dynamic batch size
            origin_shape_list = []
            if isinstance(roi, list):
                batch_size = len(roi)
            else:
                batch_size = im_shape.size(0)

            # bbox_pred.shape: [N, C*4] where N is total RoIs across all images in the batch
            for idx in range(batch_size):
                rois_num_per_im = rois_num[idx].item()
                # Get the corresponding im_shape for this image
                im_h, im_w = im_shape[idx, 0].item(), im_shape[idx, 1].item()
                # Expand this image's shape to match its number of RoIs
                expand_im_shape = torch.tensor([[im_h, im_w]], dtype=im_shape.dtype, device=im_shape.device).expand(
                    rois_num_per_im, -1)  # [rois_num_for_img_i, 2]
                origin_shape_list.append(expand_im_shape)

            # Concatenate shapes for all images in the batch
            origin_shape = torch.cat(origin_shape_list, dim=0)  # [N, 2]

        # bbox_pred.shape: [N, C*4]
        # roi.shape: [N, 4]
        # delta2bbox expects roi [N, 4] and bbox_pred [N, C*4]
        # It should return bbox shape [N, C, 4] where C is the number of classes predicted per ROI (could be 1 for Cascade)
        bbox = torch.concat(roi)
        bbox = delta2bbox(bbox_pred, bbox, weights=self.weights)  # [N, C, 4]

        # Extract scores (excluding background class)
        scores = cls_prob[:, :-1]  # [N, num_classes] # Note: cls_prob should have shape [N, num_classes + 1]
        total_num = bbox.size(0)
        bbox_dim = bbox.size(-1)
        bbox = bbox.expand(total_num, self.num_classes, bbox_dim)

        origin_h = torch.unsqueeze(origin_shape[:, 0], dim=1)
        origin_w = torch.unsqueeze(origin_shape[:, 1], dim=1)

        x1 = torch.clamp(bbox[:, :, 0:1], min=0, max=origin_w)  # [N, num_classes, 1]
        y1 = torch.clamp(bbox[:, :, 1:2], min=0, max=origin_h)  # [N, num_classes, 1]
        x2 = torch.clamp(bbox[:, :, 2:3], min=0, max=origin_w)  # [N, num_classes, 1]
        y2 = torch.clamp(bbox[:, :, 3:4], min=0, max=origin_h)  # [N, num_classes, 1]

        # Stack clipped coordinates back to [N, num_classes, 4]
        bbox = torch.cat([x1, y1, x2, y2], dim=-1)  # [N, num_classes, 4]

        bboxes = (bbox, rois_num)  # bbox: [N, num_classes, 4], rois_num: [B]
        return bboxes, scores  # ( (bbox [N, num_classes, 4], rois_num [B]), scores [N, num_classes] )


def _convert_attention_mask(attn_mask, dtype):
    """
        将注意力掩码转换为目标数据类型。

        参数:
            attn_mask (Tensor, optional): 用于多头注意力的张量，用于防止对某些不想要的位置进行注意力，
                    通常是填充或后续位置。它是一个张量，形状广播到`[batch_size, n_head, sequence_length, sequence_length]`。
                    当数据类型为bool时，不想要的位置有`False`值，其他位置有`True`值。
                    当数据类型为int时，不想要的位置有0值，其他位置有1值。
                    当数据类型为float时，不想要的位置有`-INF`值，其他位置有0值。
                    当不需要或不需要防止注意力时，它可以是None。默认None。
            dtype (torch.dtype): 我们期望的`attn_mask`的目标类型。

        返回:
            Tensor: 一个与输入`attn_mask`形状相同的张量，数据类型为`dtype`。
        """
    if attn_mask is None:
        return attn_mask

    if attn_mask.dtype == dtype:
        return attn_mask

    converted_mask = attn_mask.to(dtype)

    if attn_mask.dtype == torch.bool and dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        # For attention: typically True -> 0.0, False -> -inf
        # But this function just converts dtype, so return as is
        # If specific attention logic needed, add it here
        pass

    return converted_mask


class MultiHeadAttention(nn.Module):
    """
    注意力将查询和一组键值对映射到输出，多头注意力执行多个并行注意力，
    以共同关注来自不同表示子空间的信息。

    请参考 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    获取更多详细信息。

    参数:
        embed_dim (int): 输入和输出中的预期特征大小。
        num_heads (int): 多头注意力中的头数。
        dropout (float, optional): 在注意力权重上使用的dropout概率，用于丢弃一些注意力目标。
            0表示无dropout。默认0
        kdim (int, optional): 键中的特征大小。如果为None，则假定等于`embed_dim`。默认None。
        vdim (int, optional): 值中的特征大小。如果为None，则假定等于`embed_dim`。默认None。
        need_weights (bool, optional): 指示是否返回注意力权重。默认False。

    示例:

        .. code-block:: python

            import torch

            # 编码器输入: [batch_size, sequence_length, d_model]
            query = torch.rand((2, 4, 128))
            # 自注意力掩码: [batch_size, num_heads, query_len, query_len]
            attn_mask = torch.rand((2, 2, 4, 4))
            multi_head_attn = torch.nn.MultiheadAttention(128, 2)
            output, attn_weights = multi_head_attn(query, query, query, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            # 合并QKV的权重矩阵
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.k_proj = nn.Linear(self.kdim, embed_dim, bias=True)
            self.v_proj = nn.Linear(self.vdim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
            constant_(self.in_proj_bias, 0.)
        else:
            xavier_uniform_(self.q_proj.weight)
            xavier_uniform_(self.k_proj.weight)
            xavier_uniform_(self.v_proj.weight)
            constant_(self.q_proj.bias, 0.)
            constant_(self.k_proj.bias, 0.)
            constant_(self.v_proj.bias, 0.)

        xavier_uniform_(self.out_proj.weight)
        constant_(self.out_proj.bias, 0.)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            # 使用单个投影矩阵计算Q/K/V
            weight = self.in_proj_weight[index * self.embed_dim:(index + 1) * self.embed_dim, :]
            bias = self.in_proj_bias[index * self.embed_dim:(index + 1) * self.embed_dim]
            tensor = F.linear(tensor, weight, bias)
        else:
            # 使用单独的投影层
            if index == 0:  # query
                tensor = self.q_proj(tensor)
            elif index == 1:  # key
                tensor = self.k_proj(tensor)
            else:  # value
                tensor = self.v_proj(tensor)

        # 重塑为多头格式: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len, embed_dim = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 转置为: [batch_size, num_heads, seq_len, head_dim]
        tensor = tensor.transpose(1, 2)
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        应用多头注意力将查询和一组键值对映射到输出。

        参数:
            query (Tensor): 多头注意力的查询。它是一个张量，形状为`[batch_size, query_length, embed_dim]`。
                数据类型应为float32或float64。
            key (Tensor, optional): 多头注意力的键。它是一个张量，形状为`[batch_size, key_length, kdim]`。
                数据类型应为float32或float64。如果为None，使用`query`作为`key`。默认None。
            value (Tensor, optional): 多头注意力的值。它是一个张量，形状为`[batch_size, value_length, vdim]`。
                数据类型应为float32或float64。如果为None，使用`query`作为`value`。默认None。
            attn_mask (Tensor, optional): 用于多头注意力的张量，用于防止对某些不想要的位置进行注意力，
                通常是填充或后续位置。它是一个张量，形状广播到`[batch_size, n_head, sequence_length, sequence_length]`。
                当数据类型为bool时，不想要的位置有`False`值，其他位置有`True`值。
                当数据类型为int时，不想要的位置有0值，其他位置有1值。
                当数据类型为float时，不想要的位置有`-INF`值，其他位置有0值。
                当不需要或不需要防止注意力时，它可以是None。默认None。

        返回:
            Tensor|tuple: 它是一个与`query`具有相同形状和数据类型的张量，表示注意力输出。
                或者如果`need_weights`为True则返回元组。如果`need_weights`为True，
                除了注意力输出外，元组还包括形状为`[batch_size, num_heads, query_length, key_length]`的注意力权重张量。
        """
        key = query if key is None else key
        value = query if value is None else value

        # 计算q, k, v
        q = self.compute_qkv(query, 0)  # [batch_size, num_heads, query_length, head_dim]
        k = self.compute_qkv(key, 1)  # [batch_size, num_heads, key_length, head_dim]
        v = self.compute_qkv(value, 2)  # [batch_size, num_heads, value_length, head_dim]

        # 缩放点积注意力
        # [batch_size, num_heads, query_length, key_length]
        product = torch.matmul(q, k.transpose(-2, -1))
        scaling = float(self.head_dim) ** -0.5
        product = product * scaling

        if attn_mask is not None:
            # 支持bool或int掩码
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            # 注意：PyTorch的注意力期望掩码加到logits上，负无穷表示忽略
            product = product + attn_mask

        weights = F.softmax(product, dim=-1)

        if self.dropout > 0:
            weights = F.dropout(weights, self.dropout, training=self.training)

        # [batch_size, num_heads, query_length, head_dim]
        out = torch.matmul(weights, v)

        # 合并头
        # [batch_size, query_length, num_heads, head_dim]
        out = out.transpose(1, 2).contiguous()
        # [batch_size, query_length, embed_dim]
        out = out.view(out.size(0), out.size(1), -1)

        # 投影到输出
        out = self.out_proj(out)

        if self.need_weights:
            return out, weights
        else:
            return out, None