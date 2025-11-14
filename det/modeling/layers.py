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
from torch.nn.init import xavier_uniform_ as XavierUniform


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class AlignConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
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
        Args:
            anchors: [B, L, 5] xc,yc,w,h,angle
            featmap_size: (feat_h, feat_w)
            stride: 8
        Returns:
            offset: [B, 2 * kernel_size * kernel_size, feat_h, feat_w]
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


class DeformableConvV2(nn.Module):  # Changed from nn.Layer to nn.Module
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
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (float): keep probability (between 0 and 1)
            name (str): layer name (not used in PyTorch layer logic)
            data_format (str): data format, 'NCHW' or 'NHWC'. Only 'NCHW' is handled directly by PyTorch ops here.
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format
        if data_format not in ['NCHW', 'NHWC']:
            raise ValueError("data_format must be 'NCHW' or 'NHWC'")

    def forward(self, x):
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
        Generate anchor boxes for a list of feature maps.

        Args:
            inputs (list[Tensor]): List of feature maps, e.g., [feat1, feat2, ...].
                                   Shape of each feat: [B, C, H, W]
            image (Tensor): Original input image. Shape: [B, C, H_orig, W_orig]

        Returns:
            boxes (list[Tensor]): List of anchor boxes for each input feature map.
                                  Shape of each box tensor: [num_anchors, 4],
                                  where 4 is [x_min, y_min, x_max, y_max].
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
                expand_im_shape = torch.tensor([[im_h, im_w]], dtype=im_shape.dtype, device=im_shape.device).expand(rois_num_per_im, -1) # [rois_num_for_img_i, 2]
                origin_shape_list.append(expand_im_shape)

            # Concatenate shapes for all images in the batch
            origin_shape = torch.cat(origin_shape_list, dim=0) # [N, 2]

        # bbox_pred.shape: [N, C*4]
        # roi.shape: [N, 4]
        # delta2bbox expects roi [N, 4] and bbox_pred [N, C*4]
        # It should return bbox shape [N, C, 4] where C is the number of classes predicted per ROI (could be 1 for Cascade)
        bbox = torch.concat(roi)
        bbox = delta2bbox(bbox_pred, bbox, weights=self.weights) # [N, C, 4]

        # Extract scores (excluding background class)
        scores = cls_prob[:, :-1] # [N, num_classes] # Note: cls_prob should have shape [N, num_classes + 1]
        total_num = bbox.size(0)
        bbox_dim = bbox.size(-1)
        bbox = bbox.expand(total_num, self.num_classes, bbox_dim)

        origin_h = torch.unsqueeze(origin_shape[:, 0], dim=1)
        origin_w = torch.unsqueeze(origin_shape[:, 1], dim=1)

        x1 = torch.clamp(bbox[:, :, 0:1], min=0, max=origin_w) # [N, num_classes, 1]
        y1 = torch.clamp(bbox[:, :, 1:2], min=0, max=origin_h) # [N, num_classes, 1]
        x2 = torch.clamp(bbox[:, :, 2:3], min=0, max=origin_w) # [N, num_classes, 1]
        y2 = torch.clamp(bbox[:, :, 3:4], min=0, max=origin_h) # [N, num_classes, 1]

        # Stack clipped coordinates back to [N, num_classes, 4]
        bbox = torch.cat([x1, y1, x2, y2], dim=-1) # [N, num_classes, 4]

        bboxes = (bbox, rois_num) # bbox: [N, num_classes, 4], rois_num: [B]
        return bboxes, scores # ( (bbox [N, num_classes, 4], rois_num [B]), scores [N, num_classes] )