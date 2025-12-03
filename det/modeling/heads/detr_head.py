#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :detr_head.py
@Author :CodeCat
@Date   :2025/11/14 14:45
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pycocotools.mask as mask_util
from torch.nn.init import constant_
from det.modeling.initializer import linear_init_
from det.modeling.transformers.utils import inverse_sigmoid


__all__ = ['DETRHead']


class MLP(nn.Module):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiHeadAttentionMap(nn.Module):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
    """

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0,
                 bias=True):
        """
        初始化多头注意力映射模块

        参数:
            query_dim: 查询向量的维度
            hidden_dim: 隐藏层维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            dropout: dropout概率
            bias: 是否使用偏置
        """
        super(MultiHeadAttentionMap, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 查询投影层：将查询向量投影到隐藏空间
        self.q_proj = nn.Linear(query_dim, hidden_dim, bias=bias)
        # 键投影层：将键特征图投影到隐藏空间（使用1x1卷积）
        self.k_proj = nn.Conv2d(
            query_dim,
            hidden_dim,
            1,
            bias=bias)

        # 注意力分数的缩放因子（用于缩放点积，防止梯度消失）
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        """
        前向传播

        参数:
            q: 查询张量，形状为 [batch_size, num_queries, query_dim]
            k: 键张量，形状为 [batch_size, query_dim, height, width]
            mask: 可选的注意力掩码，形状为 [batch_size, num_queries, num_heads, height, width]
        返回:
            weights: 注意力权重，形状为 [batch_size, num_queries, num_heads, height, width]
        """
        # 投影查询向量到隐藏空间
        q = self.q_proj(q)  # [bs, num_queries, hidden_dim]
        # 投影键特征图到隐藏空间
        k = self.k_proj(k)  # [bs, hidden_dim, h, w]

        # 获取维度信息
        bs, num_queries, n, c, h, w = q.shape[0], q.shape[1], self.num_heads, \
                                      self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]

        # 重塑查询和键张量以支持多头注意力
        qh = q.reshape([bs, num_queries, n, c])  # [bs, num_queries, n, c] - 每个头的查询
        kh = k.reshape([bs, n, c, h, w])  # [bs, n, c, h, w] - 每个头的键

        # 重塑并计算注意力权重
        # 将查询重塑为 [bs*n, num_queries, c] 以便批量矩阵乘法
        qh = qh.permute([0, 2, 1, 3]).reshape([-1, num_queries, c])  # [bs*n, num_queries, c]
        # 将键重塑为 [bs*n, c, h*w] 以便批量矩阵乘法
        kh = kh.reshape([-1, c, h * w])  # [bs*n, c, h*w]

        # 计算注意力分数: [bs*n, num_queries, h*w]
        # qh * normalize_fact: 缩放查询向量
        # torch.bmm: 批量矩阵乘法，得到 [bs*n, num_queries, h*w] 的注意力分数
        weights = torch.bmm(qh * self.normalize_fact, kh)  # [bs*n, num_queries, h*w]

        # 重塑回原始形状并调整维度顺序
        weights = weights.reshape([bs, n, num_queries, h, w]).permute([0, 2, 1, 3, 4])
        # [bs, num_queries, n, h, w] - [batch_size, num_queries, num_heads, height, width]

        # 应用掩码（如果提供）
        if mask is not None:
            weights += mask

        # fix a potenial bug: https://github.com/facebookresearch/detr/issues/247
        # 应用softmax函数。flatten(3) 将 [bs, num_queries, n, h, w] 变为 [bs, num_queries, n*h*w]
        # 然后在最后一个维度（h*w）上应用softmax，确保每个头内的空间位置权重和为1
        weights = F.softmax(weights.flatten(3), dim=-1).reshape(weights.shape)

        # 应用dropout
        weights = self.dropout(weights)

        return weights


class MaskHeadFPNConv(nn.Module):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        Simple convolutional head, using group norm.
        Upsampling is done using a FPN approach
    """

    def __init__(self, input_dim, fpn_dims, context_dim, num_groups=8):
        super(MaskHeadFPNConv, self).__init__()

        # 计算中间维度：从input_dim开始，逐步减半直到context_dim/16
        inter_dims = [input_dim] + [context_dim // (2 ** i) for i in range(1, 5)]

        self.conv0 = self._make_layers(input_dim * 2, input_dim, 3, num_groups)  # 输入维度翻倍（concat了bbox_attention_map）
        self.conv_inter = nn.ModuleList()
        for in_dims, out_dims in zip(inter_dims[:-1], inter_dims[1:]):
            self.conv_inter.append(
                self._make_layers(in_dims, out_dims, 3, num_groups))

        self.conv_out = nn.Conv2d(
            inter_dims[-1],
            1,
            3,
            padding=1)

        self.adapter = nn.ModuleList()
        for i in range(len(fpn_dims)):
            self.adapter.append(
                nn.Conv2d(
                    fpn_dims[i],
                    inter_dims[i + 1],
                    1))

    @staticmethod
    def _make_layers(in_dims,
                     out_dims,
                     kernel_size,
                     num_groups):
        """
        创建卷积层、组归一化和ReLU的序列

        参数:
            in_dims: 输入维度
            out_dims: 输出维度
            kernel_size: 卷积核大小
            num_groups: 组归一化的组数
        返回:
            nn.Sequential: 包含卷积、组归一化和ReLU的序列
        """
        return nn.Sequential(
            nn.Conv2d(
                in_dims,
                out_dims,
                kernel_size,
                padding=kernel_size // 2),
            nn.GroupNorm(num_groups, out_dims),
            nn.ReLU())

    def forward(self, x, bbox_attention_map, fpns):
        """
        前向传播

        参数:
            x: 输入特征 [batch_size, channels, height, width]
            bbox_attention_map: 边界框注意力图 [batch_size, num_queries, num_heads, height, width]
            fpns: FPN特征列表，包含多个不同层级的特征图
        返回:
            x: 输出分割掩码 [batch_size * num_queries, 1, height, width]
        """
        # 将x重复num_queries次，并与bbox_attention_map拼接
        # x: [batch_size, channels, height, width] -> [batch_size * num_queries, channels, height, width]
        # bbox_attention_map: [batch_size, num_queries, num_heads, height, width] -> [batch_size * num_queries, num_heads, height, width]
        x = torch.cat([
            x.unsqueeze(1).expand(-1, bbox_attention_map.shape[1], -1, -1, -1).flatten(0, 1),  # 重复x
            bbox_attention_map.flatten(0, 1)  # 展平bbox_attention_map
        ], dim=1)  # 拼接后维度为 [batch_size * num_queries, channels + num_heads, height, width]

        x = self.conv0(x)  # 初始卷积

        # 通过中间卷积层和FPN适配层进行特征融合
        for inter_layer, adapter_layer, feat in zip(self.conv_inter[:-1], self.adapter, fpns):
            # 适配FPN特征维度并重复num_queries次
            feat = adapter_layer(feat).unsqueeze(1).expand(
                -1, bbox_attention_map.shape[1], -1, -1, -1).flatten(0, 1)

            x = inter_layer(x)  # 中间卷积层

            # FPN特征与当前特征相加（通过插值调整大小以匹配）
            x = feat + F.interpolate(x, size=feat.shape[-2:])

        x = self.conv_inter[-1](x)  # 最后一个中间卷积层
        x = self.conv_out(x)  # 输出卷积层，生成最终的分割掩码
        return x


class DETRHead(nn.Module):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 nhead=8,
                 num_mlp_layers=3,
                 loss=None,
                 fpn_dims=[1024, 512, 256],
                 with_mask_head=False,
                 use_focal_loss=False):
        super(DETRHead, self).__init__()
        # 添加背景类
        self.num_classes = num_classes if use_focal_loss else num_classes + 1
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.with_mask_head = with_mask_head
        self.use_focal_loss = use_focal_loss

        # 分类头：将隐藏特征映射到类别得分
        self.score_head = nn.Linear(hidden_dim, self.num_classes)
        # 边界框回归头：将隐藏特征映射到4个坐标值
        self.bbox_head = MLP(hidden_dim,
                             hidden_dim,
                             output_dim=4,
                             num_layers=num_mlp_layers)

        if self.with_mask_head:
            # 边界框注意力模块：计算边界框与特征图的空间注意力
            self.bbox_attention = MultiHeadAttentionMap(hidden_dim, hidden_dim,
                                                        nhead)
            # 掩码头：用于实例分割
            self.mask_head = MaskHeadFPNConv(hidden_dim + nhead, fpn_dims,
                                             hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """重置参数"""
        linear_init_(self.score_head)

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):
        """
        从配置创建DETRHead实例

        参数:
            cfg: 配置对象
            hidden_dim: 隐藏层维度
            nhead: 注意力头数
            input_shape: 输入特征图的形状
        返回:
            包含初始化参数的字典
        """
        return {
            'hidden_dim': hidden_dim,
            'nhead': nhead,
            'fpn_dims': [i.channels for i in input_shape[::-1]][1:]  # 反向并跳过第一个
        }

    @staticmethod
    def get_gt_mask_from_polygons(gt_poly, pad_mask):
        """
        从多边形注释和填充掩码生成Ground Truth掩码张量列表。

        参数:
            gt_poly (List[List[List[float]]]): 每个样本的多边形列表，其中每个对象有多个多边形。
                                               格式: [batch, num_objects, [x1,y1,x2,y2,...]]
            pad_mask (Tensor): 填充掩码，形状为 [batch, padded_height, padded_width]。
                              1表示有效像素，0表示填充像素。

        返回:
            List[Tensor]: 每个样本的Ground Truth掩码张量列表，形状为 [num_objects, padded_height, padded_width]。
        """
        out_gt_mask = []
        for polygons, padding in zip(gt_poly, pad_mask):
            # 计算实际的高度和宽度（通过统计非填充区域）
            height = int(padding[:, 0].sum().item())  # .item() 确保得到Python标量
            width = int(padding[0, :].sum().item())

            masks = []
            for obj_poly in polygons:
                # 使用pycocotools将多边形转换为RLE编码，然后解码为掩码
                rles = mask_util.frPyObjects(obj_poly, height, width)
                rle = mask_util.merge(rles)  # 合并多个多边形（如果一个对象由多个多边形组成）
                mask_np = mask_util.decode(rle)  # 解码为numpy数组 [H, W]
                mask_tensor = torch.from_numpy(mask_np).float()  # 转换为torch张量，数据类型为float32
                masks.append(mask_tensor)

            # 将当前样本的所有对象掩码堆叠起来
            # 形状: [num_objects_in_this_sample, actual_height, actual_width]
            masks = torch.stack(masks)

            # 创建一个填充后的掩码张量，形状与pad_mask一致
            # 形状: [num_objects_in_this_sample, padded_height, padded_width]
            masks_pad = torch.zeros(
                (masks.shape[0], pad_mask.shape[1], pad_mask.shape[2]),
                dtype=torch.float32,
                device=pad_mask.device  # 确保在相同设备上
            )

            # 将实际的掩码区域复制到填充张量中
            masks_pad[:, :height, :width] = masks

            out_gt_mask.append(masks_pad)

        return out_gt_mask

    def forward(self, out_transformer, body_feats, inputs=None):
        """
        前向传播

        参数:
            out_transformer (Tuple):
                (feats: [num_levels, batch_size, num_queries, hidden_dim],
                 memory: [batch_size, hidden_dim, h, w],
                 src_proj: [batch_size, hidden_dim, h, w],
                 src_mask: [batch_size, 1, 1, h, w])
            body_feats (List[Tensor]): FPN特征列表 [[B, C, H, W]]
            inputs (dict, optional): 训练时的输入字典，包含ground truth信息
        返回:
            训练时返回损失字典，推理时返回预测结果元组
        """
        feats, memory, src_proj, src_mask = out_transformer

        # 计算分类得分
        outputs_logit = self.score_head(feats)  # [num_levels, batch_size, num_queries, num_classes]
        # 计算边界框坐标（使用sigmoid激活以确保在[0,1]范围内）
        outputs_bbox = F.sigmoid(self.bbox_head(feats))  # [num_levels, batch_size, num_queries, 4]

        outputs_seg = None
        if self.with_mask_head:
            # 计算边界框注意力图
            # feats[-1]: [batch_size, num_queries, hidden_dim]
            # memory: [batch_size, hidden_dim, h, w]
            # src_mask: [batch_size, 1, 1, h, w] 或 None
            bbox_attention_map = self.bbox_attention(feats[-1], memory,
                                                     src_mask)  # [batch_size, num_queries, nhead, h, w]

            # 获取FPN特征（反向并跳过第一个）
            fpn_feats = [a for a in body_feats[::-1]][1:]

            # 计算分割掩码
            # src_proj: [batch_size, h*w, hidden_dim]
            # bbox_attention_map: [batch_size, num_queries, nhead, h, w]
            # fpn_feats: List[Tensor]
            outputs_seg = self.mask_head(src_proj, bbox_attention_map,
                                         fpn_feats)  # [batch_size * num_queries, 1, H, W]

            # 重塑分割输出为 [batch_size, num_queries, H, W]
            outputs_seg = outputs_seg.reshape([
                feats.shape[1], feats.shape[2], outputs_seg.shape[-2],
                outputs_seg.shape[-1]
            ])

        if self.training:
            # 训练模式：计算损失
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            gt_mask = self.get_gt_mask_from_polygons(
                inputs['gt_poly'],
                inputs['pad_mask']) if 'gt_poly' in inputs else None
            return self.loss(
                outputs_bbox,
                outputs_logit,
                inputs['gt_bbox'],
                inputs['gt_class'],
                masks=outputs_seg,
                gt_mask=gt_mask)
        else:
            # 推理模式：返回预测结果
            # 只返回最后一层的预测结果
            return (outputs_bbox[-1], outputs_logit[-1], outputs_seg)


if __name__ == '__main__':
    from det.modeling.losses.detr_loss import DETRLoss
    from det.modeling.transformers.matchers import HungarianMatcher

    matcher = HungarianMatcher(
        matcher_coeff={
            'class': 1,
            'bbox': 5,
            'giou': 2
        }
    )
    loss = DETRLoss(
        num_classes=1,
        matcher=matcher,
        loss_coeff={
            'class': 1,
            'bbox': 5,
            'giou': 2,
            'no_object': 0.1
        },
        aux_loss=True,
    )
    model = DETRHead(
        num_classes=1,
        loss=loss,
        num_mlp_layers=3,
        hidden_dim=256,
    )
    out_transformer = (
        torch.randn((6, 1, 100, 256)),
        torch.randn((1, 256, 25, 25)),
        torch.randn((1, 256, 25, 25)),
        torch.randn((1, 1, 1, 25, 25))
    )
    body_feats = [torch.randn((1, 2048, 25, 25))]
    model.eval()
    outs = model(out_transformer, body_feats)
    for out in outs:
        print(out.shape)
