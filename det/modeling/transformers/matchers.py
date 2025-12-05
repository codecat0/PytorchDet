#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :matchers.py
@Author :CodeCat
@Date   :2025/11/14 17:27
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from det.modeling.losses.iou_loss import GIoULoss
from det.modeling.transformers.utils import bbox_cxcywh_to_xyxy


__all__ = ['HungarianMatcher']


class HungarianMatcher(nn.Module):
    def __init__(self,
                 matcher_coeff=
                 {
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 1,
                     'dice': 1
                 },
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        初始化匈牙利匹配器

        参数:
            matcher_coeff (dict): 匈牙利匹配器成本的系数
                - 'class': 分类成本系数
                - 'bbox': 边界框L1损失成本系数
                - 'giou': GIoU损失成本系数
                - 'mask': 掩码BCE成本系数（如果with_mask=True）
                - 'dice': Dice损失成本系数（如果with_mask=True）
            use_focal_loss (bool): 是否使用focal loss进行分类
            with_mask (bool): 是否包含掩码匹配
            num_sample_points (int): 采样点数量，用于掩码匹配
            alpha (float): focal loss的alpha参数
            gamma (float): focal loss的gamma参数
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None):
        r"""
        执行匈牙利匹配

        参数:
            boxes (Tensor): 预测边界框，形状为 [batch_size, num_queries, 4]，格式为[cx, cy, w, h]
            logits (Tensor): 分类logits，形状为 [batch_size, num_queries, num_classes]
            gt_bbox (List[Tensor]): 每个样本的ground truth边界框列表，每个张量形状为[n, 4]
            gt_class (List[Tensor]): 每个样本的ground truth类别列表，每个张量形状为[n, 1]
            masks (Tensor|None): 预测掩码，形状为 [batch_size, num_queries, h, w]
            gt_mask (List[Tensor]): 每个样本的ground truth掩码列表，每个张量形状为[n, H, W]

        返回:
            一个包含(batch_size)个元组的列表，每个元组包含(index_i, index_j)：
                - index_i: 选中的预测索引（按顺序）
                - index_j: 对应选中的目标索引（按顺序）
            对于每个批次元素，满足：
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        # 计算每个样本中的ground truth数量
        num_gts = [len(a) for a in gt_class]

        # 如果没有ground truth，返回空索引
        if sum(num_gts) == 0:
            return [(torch.tensor([], dtype=torch.long, device=boxes.device),
                     torch.tensor([], dtype=torch.long, device=boxes.device))
                    for _ in range(bs)]

        # 展平张量以批量计算成本矩阵
        # [batch_size * num_queries, num_classes]
        logits = logits.detach()
        out_prob = F.sigmoid(logits.flatten(0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1), dim=-1)
        # [batch_size * num_queries, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        # 连接所有样本的ground truth标签和边界框
        tgt_ids = torch.cat(gt_class).flatten()  # [total_num_gts]
        tgt_bbox = torch.cat(gt_bbox)  # [total_num_gts, 4]

        # 计算分类成本
        # 选择对应类别的概率
        tgt_ids = tgt_ids.long()
        out_prob = out_prob.gather(1, tgt_ids.unsqueeze(1)).squeeze(1)  # [total_num_queries * total_num_gts]

        if self.use_focal_loss:
            # Focal Loss的成本计算
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            # 标准交叉熵损失的成本计算
            cost_class = -out_prob

        # 计算边界框L1成本
        # [batch_size * num_queries, total_num_gts, 4] -> [batch_size * num_queries, total_num_gts]
        cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # 计算GIoU成本
        # 需要将中心点格式转换为xyxy格式
        out_bbox_xyxy = bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1))  # [total_queries, 1, 4]
        tgt_bbox_xyxy = bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))  # [1, total_gts, 4]
        giou_loss = self.giou_loss(out_bbox_xyxy, tgt_bbox_xyxy).squeeze(-1)  # [total_queries, total_gts]
        cost_giou = giou_loss - 1  # GIoU在[0,1]之间，所以减1使其在[-1,0]之间

        # 计算总成本矩阵
        C = (self.matcher_coeff['class'] * cost_class +
             self.matcher_coeff['bbox'] * cost_bbox +
             self.matcher_coeff['giou'] * cost_giou)

        # 计算掩码成本和Dice成本（如果启用）
        if self.with_mask:
            assert (masks is not None and gt_mask is not None), 'Make sure the input has `mask` and `gt_mask`'

            # 所有掩码共享相同的采样点以提高效率
            sample_points = torch.rand([bs, 1, self.num_sample_points, 2], device=masks.device)
            sample_points = 2.0 * sample_points - 1.0  # 将范围从[0,1]转换到[-1,1]用于grid_sample

            # 采样预测掩码
            out_mask = F.grid_sample(
                masks.detach(), sample_points, align_corners=False).squeeze(
                -2)  # [batch_size, num_queries, num_sample_points]
            out_mask = out_mask.flatten(0, 1)  # [total_queries, num_sample_points]

            # 采样ground truth掩码
            tgt_mask = torch.cat(gt_mask).unsqueeze(1)  # [total_gts, 1, H, W]

            # 为每个ground truth样本创建对应的采样点
            sample_points_expanded = []
            for i, num_gt in enumerate(num_gts):
                if num_gt > 0:
                    sample_points_expanded.append(sample_points[i].expand(num_gt, -1, -1, -1))
            sample_points_expanded = torch.cat(sample_points_expanded, dim=0)  # [total_gts, 1, num_sample_points, 2]

            # 采样ground truth掩码
            tgt_mask = F.grid_sample(
                tgt_mask, sample_points_expanded, align_corners=False).squeeze([1, 2])  # [total_gts, num_sample_points]

            # 二元交叉熵成本
            pos_cost_mask = F.binary_cross_entropy_with_logits(
                out_mask, torch.ones_like(out_mask), reduction='none')
            neg_cost_mask = F.binary_cross_entropy_with_logits(
                out_mask, torch.zeros_like(out_mask), reduction='none')
            cost_mask = (torch.matmul(pos_cost_mask, tgt_mask.t()) +
                         torch.matmul(neg_cost_mask, (1 - tgt_mask).t()))
            cost_mask /= self.num_sample_points  # 归一化

            # Dice成本
            out_mask_sigmoid = torch.sigmoid(out_mask)  # [total_queries, num_sample_points]
            numerator = 2 * torch.matmul(out_mask_sigmoid, tgt_mask.t())  # [total_queries, total_gts]
            denominator = out_mask_sigmoid.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(
                0)  # [total_queries, total_gts]
            cost_dice = 1 - (numerator + 1) / (denominator + 1)  # 加1防止除零

            # 将掩码成本添加到总成本中
            C = C + self.matcher_coeff['mask'] * cost_mask + self.matcher_coeff['dice'] * cost_dice

        C = C.view(bs, num_queries, -1)  # reshape
        C_list = C.chunk(bs, dim=0)      # split into list of [1, num_queries, total_gts]
        sizes = [a.shape[0] for a in gt_bbox]  # number of gts per image

        # Split C per image and run Hungarian matching
        indices = []
        for i, c in enumerate(C_list):
            # c: [1, num_queries, total_gts] → squeeze to [num_queries, total_gts]
            c = c.squeeze(0)  # [num_queries, total_gts]
            
            # Split cost matrix along gt dimension: [num_queries, s] for s in sizes
            # But we only need the i-th image's gt part: first `sizes[i]` columns
            c_i = c[:, :sizes[i]]  # shape: [num_queries, sizes[i]]
            
            # Convert to numpy for linear_sum_assignment
            c_i_np = c_i.detach().cpu().numpy()
            
            # Solve assignment problem
            row_idx, col_idx = linear_sum_assignment(c_i_np)
            indices.append((row_idx, col_idx))

        return [(torch.from_numpy(i).long().to(logits.device), torch.from_numpy(j).long().to(logits.device)) for i, j in indices]
