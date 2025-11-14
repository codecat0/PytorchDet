#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :iou_loss.py
@Author :CodeCat
@Date   :2025/11/14 15:30
"""
import numpy as np
import math
import torch

from det.modeling.bbox_utils import bbox_iou


class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630

    参数:
        loss_weight (float): giou损失权重，默认为1
        eps (float): 防止除零的epsilon值，默认为1e-10
        reduction (string): 可选项为"none", "mean" 和 "sum"。默认为"none"
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    @staticmethod
    def bbox_overlap(box1, box2, eps=1e-10):
        """
        计算box1和box2的交并比

        参数:
            box1 (Tensor): 形状为(..., 4)的box1，格式为[x1, y1, x2, y2]
            box2 (Tensor): 形状为(..., 4)的box2，格式为[x1, y1, x2, y2]
            eps (float): 防止除零的epsilon值
        返回:
            iou (Tensor): box1和box2的交并比
            overlap (Tensor): box1和box2的重叠面积
            union (Tensor): box1和box2的并集面积
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        # 计算交集区域的坐标
        xkis1 = torch.maximum(x1, x1g)  # 交集左上角x坐标
        ykis1 = torch.maximum(y1, y1g)  # 交集左上角y坐标
        xkis2 = torch.minimum(x2, x2g)  # 交集右下角x坐标
        ykis2 = torch.minimum(y2, y2g)  # 交集右下角y坐标

        # 计算交集宽度和高度，确保不为负数
        w_inter = torch.clamp(xkis2 - xkis1, min=0)  # 交集宽度
        h_inter = torch.clamp(ykis2 - ykis1, min=0)  # 交集高度
        overlap = w_inter * h_inter  # 交集面积

        # 计算各自面积
        area1 = (x2 - x1) * (y2 - y1)  # box1面积
        area2 = (x2g - x1g) * (y2g - y1g)  # box2面积
        # 计算并集面积，加上eps防止除零
        union = area1 + area2 - overlap + eps
        iou = overlap / union  # 交并比

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        """
        计算GIoU损失

        参数:
            pbox (Tensor): 预测框，形状为[N, 4]，格式为[x1, y1, x2, y2]
            gbox (Tensor): 真实框（ground truth），形状为[N, 4]，格式为[x1, y1, x2, y2]
            iou_weight (float): iou权重，默认为1
            loc_reweight (Tensor, optional): 位置重权重，形状为[N,]
        返回:
            loss (Tensor): GIoU损失值
        """
        # 分割预测框和真实框的坐标
        x1, y1, x2, y2 = torch.split(pbox, 1, dim=-1)  # [N, 1] 每个坐标
        x1g, y1g, x2g, y2g = torch.split(gbox, 1, dim=-1)  # [N, 1] 每个坐标

        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]

        # 计算IoU、重叠面积和并集面积
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)

        # 计算包含两个框的最小闭包区域的坐标
        xc1 = torch.minimum(x1, x1g)  # 闭包区域左上角x坐标
        yc1 = torch.minimum(y1, y1g)  # 闭包区域左上角y坐标
        xc2 = torch.maximum(x2, x2g)  # 闭包区域右下角x坐标
        yc2 = torch.maximum(y2, y2g)  # 闭包区域右下角y坐标

        # 计算闭包区域面积，加上eps防止除零
        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps

        # 计算GIoU: IoU - (C\U)/C，其中C是闭包区域，U是并集
        miou = iou - ((area_c - union) / area_c)

        if loc_reweight is not None:
            # 如果提供了位置重权重，应用加权
            loc_reweight = loc_reweight.reshape(-1, 1)  # [N, 1]
            loc_thresh = 0.9
            # 加权GIoU计算
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            # 标准GIoU计算: 1 - GIoU
            giou = 1 - miou

        # 根据reduction模式计算最终损失
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:  # 'mean'
            loss = torch.mean(giou * iou_weight)

        # 应用损失权重并返回
        return loss * self.loss_weight


class DIouLoss(object):
    """
    Distance-IoU Loss, see https://arxiv.org/abs/1911.08287

    参数:
        loss_weight (float): DIoU损失权重，默认为1
        eps (float): 防止除零的epsilon值，默认为1e-10
        use_complete_ciou_loss (bool): 是否使用CIoU损失
    """

    def __init__(self, loss_weight=1., eps=1e-10, use_complete_ciou_loss=True):
        self.loss_weight = loss_weight
        self.eps = eps
        self.use_complete_iou_loss = use_complete_ciou_loss

    def forward(self, pbox, gbox, iou_weight=1.):
        """
        计算DIoU/CIoU损失

        参数:
            pbox (Tensor): 预测框，形状为[N, 4]，格式为[x1, y1, x2, y2]
            gbox (Tensor): 真实框（ground truth），形状为[N, 4]，格式为[x1, y1, x2, y2]
            iou_weight (float): IoU权重，默认为1
        返回:
            loss (Tensor): DIoU或CIoU损失值
        """
        # 分割预测框和真实框的坐标
        x1, y1, x2, y2 = torch.split(pbox, 1, dim=-1)  # [N, 1] 每个坐标
        x1g, y1g, x2g, y2g = torch.split(gbox, 1, dim=-1)  # [N, 1] 每个坐标

        # 计算预测框的中心点、宽度和高度
        cx = (x1 + x2) / 2  # 预测框中心x坐标
        cy = (y1 + y2) / 2  # 预测框中心y坐标
        w = x2 - x1  # 预测框宽度
        h = y2 - y1  # 预测框高度

        # 计算真实框的中心点、宽度和高度
        cxg = (x1g + x2g) / 2  # 真实框中心x坐标
        cyg = (y1g + y2g) / 2  # 真实框中心y坐标
        wg = x2g - x1g  # 真实框宽度
        hg = y2g - y1g  # 真实框高度

        # 确保坐标顺序正确（虽然通常输入已经是正确的）
        x2 = torch.maximum(x1, x2)
        y2 = torch.maximum(y1, y2)

        # 计算交集区域坐标
        xkis1 = torch.maximum(x1, x1g)  # 交集左上角x坐标
        ykis1 = torch.maximum(y1, y1g)  # 交集左上角y坐标
        xkis2 = torch.minimum(x2, x2g)  # 交集右下角x坐标
        ykis2 = torch.minimum(y2, y2g)  # 交集右下角y坐标

        # 计算包含两个框的最小闭包区域坐标
        xc1 = torch.minimum(x1, x1g)  # 闭包区域左上角x坐标
        yc1 = torch.minimum(y1, y1g)  # 闭包区域左上角y坐标
        xc2 = torch.maximum(x2, x2g)  # 闭包区域右下角x坐标
        yc2 = torch.maximum(y2, y2g)  # 闭包区域右下角y坐标

        # 计算交集面积
        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        # 使用掩码确保交集面积不为负数（当两框不相交时为0）
        intsct_mask = (xkis2 > xkis1) & (ykis2 > ykis1)
        intsctk = intsctk * intsct_mask.float()

        # 计算并集面积
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + self.eps
        # 计算IoU
        iouk = intsctk / unionk

        # 计算DIoU项：中心点距离惩罚项
        # 中心点之间的欧氏距离的平方
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        # 闭包区域对角线长度的平方
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        # DIoU惩罚项：(中心点距离^2) / (闭包区域对角线^2)
        diou_term = (dist_intersection + self.eps) / (dist_union + self.eps)

        # 计算CIoU项（如果启用）
        ciou_term = 0
        if self.use_complete_ciou_loss:
            # 计算宽高比
            ar_gt = wg / hg  # 真实框宽高比
            ar_pred = w / h  # 预测框宽高比

            # 计算宽高比差异的惩罚项
            # 使用反正切函数来衡量宽高比差异
            arctan = torch.atan(ar_gt) - torch.atan(ar_pred)
            # 宽高比差异惩罚项
            ar_loss = 4. / np.pi / np.pi * arctan * arctan

            # CIoU中的alpha参数，用于平衡宽高比损失和IoU损失
            alpha = ar_loss / (1 - iouk + ar_loss + self.eps)
            # alpha在反向传播中不计算梯度
            alpha = alpha.detach()
            # CIoU惩罚项
            ciou_term = alpha * ar_loss

        # 计算最终的DIoU/CIoU损失
        # 损失 = 1 - IoU + DIoU项 + CIoU项
        diou = torch.mean((1 - iouk + ciou_term + diou_term) * iou_weight)

        return diou * self.loss_weight


class SIoULoss(object):
    """
    see https://arxiv.org/pdf/2205.12740.pdf

    参数:
        loss_weight (float): siou损失权重，默认为1
        eps (float): 防止除零的epsilon值，默认为1e-10
        theta (float): 形状成本的指数参数，默认为4
        reduction (str): 可选项为"none", "mean" 和 "sum"。默认为"none"
    """

    def __init__(self, loss_weight=1., eps=1e-10, theta=4., reduction='none'):
        super(SIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.theta = theta
        self.reduction = reduction

    @staticmethod
    def bbox_iou(box1, box2, eps=1e-10):
        """
        计算两个边界框的IoU

        参数:
            box1 (list): [x1, y1, x2, y2] 形式的第一个边界框
            box2 (list): [x1, y1, x2, y2] 形式的第二个边界框
            eps (float): 防止除零的epsilon值
        返回:
            iou (Tensor): 两个边界框的IoU值
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        # 计算交集区域坐标
        xkis1 = torch.maximum(x1, x1g)  # 交集左上角x坐标
        ykis1 = torch.maximum(y1, y1g)  # 交集左上角y坐标
        xkis2 = torch.minimum(x2, x2g)  # 交集右下角x坐标
        ykis2 = torch.minimum(y2, y2g)  # 交集右下角y坐标

        # 计算交集面积，确保不为负数
        w_inter = torch.clamp(xkis2 - xkis1, min=0)  # 交集宽度
        h_inter = torch.clamp(ykis2 - ykis1, min=0)  # 交集高度
        overlap = w_inter * h_inter  # 交集面积

        # 计算各自面积
        area1 = (x2 - x1) * (y2 - y1)  # box1面积
        area2 = (x2g - x1g) * (y2g - y1g)  # box2面积
        # 计算并集面积
        union = area1 + area2 - overlap + eps
        iou = overlap / union  # 交并比

        return iou

    def __call__(self, pbox, gbox):
        """
        计算SIoU损失

        参数:
            pbox (Tensor): 预测框，形状为[N, 4]，格式为[x1, y1, x2, y2]
            gbox (Tensor): 真实框（ground truth），形状为[N, 4]，格式为[x1, y1, x2, y2]
        返回:
            loss (Tensor): SIoU损失值
        """
        # 分割预测框和真实框的坐标
        x1, y1, x2, y2 = torch.split(pbox, 1, dim=-1)  # [N, 1] 每个坐标
        x1g, y1g, x2g, y2g = torch.split(gbox, 1, dim=-1)  # [N, 1] 每个坐标

        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        # 计算IoU
        iou = self.bbox_iou(box1, box2)

        # 计算预测框的中心点、宽度和高度
        cx = (x1 + x2) / 2  # 预测框中心x坐标
        cy = (y1 + y2) / 2  # 预测框中心y坐标
        w = x2 - x1 + self.eps  # 预测框宽度（加eps防止为0）
        h = y2 - y1 + self.eps  # 预测框高度（加eps防止为0）

        # 计算真实框的中心点、宽度和高度
        cxg = (x1g + x2g) / 2  # 真实框中心x坐标
        cyg = (y1g + y2g) / 2  # 真实框中心y坐标
        wg = x2g - x1g + self.eps  # 真实框宽度（加eps防止为0）
        hg = y2g - y1g + self.eps  # 真实框高度（加eps防止为0）

        # 确保坐标顺序正确（虽然通常输入已经是正确的）
        x2 = torch.maximum(x1, x2)
        y2 = torch.maximum(y1, y2)

        # 计算包含两个框的最小闭包区域坐标
        xc1 = torch.minimum(x1, x1g)  # 闭包区域左上角x坐标
        yc1 = torch.minimum(y1, y1g)  # 闭包区域左上角y坐标
        xc2 = torch.maximum(x2, x2g)  # 闭包区域右下角x坐标
        yc2 = torch.maximum(y2, y2g)  # 闭包区域右下角y坐标

        # 计算闭包区域的宽度和高度
        cw_out = xc2 - xc1  # 闭包区域宽度
        ch_out = yc2 - yc1  # 闭包区域高度

        # 计算中心点之间的距离（在x和y方向上的分量）
        ch = torch.maximum(cy, cyg) - torch.minimum(cy, cyg)  # 中心点y方向距离
        cw = torch.maximum(cx, cxg) - torch.minimum(cx, cxg)  # 中心点x方向距离

        # 计算角度成本（angle cost）
        # 中心点之间的欧氏距离
        dist_intersection = torch.sqrt((cx - cxg) ** 2 + (cy - cyg) ** 2)
        # 计算正弦值
        sin_angle_alpha = ch / dist_intersection  # y方向距离/总距离
        sin_angle_beta = cw / dist_intersection  # x方向距离/总距离
        # 阈值，用于选择较小的正弦值
        thred = torch.pow(torch.tensor(2.0), 0.5) / 2  # sqrt(2)/2 ≈ 0.707
        # 选择较小的正弦值（更接近45度）
        sin_alpha = torch.where(sin_angle_alpha > thred, sin_angle_beta, sin_angle_alpha)
        # 计算角度成本：1 - 2 * sin^2(asin(sin_aplha) - π/4)
        #            = cos^2(asin(sin_alpha) - π/4) - sin^2(asin(sin_aplha) - π/4)
        #            = cos(asin(sin_alpha) * 2 - π/2)
        angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

        # 计算距离成本（distance cost）
        gamma = 2 - angle_cost  # 调整因子
        # 计算标准化的距离（相对于闭包区域）
        beta_x = ((cxg - cx) / cw_out) ** 2  # x方向标准化距离的平方
        beta_y = ((cyg - cy) / ch_out) ** 2  # y方向标准化距离的平方
        # 计算距离成本
        dist_cost = 1 - torch.exp(-gamma * beta_x) + 1 - torch.exp(-gamma * beta_y)

        # 计算形状成本（shape cost）
        # 计算宽高差异的标准化值
        omega_w = torch.abs(w - wg) / torch.maximum(w, wg)  # 宽度差异
        omega_h = torch.abs(hg - h) / torch.maximum(h, hg)  # 高度差异
        # 计算形状成本
        omega = (1 - torch.exp(-omega_w)) ** self.theta + (1 - torch.exp(-omega_h)) ** self.theta

        # 计算最终的SIoU损失
        # 损失 = 1 - IoU + (形状成本 + 距离成本) / 2
        siou_loss = 1 - iou + (omega + dist_cost) / 2

        # 根据reduction模式计算最终损失
        if self.reduction == 'mean':
            siou_loss = torch.mean(siou_loss)
        elif self.reduction == 'sum':
            siou_loss = torch.sum(siou_loss)

        return siou_loss * self.loss_weight