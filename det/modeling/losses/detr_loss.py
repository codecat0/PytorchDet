#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :detr_loss.py
@Author :CodeCat
@Date   :2025/11/14 15:29
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple

from det.modeling.losses.iou_loss import GIoULoss
from det.modeling.transformers.utils import bbox_cxcywh_to_xyxy, sigmoid_focal_loss, varifocal_loss_with_logits
from det.modeling.bbox_utils import bbox_iou


class DETRLoss(nn.Module):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 vfl_iou_type='bbox',
                 use_uni_match=False,
                 uni_match_ind=0):
        r"""
        DETR损失函数初始化
        Args:
            num_classes (int): 类别数量
            matcher (HungarianMatcher): 计算目标与网络预测结果之间的匹配
            loss_coeff (dict): 损失系数
            aux_loss (bool): 如果'aux_loss = True'，则使用每个解码器层的损失
            use_focal_loss (bool): 是否使用focal loss
        """
        super(DETRLoss, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.vfl_iou_type = vfl_iou_type
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        if not self.use_focal_loss:
            # 为每个类别设置损失权重，背景类使用no_object权重
            self.loss_coeff['class'] = torch.full([num_classes + 1],
                                                  loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    def _get_loss_class(self,
                        logits: torch.Tensor,
                        gt_class: List[torch.Tensor],
                        match_indices: List[Tuple[torch.Tensor, torch.Tensor]],
                        bg_index: int,
                        num_gts: torch.Tensor,
                        postfix: str = "",
                        iou_score: Optional[torch.Tensor] = None,
                        gt_score: Optional[torch.Tensor] = None):
        """
        计算分类损失
        Args:
            logits: [b, query, num_classes]，预测的logits
            gt_class: [[n, 1]]的列表，真实类别标签
            match_indices: 匹配索引列表
            bg_index: 背景类别索引
            num_gts: 真实目标数量
            postfix: 损失名称后缀
            iou_score: IoU分数
            gt_score: 真实分数
        Returns:
            包含分类损失的字典
        """
        name_class = "loss_class" + postfix

        # 初始化目标标签为背景类
        target_label = torch.full(logits.shape[:2], bg_index, dtype=torch.long, device=logits.device)
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)

        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            # 使用scatter更新目标标签
            target_label_flat = target_label.reshape([-1, 1])
            target_label_flat.scatter_(0, index.unsqueeze(1), updates.long().unsqueeze(1))
            target_label = target_label_flat.reshape([bs, num_query_objects])

        if self.use_focal_loss:
            # 使用focal loss
            target_one_hot = F.one_hot(target_label, self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                if gt_score is not None:
                    target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                    target_score_flat = target_score.reshape([-1, 1])
                    target_score_flat.scatter_(0, index.unsqueeze(1), gt_score)
                    target_score = target_score_flat.reshape([bs, num_query_objects, 1]) * target_one_hot

                    target_score_iou = torch.zeros([bs, num_query_objects], device=logits.device)
                    target_score_iou_flat = target_score_iou.reshape([-1, 1])
                    target_score_iou_flat.scatter_(0, index.unsqueeze(1), iou_score)
                    target_score_iou = target_score_iou_flat.reshape([bs, num_query_objects, 1]) * target_one_hot
                    target_score = target_score * target_score_iou
                    loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                        logits, target_score, target_one_hot,
                        num_gts / num_query_objects)
                else:
                    target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                    if num_gt > 0:
                        target_score_flat = target_score.reshape([-1, 1])
                        target_score_flat.scatter_(0, index.unsqueeze(1), iou_score)
                    target_score = target_score.reshape([bs, num_query_objects, 1]) * target_one_hot
                    loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                        logits, target_score, target_one_hot,
                        num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_one_hot, num_gts / num_query_objects)
        else:
            # 使用交叉熵损失
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])

        return {name_class: loss_}

    def _get_loss_bbox(self, boxes: torch.Tensor, gt_bbox: List[torch.Tensor],
                       match_indices: List[Tuple[torch.Tensor, torch.Tensor]],
                       num_gts: torch.Tensor, postfix: str = ""):
        """
        计算边界框损失（L1损失和GIOU损失）
        Args:
            boxes: [b, query, 4]，预测的边界框
            gt_bbox: [[n, 4]]的列表，真实边界框
            match_indices: 匹配索引列表
            num_gts: 真实目标数量
            postfix: 损失名称后缀
        Returns:
            包含边界框损失的字典
        """
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = torch.tensor([0.], device=boxes.device)
            loss[name_giou] = torch.tensor([0.], device=boxes.device)
            return loss

        # 获取匹配的源边界框和目标边界框
        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        # 计算L1损失
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        # 计算GIOU损失
        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks: torch.Tensor, gt_mask: List[torch.Tensor],
                       match_indices: List[Tuple[torch.Tensor, torch.Tensor]],
                       num_gts: torch.Tensor, postfix: str = ""):
        """
        计算掩码损失（sigmoid focal loss和dice loss）
        Args:
            masks: [b, query, h, w]，预测的掩码
            gt_mask: [[n, H, W]]的列表，真实掩码
            match_indices: 匹配索引列表
            num_gts: 真实目标数量
            postfix: 损失名称后缀
        Returns:
            包含掩码损失的字典
        """
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor([0.], device=masks.device)
            loss[name_dice] = torch.tensor([0.], device=masks.device)
            return loss

        # 获取匹配的源掩码和目标掩码
        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        # 调整掩码尺寸
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        # 计算sigmoid focal loss
        loss[name_mask] = self.loss_coeff['mask'] * sigmoid_focal_loss(
            src_masks,
            target_masks,
            torch.tensor([num_gts], dtype=torch.float32, device=masks.device))
        # 计算dice loss
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    @staticmethod
    def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_gts: torch.Tensor):
        """
        计算dice损失
        Args:
            inputs: 输入张量
            targets: 目标张量
            num_gts: 真实目标数量
        Returns:
            dice损失值
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes: List[torch.Tensor],
                      logits: List[torch.Tensor],
                      gt_bbox: List[torch.Tensor],
                      gt_class: List[torch.Tensor],
                      bg_index: int,
                      num_gts: torch.Tensor,
                      dn_match_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                      postfix: str = "",
                      masks: Optional[List[torch.Tensor]] = None,
                      gt_mask: Optional[List[torch.Tensor]] = None,
                      gt_score: Optional[torch.Tensor] = None):
        """
        计算辅助损失（decoder各层的损失）
        Args:
            boxes: 预测边界框列表
            logits: 预测logits列表
            gt_bbox: 真实边界框列表
            gt_class: 真实类别列表
            bg_index: 背景索引
            num_gts: 真实目标数量
            dn_match_indices: DN匹配索引
            postfix: 损失名称后缀
            masks: 预测掩码列表
            gt_mask: 真实掩码列表
            gt_score: 真实分数
        Returns:
            包含辅助损失的字典
        """
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []

        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)

        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)

            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    # 计算IoU分数
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
                if gt_score is not None:
                    _, target_score = self._get_src_target_assign(
                        logits[-1].detach(), gt_score, match_indices)
            else:
                iou_score = None

            # 计算分类损失
            loss_class.append(
                self._get_loss_class(
                    aux_logits,
                    gt_class,
                    match_indices,
                    bg_index,
                    num_gts,
                    postfix,
                    iou_score,
                    gt_score=target_score
                    if gt_score is not None else None)['loss_class' + postfix])

            # 计算边界框损失
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])

            # 计算掩码损失
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])

        # 汇总所有辅助层的损失
        loss = {
            "loss_class_aux" + postfix: sum(loss_class),
            "loss_bbox_aux" + postfix: sum(loss_bbox),
            "loss_giou_aux" + postfix: sum(loss_giou)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = sum(loss_mask)
            loss["loss_dice_aux" + postfix] = sum(loss_dice)
        return loss

    @staticmethod
    def _get_index_updates(num_query_objects: int, target: List[torch.Tensor],
                           match_indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取索引和更新值用于scatter操作
        Args:
            num_query_objects: 查询对象数量
            target: 目标张量列表
            match_indices: 匹配索引列表
        Returns:
            索引和更新值的元组
        """
        batch_idx = torch.cat([
            torch.full_like(src, i, device=src.device) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)

        target_assign = torch.cat([
            torch.gather(t, 0, dst) if len(dst) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    @staticmethod
    def _get_src_target_assign(src: torch.Tensor, target: List[torch.Tensor],
                               match_indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        获取匹配的源和目标分配
        Args:
            src: 源张量
            target: 目标张量列表
            match_indices: 匹配索引列表
        Returns:
            源分配和目标分配的元组
        """
        src_assign = torch.cat([
            torch.gather(t, 0, I) if len(I) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = torch.cat([
            torch.gather(t, 0, J) if len(J) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    @staticmethod
    def _get_num_gts(targets: List[torch.Tensor], dtype=torch.float32) -> torch.Tensor:
        """
        获取真实目标数量
        Args:
            targets: 目标列表
            dtype: 数据类型
        Returns:
            真实目标数量张量
        """
        num_gts = sum(len(a) for a in targets)
        num_gts = torch.tensor([num_gts], dtype=dtype)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_gts)
            num_gts /= dist.get_world_size()
        num_gts = torch.clamp(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes: torch.Tensor,
                             logits: torch.Tensor,
                             gt_bbox: List[torch.Tensor],
                             gt_class: List[torch.Tensor],
                             masks: Optional[torch.Tensor] = None,
                             gt_mask: Optional[List[torch.Tensor]] = None,
                             postfix: str = "",
                             dn_match_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                             num_gts: torch.Tensor = torch.tensor(1),
                             gt_score: Optional[torch.Tensor] = None):
        """
        获取预测损失
        Args:
            boxes: 预测边界框
            logits: 预测logits
            gt_bbox: 真实边界框列表
            gt_class: 真实类别列表
            masks: 预测掩码
            gt_mask: 真实掩码列表
            postfix: 损失名称后缀
            dn_match_indices: DN匹配索引
            num_gts: 真实目标数量
            gt_score: 真实分数
        Returns:
            包含所有损失的字典
        """
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if gt_score is not None:  # 半监督目标检测
                _, target_score = self._get_src_target_assign(
                    logits[-1].detach(), gt_score, match_indices)
            elif sum(len(a) for a in gt_bbox) > 0:
                if self.vfl_iou_type == 'bbox':
                    # 使用边界框计算IoU
                    src_bbox, target_bbox = self._get_src_target_assign(
                        boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                elif self.vfl_iou_type == 'mask':
                    # 使用掩码计算IoU
                    assert (masks is not None and gt_mask is not None,
                            'Make sure the input has `mask` and `gt_mask`')
                    assert sum(len(a) for a in gt_mask) > 0
                    src_mask, target_mask = self._get_src_target_assign(
                        masks.detach(), gt_mask, match_indices)
                    src_mask = F.interpolate(
                        src_mask.unsqueeze(0),
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    target_mask = F.interpolate(
                        target_mask.unsqueeze(0),
                        size=src_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    src_mask = src_mask.flatten(1)
                    src_mask = torch.sigmoid(src_mask)
                    src_mask = torch.where(src_mask > 0.5, 1., 0.).float()
                    target_mask = target_mask.flatten(1)
                    target_mask = torch.where(
                        target_mask > 0.5, 1., 0.).float()
                    inter = (src_mask * target_mask).sum(1)
                    union = src_mask.sum(1) + target_mask.sum(1) - inter
                    iou_score = (inter + 1e-2) / (union + 1e-2)
                    iou_score = iou_score.unsqueeze(-1)
                else:
                    iou_score = None
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        # 计算分类损失
        loss.update(
            self._get_loss_class(
                logits,
                gt_class,
                match_indices,
                self.num_classes,
                num_gts,
                postfix,
                iou_score,
                gt_score=target_score if gt_score is not None else None))
        # 计算边界框损失
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
                                postfix))
        # 计算掩码损失
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes: torch.Tensor,
                logits: torch.Tensor,
                gt_bbox: List[torch.Tensor],
                gt_class: List[torch.Tensor],
                masks: Optional[torch.Tensor] = None,
                gt_mask: Optional[List[torch.Tensor]] = None,
                postfix: str = "",
                gt_score: Optional[torch.Tensor] = None,
                **kwargs):
        r"""
        前向传播计算损失
        Args:
            boxes (Tensor): [l, b, query, 4]，预测边界框
            logits (Tensor): [l, b, query, num_classes]，预测logits
            gt_bbox (List(Tensor)): [[n, 4]]的列表，真实边界框
            gt_class (List(Tensor)): [[n, 1]]的列表，真实类别
            masks (Tensor, optional): [l, b, query, h, w]，预测掩码
            gt_mask (List(Tensor), optional): [[n, H, W]]的列表，真实掩码
            postfix (str): 损失名称后缀
        """
        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        # 计算最后一层的预测损失
        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts,
            gt_score=gt_score if gt_score is not None else None)

        # 如果使用辅助损失，计算各层的辅助损失
        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask,
                    gt_score=gt_score if gt_score is not None else None))

        return total_loss
