#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :detr.py
@Author :CodeCat
@Date   :2025/12/3 14:59
"""
import torch
from det.modeling.architectures.meta_arch import BaseArch


class DETR(BaseArch):
    """
    DETR（Detection Transformer）架构实现。

    支持检测 + 可选的实例分割（mask）。
    训练时返回 loss 字典，推理时返回 bbox、bbox_num 和 mask（如果启用）。
    """

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 post_process_semi=None,
                 with_mask=False,
                 exclude_post_process=False):
        """
        初始化 DETR 模型。

        Args:
            backbone (nn.Module): 主干网络（如 ResNet）
            transformer (nn.Module): Transformer 编码器-解码器模块
            detr_head (nn.Module): DETR 检测头（负责分类和回归）
            neck (nn.Module, optional): 特征金字塔网络（如 FPN），默认为 None
            post_process (callable): 后处理函数（如 NMS、坐标还原）
            post_process_semi (callable, optional): 半监督训练专用后处理
            with_mask (bool): 是否启用实例分割 mask 输出
            exclude_post_process (bool): 是否跳过后处理（用于中间训练阶段）
        """
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.post_process_semi = post_process_semi

    def _forward(self, inputs):
        """
        DETR 前向传播主逻辑。

        Args:
            inputs (dict): 输入数据字典，应包含:
                - 'image': [B, C, H, W] 图像张量
                - 'pad_mask': [B, H, W] 填充掩码（可选）
                - 'im_shape': [B, 2] 原始图像尺寸（推理时用）
                - 'scale_factor': [B, 2] 缩放因子（推理时用）

        Returns:
            dict:
                - 训练模式: {'loss': total_loss, 'loss_xxx': ..., ...}
                - 推理模式: {'bbox': ..., 'bbox_num': ..., 'mask': (可选)}
        """
        # 1. 主干网络提取特征
        body_feats = self.backbone(inputs['image'])

        # 2. Neck（如 FPN）融合多尺度特征
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # 3. Transformer 编码 + 解码
        pad_mask = inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, inputs)

        # 4. DETR Head 预测
        if self.training:
            # 训练：计算损失
            detr_losses = self.detr_head(out_transformer, body_feats, inputs)
            # 总损失 = 所有非 log 损失项之和
            total_loss = sum(v for k, v in detr_losses.items() if 'log' not in k)
            detr_losses['loss'] = total_loss
            return detr_losses
        else:
            # 推理：预测 + 后处理
            preds = self.detr_head(out_transformer, body_feats)

            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                # 应用后处理（坐标还原、NMS 等）
                bbox, bbox_num, mask = self.post_process(
                    preds,
                    inputs['im_shape'],
                    inputs['scale_factor'],
                    inputs['image'].shape[2:]  # (H, W)
                )

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def forward(self, inputs):
        """
        标准 PyTorch 前向接口，根据训练/评估模式调用 _forward。
        """
        return self._forward(inputs)

    def get_loss(self, inputs):
        """
        获取训练损失（兼容旧接口）。

        Args:
            inputs (dict): 输入数据

        Returns:
            dict: 损失字典
        """
        self.train()
        return self._forward(inputs)

    def get_pred(self, inputs):
        """
        获取推理预测（兼容旧接口）。

        Args:
            inputs (dict): 输入数据

        Returns:
            dict: 预测结果
        """
        self.eval()
        with torch.no_grad():
            return self._forward(inputs)
