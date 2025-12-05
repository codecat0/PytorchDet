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

    def _forward(self):
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
        body_feats = self.backbone(self.inputs)

        # 2. Neck（如 FPN）融合多尺度特征
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # 3. Transformer 编码 + 解码
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # 4. DETR Head 预测
        if self.training:
            # 训练：计算损失
            detr_losses = self.detr_head(out_transformer, body_feats, self.inputs)
            # 总损失 = 所有非 log 损失项之和
            detr_losses.update({
                'loss': sum(v for k, v in detr_losses.items() if 'log' not in k)
            })
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
                    self.inputs['im_shape'],
                    self.inputs['scale_factor'],
                    self.inputs['image'].shape[2:]  # (H, W)
                )

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        """
        获取训练损失
        """
        return self._forward()

    def get_pred(self):
        """
        获取推理预测
        """
        return self._forward()


if __name__ == '__main__':
    from det.modeling.backbones.resnet import ResNet
    from det.modeling.transformers.detr_transformer import DETRTransformer
    from det.modeling.heads.detr_head import DETRHead
    from det.modeling.losses.detr_loss import DETRLoss
    from det.modeling.transformers.matchers import HungarianMatcher
    from det.modeling.post_process import DETRPostProcess

    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    def deep_pin(blob, non_blocking=True):
        if isinstance(blob, torch.Tensor):
            return blob.cuda(non_blocking=non_blocking)
        elif isinstance(blob, dict):
            return {k: deep_pin(v, non_blocking) for k, v in blob.items()}
        elif isinstance(blob, (list, tuple)):
            return type(blob)([deep_pin(x, non_blocking) for x in blob])
        return blob
    
    cfg = OmegaConf.load('config/base/dataset/coco_dataloader.yaml')
    dataset = instantiate(cfg.dataset)
    dataloader = instantiate(cfg.dataloader, _convert_="all")(dataset, cfg.worker_num)

    num_classes = 1
    hidden_dim = 256
    use_focal_loss = True
    with_mask = False

    backbone = ResNet(
        depth=50,
        norm_type='bn',
        freeze_at=0,
        return_idx=[3],
        num_stages=4,
        freeze_stem_only=True,
    )
    transformer = DETRTransformer(
        num_queries=100,
        position_embed_type='sine',
        nhead=8,
        hidden_dim=hidden_dim,
        num_decoder_layers=6,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
    )
    matcher = HungarianMatcher(
        matcher_coeff={
            'class': 1,
            'bbox': 5,
            'giou': 2,
        },
        with_mask=with_mask,
        use_focal_loss=use_focal_loss,
    )
    loss = DETRLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_coeff={
            'class': 1,
            'bbox': 5,
            'giou': 2,
            'no_object': 0.1,
        },
        aux_loss=True,
        use_focal_loss=use_focal_loss,
    )
    head = DETRHead(
        num_classes=num_classes,
        loss=loss,
        num_mlp_layers=3,
        hidden_dim=hidden_dim,
        use_focal_loss=use_focal_loss,
    )
    post_process = DETRPostProcess(
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
        with_mask=with_mask,
    )
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        detr_head=head,
        post_process=post_process,
        with_mask=with_mask,
    ).cuda()

    # model_cfg = OmegaConf.load('config/base/model/detr.yaml')
    # model = instantiate(model_cfg.model)
    # model = model.cuda()

    inputs = next(iter(dataloader))
    inputs = deep_pin(inputs)
    for k, v in inputs.items():
        if hasattr(v, 'shape'):  
            print(k, v.shape, type(v))
        else:
            print(k, v, type(v))

    print("=== 推理测试 ===")
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    print("输出 keys:", outputs.keys())
    print("bbox shape:", outputs['bbox'].shape)
    print("bbox_num:", outputs['bbox_num'])

    # -----------------------
    # 测试训练
    # -----------------------
    print("\n=== 训练测试 ===")
    model.train()
    losses = model(inputs)
    print("损失 keys:", losses.keys())
    print("总 loss:", losses['loss'].item())
