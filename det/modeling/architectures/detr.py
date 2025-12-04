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

    # 图像尺寸
    batch_size = 2
    img_h, img_w = 800, 1024
    num_classes = 1
    hidden_dim = 256
    use_focal_loss = False
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
    )


    # -----------------------
    # 1. 推理模式输入
    # -----------------------
    def make_infer_inputs(batch_size, img_h, img_w):
        """构建推理模式的输入字典"""
        image = torch.randn(batch_size, 3, img_h, img_w)
        im_shape = torch.tensor([[img_h, img_w]] * batch_size, dtype=torch.float32)
        scale_factor = torch.ones(batch_size, 2, dtype=torch.float32)  # 假设无缩放

        return {
            'image': image,
            'im_shape': im_shape,
            'scale_factor': scale_factor
        }


    # -----------------------
    # 2. 训练模式输入
    # -----------------------
    def make_train_inputs(batch_size, img_h, img_w, num_classes):
        """构建训练模式的输入字典（含真实标签）"""
        image = torch.randn(batch_size, 3, img_h, img_w)

        # 真实边界框（格式: cxcywh，归一化到 [0,1]）
        # 假设每张图有 3 个真实目标
        gt_bbox = []
        gt_class = []
        for b in range(batch_size):
            num_gt = 3
            boxes = torch.rand(num_gt, 4)  # cxcywh
            boxes[:, 2:] = boxes[:, 2:].clamp(0.1, 0.9)  # 避免过小框
            classes = torch.randint(1, num_classes+1, (num_gt, 1))
            gt_bbox.append(boxes)
            gt_class.append(classes)

        # 填充到相同长度（模拟 DataLoader 的 collate_fn）
        max_gt = max(len(b) for b in gt_bbox)
        for i in range(batch_size):
            pad_len = max_gt - len(gt_bbox[i])
            if pad_len > 0:
                gt_bbox[i] = F.pad(gt_bbox[i], (0, 0, 0, pad_len))
                gt_class[i] = F.pad(gt_class[i], (0, 0, 0, pad_len))

        gt_bbox = torch.stack(gt_bbox)
        gt_class = torch.stack(gt_class)

        # 填充掩码（假设无 padding）
        pad_mask = torch.zeros(batch_size, img_h, img_w, dtype=torch.float32)

        return {
            'image': image,
            'gt_bbox': [bbox for bbox in gt_bbox],  # [B, max_gt, 4]
            'gt_class': [_class for _class in gt_class],  # [B, max_gt, 1]
            'pad_mask': pad_mask
        }


    # -----------------------
    # 测试推理
    # -----------------------
    print("=== 推理测试 ===")
    model.eval()
    infer_inputs = make_infer_inputs(batch_size, img_h, img_w)
    for k, v in infer_inputs.items():
        print(k, v.shape)
    with torch.no_grad():
        outputs = model(infer_inputs)
    print("输出 keys:", outputs.keys())
    print("bbox shape:", outputs['bbox'].shape)
    print("bbox_num:", outputs['bbox_num'])

    # -----------------------
    # 测试训练
    # -----------------------
    print("\n=== 训练测试 ===")
    model.train()
    train_inputs = make_train_inputs(batch_size, img_h, img_w, num_classes)
    for k, v in train_inputs.items():
        if hasattr(v, 'shape'):
            print(k, v.shape)
    losses = model(train_inputs)
    print("损失 keys:", losses.keys())
    print("总 loss:", losses['loss'].item())
