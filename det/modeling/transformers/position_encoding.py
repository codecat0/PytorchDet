#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :position_encoding.py
@Author :CodeCat
@Date   :2025/11/14 10:40
"""
import torch
import torch.nn as nn
import math


class PositionEmbedding(nn.Module):
    def __init__(self,
                 num_pos_feats=128,
                 temperature=10000,
                 normalize=True,
                 scale=2 * math.pi,
                 embed_type='sine',
                 num_embeddings=50,
                 offset=0.,
                 eps=1e-6):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ['sine', 'learned']

        self.embed_type = embed_type
        self.offset = offset
        self.eps = eps
        if self.embed_type == 'sine':
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            self.scale = scale
        elif self.embed_type == 'learned':
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"{self.embed_type} is not supported.")

    def forward(self, mask):
        """
        参数:
            mask (Tensor): [B, H, W]
        返回:
            pos (Tensor): [B, H, W, C]
        """
        if self.embed_type == 'sine':
            # 计算累积坐标
            y_embed = mask.cumsum(1)  # [B, H, W] - 沿高度方向累积
            x_embed = mask.cumsum(2)  # [B, H, W] - 沿宽度方向累积

            if self.normalize:
                # 归一化坐标到 [0, scale]
                y_embed = (y_embed + self.offset) / (
                        y_embed[:, -1:, :] + self.eps) * self.scale  # [B, H, W]
                x_embed = (x_embed + self.offset) / (
                        x_embed[:, :, -1:] + self.eps) * self.scale  # [B, H, W]

            # 计算正弦位置编码的频率
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32) // 2
            dim_t = dim_t * 2  # 确保偶数索引
            dim_t = self.temperature ** (dim_t / self.num_pos_feats)  # [num_pos_feats]

            # 计算位置编码
            pos_x = x_embed.unsqueeze(-1) / dim_t  # [B, H, W, num_pos_feats]
            pos_y = y_embed.unsqueeze(-1) / dim_t  # [B, H, W, num_pos_feats]

            # 交替应用 sin 和 cos
            # 对于偶数索引使用 sin，奇数索引使用 cos
            pos_x_sin = pos_x[:, :, :, 0::2].sin()  # 偶数位置的 sin
            pos_x_cos = pos_x[:, :, :, 1::2].cos()  # 奇数位置的 cos
            pos_x = torch.cat([pos_x_sin, pos_x_cos], dim=-1)  # [B, H, W, num_pos_feats]

            pos_y_sin = pos_y[:, :, :, 0::2].sin()  # 偶数位置的 sin
            pos_y_cos = pos_y[:, :, :, 1::2].cos()  # 奇数位置的 cos
            pos_y = torch.cat([pos_y_sin, pos_y_cos], dim=-1)  # [B, H, W, num_pos_feats]

            # 拼接 x 和 y 的位置编码
            return torch.cat((pos_y, pos_x), dim=3)  # [B, H, W, 2*num_pos_feats]

        elif self.embed_type == 'learned':
            # 获取高度和宽度
            h, w = mask.shape[-2:]

            # 创建坐标索引
            i = torch.arange(w, device=mask.device)  # [w] - 宽度方向索引
            j = torch.arange(h, device=mask.device)  # [h] - 高度方向索引

            # 获取位置嵌入
            x_emb = self.col_embed(i)  # [w, num_pos_feats] - 宽度方向嵌入
            y_emb = self.row_embed(j)  # [h, num_pos_feats] - 高度方向嵌入

            # 扩展并拼接位置嵌入
            x_emb = x_emb.unsqueeze(0).repeat(h, 1, 1)  # [h, w, num_pos_feats]
            y_emb = y_emb.unsqueeze(1).repeat(1, w, 1)  # [h, w, num_pos_feats]

            # 拼接 x 和 y 嵌入
            pos = torch.cat([x_emb, y_emb], dim=-1)  # [h, w, 2*num_pos_feats]
            pos = pos.unsqueeze(0)  # [1, h, w, 2*num_pos_feats]

            # 扩展到批次大小
            batch_size = mask.shape[0]
            return pos.expand(batch_size, -1, -1, -1)  # [B, h, w, 2*num_pos_feats]

        else:
            raise ValueError(f"not supported {self.embed_type}")