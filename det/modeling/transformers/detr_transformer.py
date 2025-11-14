#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :detr_transformer.py
@Author :CodeCat
@Date   :2025/11/12 15:28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from det.modeling.layers import MultiHeadAttention, _convert_attention_mask
from det.modeling.transformers.position_encoding import PositionEmbedding
from det.modeling.transformers.utils import _get_clones
from det.modeling.initializer import conv_init_, linear_init_
from torch.nn.init import xavier_normal_, normal_

__all__ = ['DETRTransformer']


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        # 自注意力层
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            need_weights=False
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = getattr(F, activation)

        self._reset_parameters()

    def _reset_parameters(self):
        """重置参数"""
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        """
        将位置嵌入添加到张量中

        参数:
            tensor: 输入张量
            pos_embed: 位置嵌入
        返回:
            添加位置嵌入后的张量
        """
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        """
        前向传播

        参数:
            src: 源序列，形状为 [batch_size, seq_len, d_model]
            src_mask: 源序列的注意力掩码
            pos_embed: 位置嵌入
        返回:
            输出张量，形状与输入相同
        """
        residual = src

        if self.normalize_before:
            src = self.norm1(src)

        q = k = self.with_pos_embed(src, pos_embed)
        src_attn, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src_attn)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        # 前馈网络
        src_ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src_ffn)

        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        """
        前向传播

        参数:
            src: 源序列，形状为 [batch_size, seq_len, d_model]
            src_mask: 源序列的注意力掩码
            pos_embed: 位置嵌入
        返回:
            输出张量，形状与输入相同
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        # 自注意力层（目标序列的自注意力）
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            need_weights=False
        )

        # 交叉注意力层（目标序列对源序列的注意力）
        self.cross_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            need_weights=False
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 三个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 三个dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数
        self.activation = getattr(F, activation)

        self._reset_parameters()

    def _reset_parameters(self):
        """重置参数"""
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        """
        将位置嵌入添加到张量中

        参数:
            tensor: 输入张量
            pos_embed: 位置嵌入
        返回:
            添加位置嵌入后的张量
        """
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        """
        前向传播

        参数:
            tgt: 目标序列，形状为 [batch_size, tgt_len, d_model]
            memory: 编码器输出（源序列），形状为 [batch_size, src_len, d_model]
            tgt_mask: 目标序列的注意力掩码
            memory_mask: 源序列的注意力掩码
            pos_embed: 源序列的位置嵌入
            query_pos_embed: 目标序列的位置嵌入
        返回:
            输出张量，形状与tgt相同
        """
        # 转换目标序列掩码的数据类型
        if tgt_mask is not None:
            tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

        # 第一步：自注意力（目标序列内部的注意力）
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt_attn, _ = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)

        tgt = residual + self.dropout1(tgt_attn)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        # 第二步：交叉注意力（目标序列对源序列的注意力）
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt_cross, _ = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)

        tgt = residual + self.dropout2(tgt_cross)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        # 第三步：前馈网络
        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)

        tgt_ffn = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt_ffn)

        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        """
        前向传播

        参数:
            tgt: 目标序列，形状为 [batch_size, tgt_len, d_model]
            memory: 编码器输出（源序列），形状为 [batch_size, src_len, d_model]
            tgt_mask: 目标序列的注意力掩码
            memory_mask: 源序列的注意力掩码
            pos_embed: 源序列的位置嵌入
            query_pos_embed: 目标序列的位置嵌入
        返回:
            输出张量，形状与tgt相同
        """
        # 转换目标序列掩码的数据类型
        if tgt_mask is not None:
            tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                pos_embed=pos_embed,
                query_pos_embed=query_pos_embed)

            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            # 将中间结果堆叠成张量
            return torch.stack(intermediate, dim=0)  # [num_layers, batch_size, tgt_len, d_model]

        return output


class DETRTransformer(nn.Module):
    def __init__(self,
                 num_queries=100,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=2048,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 pe_offset=0.,
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(DETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        # 编码器层
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        # 解码器层
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        # 输入投影层（将backbone特征投影到hidden_dim）
        self.input_proj = nn.Conv2d(
            backbone_num_channels, hidden_dim, kernel_size=1)

        # 查询位置嵌入（object queries）
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        # 位置嵌入
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            temperature=pe_temperature,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=pe_offset)

        self._reset_parameters()

    def _reset_parameters(self):
        """重置参数"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        conv_init_(self.input_proj)
        normal_(self.query_pos_embed.weight)

    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        从配置和输入形状创建DETRTransformer实例

        参数:
            cfg: 配置对象
            input_shape: 输入形状列表，包含每个层级的特征图通道数等信息

        返回:
            包含backbone_num_channels的字典，用于初始化参数
        """
        return {
            'backbone_num_channels': [i.channels for i in input_shape][-1],
        }

    @staticmethod
    def _convert_attention_mask(mask):
        """转换注意力掩码"""
        return (mask - 1.0) * 1e9

    def forward(self, src, src_mask=None, *args, **kwargs):
        r"""
        对输入应用Transformer模型。

        参数:
            src (List(Tensor)): Backbone特征图，形状为[[bs, c, h, w]]。
            src_mask (Tensor, optional): 用于多头注意力的张量，用于防止对某些不想要的位置进行注意力，
                通常是填充或后续位置。它是一个形状为[bs, H, W]的张量。当数据类型为bool时，
                不想要的位置有False值，其他位置有True值。当数据类型为int时，不想要的位置有0值，
                其他位置有1值。当数据类型为float时，不想要的位置有-INF值，其他位置有0值。
                当不需要或不需要防止注意力时，它可以是None。默认None。

        返回:
            output (Tensor): [num_levels, batch_size, num_queries, hidden_dim]
            memory (Tensor): [batch_size, hidden_dim, h, w]
        """
        # 使用最后一层特征图
        src_proj = self.input_proj(src[-1])  # [bs, hidden_dim, h, w]
        bs, c, h, w = src_proj.shape

        # 展平 [B, C, H, W] 到 [B, HxW, C]
        src_flatten = src_proj.flatten(2).transpose(1, 2)  # [B, H*W, C]

        if src_mask is not None:
            # 调整掩码大小以匹配特征图尺寸
            src_mask = F.interpolate(src_mask.unsqueeze(0), size=(h, w))[0]
        else:
            # 如果没有掩码，创建全1的掩码
            src_mask = torch.ones([bs, h, w], device=src_proj.device, dtype=src_proj.dtype)

        # 计算位置嵌入
        pos_embed = self.position_embedding(src_mask).flatten(1, 2)  # [B, H*W, C]

        if self.training:
            # 训练时转换注意力掩码
            src_mask = self._convert_attention_mask(src_mask)
            src_mask = src_mask.reshape([bs, 1, 1, h * w])
        else:
            # 推理时不需要掩码
            src_mask = None

        # 编码器前向传播
        memory = self.encoder(
            src_flatten, src_mask=src_mask, pos_embed=pos_embed)  # [B, H*W, C]

        # 创建查询位置嵌入和目标张量
        query_pos_embed = self.query_pos_embed.weight.unsqueeze(0).expand(
            bs, -1, -1)  # [B, num_queries, hidden_dim]
        tgt = torch.zeros_like(query_pos_embed)  # [B, num_queries, hidden_dim]

        # 解码器前向传播
        output = self.decoder(
            tgt,
            memory,
            memory_mask=src_mask,
            pos_embed=pos_embed,
            query_pos_embed=query_pos_embed)  # [num_levels, B, num_queries, hidden_dim]

        if self.training:
            src_mask = src_mask.reshape([bs, 1, 1, h, w])
        else:
            src_mask = None

        # 重构memory为特征图格式
        memory_reshaped = memory.transpose(1, 2).reshape([bs, c, h, w])  # [B, C, H, W]

        return output, memory_reshaped, src_proj, src_mask
