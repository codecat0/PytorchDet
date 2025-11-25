#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :naive_sync_bn.py
@Author :CodeCat
@Date   :2025/11/25 9:20
"""
import torch
import torch.distributed as dist
import torch.nn as nn


class _AllReduce(torch.autograd.Function):
    """
    自定义反向传播的 AllReduce 操作，用于实现跨 GPU 的梯度同步。
    注意：forward 中使用 all_gather + sum 来模拟 all_reduce（非 in-place），
          backward 中直接使用 all_reduce 来同步梯度。
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        前向传播：收集所有 rank 的输入张量并求和。

        Args:
            input_tensor (Tensor): 当前 rank 的输入张量

        Returns:
            Tensor: 所有 rank 输入张量的和
        """
        world_size = dist.get_world_size()
        if world_size == 1:
            return input_tensor.clone()

        # 创建接收缓冲区
        input_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
        # 使用 all_gather 收集所有 rank 的数据
        dist.all_gather(input_list, input_tensor)
        # 堆叠并求和（等效于 all_reduce(sum)）
        stacked = torch.stack(input_list, dim=0)
        return torch.sum(stacked, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        反向传播：对梯度执行 all_reduce，实现梯度同步。

        Args:
            grad_output (Tensor): 上游梯度

        Returns:
            Tensor: 同步后的梯度
        """
        world_size = dist.get_world_size()
        if world_size == 1:
            return grad_output

        # 在反向传播中对梯度做 in-place all_reduce
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


def differentiable_all_reduce(input):
    """
    `dist.all_reduce` 的可微分版本，用于在保持梯度传播的同时进行跨 GPU 的张量同步。

    该函数在单卡或未启用分布式训练时直接返回输入；
    在多卡分布式训练时，通过自定义 autograd Function `_AllReduce` 实现前向求和、反向同步梯度。

    Args:
        input (torch.Tensor): 需要同步的张量

    Returns:
        torch.Tensor: 所有 rank 上该张量的和（前向），且在反向传播时会自动同步梯度
    """
    if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
    ):
        return input
    return _AllReduce.apply(input)


class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    简易版同步 Batch Normalization（SyncBN），在多卡训练时同步全局均值和方差。

    支持两种统计模式：
      - stats_mode = "": 直接平均各卡的均值和均方（标准 SyncBN）
      - stats_mode = "N": 按 batch size 加权平均（支持 dynamic batch size，包括 batch=0）
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, stats_mode=""):
        super(NaiveSyncBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        assert stats_mode in ["", "N"], "stats_mode must be '' or 'N'"
        self._stats_mode = stats_mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 单卡或评估模式下，直接调用标准 BatchNorm
        if dist.get_world_size() == 1 or not self.training:
            return super(NaiveSyncBatchNorm, self).forward(input)

        B, C = input.shape[0], input.shape[1]
        # 计算当前卡的均值和均方值（在 N, H, W 维度上）
        mean = input.mean(dim=[0, 2, 3])
        meansqr = (input * input).mean(dim=[0, 2, 3])

        if self._stats_mode == "":
            # 标准 SyncBN：直接平均各卡统计量
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            # 拼接均值和均方值
            vec = torch.cat([mean, meansqr], dim=0)
            # 全局同步并求平均
            vec = differentiable_all_reduce(vec) / dist.get_world_size()
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum  # PyTorch 的 momentum 定义与 Paddle 一致

        else:
            # 支持动态 batch size（包括 B=0）
            if B == 0:
                # 构造零向量，但保留梯度（通过 input.sum()）
                vec = torch.zeros([2 * C + 1], dtype=mean.dtype, device=mean.device)
                vec = vec + input.sum()  # 保证对 input 有梯度
            else:
                # 拼接均值、均方值和 batch size（用于加权）
                vec = torch.cat([
                    mean,
                    meansqr,
                    torch.ones(1, dtype=mean.dtype, device=mean.device)
                ], dim=0)
            # 同步：乘以本地 batch size 后 all_reduce
            vec = differentiable_all_reduce(vec * B)

            # 提取总 batch size（最后一个元素）
            total_batch = vec[-1].detach()
            # 动态 momentum：当 total_batch 为 0 时不更新
            momentum = (total_batch.clamp(max=1) * self.momentum).item()
            # 避免除零：至少按 1 归一化
            vec = vec / total_batch.clamp(min=1)
            mean, meansqr = torch.split(vec, [C, C])

        # 计算方差和标准差
        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)

        # 融合 scale 和 bias
        if self.affine:
            scale = self.weight * invstd
            bias = self.bias - mean * scale
        else:
            scale = invstd
            bias = -mean * scale

        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        # 更新 running_mean 和 running_var（仅在训练时）
        if self.track_running_stats and self.momentum > 0:
            with torch.no_grad():
                self.running_mean = self.running_mean + momentum * (mean - self.running_mean)
                self.running_var = self.running_var + momentum * (var - self.running_var)

        # 应用归一化
        return input * scale + bias


def convert_syncbn(model):
    """
    递归地将模型中的所有标准 BatchNorm2d 层替换为 NaiveSyncBatchNorm 层，
    用于支持多 GPU 分布式训练时的全局统计量同步。

    Args:
        model (torch.nn.Module): 待转换的模型
    """
    for n, m in model.named_children():
        if isinstance(m, nn.BatchNorm2d):
            # 创建 NaiveSyncBatchNorm 实例，保留原始 BN 的参数
            syncbn = NaiveSyncBatchNorm(
                num_features=m.num_features,
                eps=m.eps,
                momentum=m.momentum,
                affine=m.affine,
                track_running_stats=m.track_running_stats
            )
            # 复制原始 BN 的可学习参数（如果存在）
            if m.affine:
                syncbn.weight.data = m.weight.data.clone()
                syncbn.bias.data = m.bias.data.clone()
            # 复制运行时统计量
            if m.track_running_stats:
                syncbn.running_mean = m.running_mean.clone()
                syncbn.running_var = m.running_var.clone()
                syncbn.num_batches_tracked = m.num_batches_tracked.clone()

            # 替换子模块
            setattr(model, n, syncbn)
        else:
            # 递归处理子模块
            convert_syncbn(m)


def convert_bn(model):
    """
    递归地将模型中的所有 SyncBatchNorm 层替换为标准 BatchNorm2d 层。
    常用于将训练好的 SyncBN 模型转换为单卡推理模型（避免分布式依赖）。

    Args:
        model (torch.nn.Module): 待转换的模型
    """
    for n, m in model.named_children():
        if isinstance(m, nn.SyncBatchNorm):
            # 创建标准 BatchNorm2d，保留原 SyncBN 的配置
            bn = nn.BatchNorm2d(
                num_features=m.num_features,
                eps=m.eps,
                momentum=m.momentum,
                affine=m.affine,
                track_running_stats=m.track_running_stats
            )
            # 复制参数和运行时统计量
            bn.load_state_dict(m.state_dict())
            # 替换子模块
            setattr(model, n, bn)
        else:
            # 递归处理子模块
            convert_bn(m)