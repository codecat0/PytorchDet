#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :pose3d_metrics.py
@Author :CodeCat
@Date   :2025/11/19 16:42
"""
import torch
import torch.distributed as dist
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np
from loguru import logger

__all__ = ['Pose3DEval']


class AverageMeter(object):
    def __init__(self):
        """
        初始化平均值计算器
        """
        self.reset()

    def reset(self):
        """
        重置计数器的所有统计值
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计值

        Args:
            val: 当前值
            n: 当前值的权重或数量，默认为1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """
    计算平均每个关节位置误差 (mPJPE)

    Args:
        pred: 预测的3D关节点坐标，形状为 [batch_size, num_joints, 3]
        gt: 真实的3D关节点坐标，形状为 [batch_size, num_joints, 3]
        has_3d_joints: 标记哪些样本有3D关节标注的掩码，形状为 [batch_size]

    Returns:
        error: 每个样本的平均关节位置误差，形状为 [num_valid_samples]
    """
    # 只选择有3D关节标注的样本
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :3]  # 只取xyz坐标
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        # 计算骨盆中心点（第2和第3个关节的平均值作为骨盆位置）
        gt_pelvis = (gt[:, 2, :] + gt[:, 3, :]) / 2
        # 将GT坐标系以骨盆为中心点进行对齐
        gt = gt - gt_pelvis[:, None, :]
        # 计算预测结果的骨盆中心点
        pred_pelvis = (pred[:, 2, :] + pred[:, 3, :]) / 2
        # 将预测结果坐标系以骨盆为中心点进行对齐
        pred = pred - pred_pelvis[:, None, :]
        # 计算预测值和真实值之间的欧氏距离，并求平均
        error = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error


def compute_similarity_transform(S1, S2):
    """
    计算一个相似变换 (sR, t)，将一组3D点 S1 (3 x N) 对齐到另一组3D点 S2，
    其中 R 是 3x3 旋转矩阵，t 是 3x1 平移向量，s 是缩放因子。
    即：求解正交 Procrustes 问题。

    Args:
        S1 (np.ndarray): 源点集，形状为 (3, N) 或 (N, 3)
        S2 (np.ndarray): 目标点集，形状为 (3, N) 或 (N, 3)

    Returns:
        S1_hat (np.ndarray): 对齐后的 S1，形状与输入 S1 一致
    """
    transposed = False
    # 如果输入是 (N, 3) 格式，则转置为 (3, N)
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True

    assert S2.shape[1] == S1.shape[1], "The number of points in S1 and S2 must be the same"

    # 1. 去中心化：移除均值
    mu1 = S1.mean(axis=1, keepdims=True)  # (3, 1)
    mu2 = S2.mean(axis=1, keepdims=True)  # (3, 1)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. 计算 X1 的方差（用于后续缩放）
    var1 = np.sum(X1 ** 2)

    # 3. 计算 X1 和 X2 的外积矩阵 K
    K = X1 @ X2.T  # (3, 3)

    # 4. 对 K 进行 SVD 分解，求最优旋转矩阵 R
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # 构造 Z 矩阵，确保旋转矩阵行列式为 1（避免镜像）
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
    # 构造旋转矩阵 R
    R = V @ (Z @ U.T)

    # 5. 恢复缩放因子 s
    scale = np.trace(R @ K) / var1

    # 6. 恢复平移向量 t
    t = mu2 - scale * (R @ mu1)

    # 7. 应用相似变换：S1_hat = s * R * S1 + t
    S1_hat = scale * R @ S1 + t

    # 如果原始输入是转置格式，则将结果转置回去
    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """
    批量计算相似变换的函数

    Args:
        S1 (np.ndarray): 源点集批次，形状为 (batch_size, ..., N, 3) 或 (batch_size, ..., 3, N)
        S2 (np.ndarray): 目标点集批次，形状与 S1 相同

    Returns:
        S1_hat (np.ndarray): 对齐后的 S1 批次，形状与输入 S1 相同
    """
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """
    执行 Procrustes 对齐并计算重建误差

    Args:
        S1 (np.ndarray): 源点集批次，形状为 (batch_size, ..., N, 3)
        S2 (np.ndarray): 目标点集批次，形状为 (batch_size, ..., N, 3)
        reduction (str): 误差聚合方式，可选 'mean' 或 'sum'

    Returns:
        re (float or np.ndarray): 重建误差
    """
    S1_hat = compute_similarity_transform_batch(S1, S2)
    # 计算对齐后 S1_hat 与 S2 之间的欧氏距离
    re = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1)).mean(axis=-1)

    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()

    return re


def all_gather(data):
    """
    在分布式训练中收集所有进程的数据

    Args:
        data: 当前进程的数据，通常是张量

    Returns:
        data: 所有进程数据拼接后的结果
    """
    if dist.get_world_size() == 1:
        return data

    # 获取当前进程数量
    world_size = dist.get_world_size()
    # 创建一个列表来存储所有进程的数据
    vlist = [torch.zeros_like(data) for _ in range(world_size)]

    # 执行 all_gather 操作
    dist.all_gather(vlist, data)

    # 将所有收集到的数据拼接在一起
    data = torch.cat(vlist, 0)
    return data


class Pose3DEval(object):
    def __init__(self, output_eval, save_prediction_only=False):
        """
        3D姿态评估类

        Args:
            output_eval: 评估结果输出目录
            save_prediction_only: 是否只保存预测结果而不进行评估
        """
        super(Pose3DEval, self).__init__()
        self.output_eval = output_eval
        self.res_file = os.path.join(output_eval, "pose3d_results.json")
        self.save_prediction_only = save_prediction_only
        self.reset()

    def reset(self):
        """
        重置评估器的统计值
        """
        self.PAmPJPE = AverageMeter()
        self.mPJPE = AverageMeter()
        self.eval_results = {}

    @staticmethod
    def get_human36m_joints(input):
        """
        提取Human3.6M数据集的关节点（从24个关节点选择14个）

        Args:
            input: 输入的关节点数据，形状为 [batch_size, 24, ...]

        Returns:
            选择后的关节点数据，形状为 [batch_size, 14, ...]
        """
        J24_TO_J14 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18])
        J24_TO_J17 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19])
        return torch.index_select(input, 1, J24_TO_J14)

    def update(self, inputs, outputs):
        """
        更新评估器的统计值

        Args:
            inputs: 输入数据字典
            outputs: 输出数据字典
        """
        local_rank = torch.distributed.get_rank()

        gt_3d_joints = all_gather(inputs['joints_3d'].cuda(local_rank))
        has_3d_joints = all_gather(inputs['has_3d_joints'].cuda(local_rank))
        pred_3d_joints = all_gather(outputs['pose3d'])

        # 如果关节点数量是24个，则转换为Human3.6M的14个关节点
        if gt_3d_joints.shape[1] == 24:
            gt_3d_joints = self.get_human36m_joints(gt_3d_joints)
        if pred_3d_joints.shape[1] == 24:
            pred_3d_joints = self.get_human36m_joints(pred_3d_joints)

        # 计算mPJPE（平均每个关节位置误差）
        mPJPE_val = mean_per_joint_position_error(pred_3d_joints, gt_3d_joints,
                                                  has_3d_joints).mean()

        # 计算PAmPJPE（Procrustes对齐后的平均每个关节位置误差）
        PAmPJPE_val = reconstruction_error(
            pred_3d_joints.cpu().numpy(),
            gt_3d_joints[:, :, :3].cpu().numpy(),
            reduction=None).mean()

        # 统计有效样本数量
        count = int(np.sum(has_3d_joints.cpu().numpy()))

        # 更新平均值计算器
        self.PAmPJPE.update(PAmPJPE_val * 1000., count)
        self.mPJPE.update(mPJPE_val * 1000., count)

    def accumulate(self):
        """
        汇总评估结果
        """
        if self.save_prediction_only:
            print(f'The pose3d result is saved to {self.res_file} '
                  'and do not evaluate the model.')
            return

        # 保存评估结果（使用负值是因为某些指标越小越好，但这里统一处理）
        self.eval_results['pose3d'] = [-self.mPJPE.avg, -self.PAmPJPE.avg]

    def log(self):
        """
        打印评估结果
        """
        if self.save_prediction_only:
            return

        stats_names = ['mPJPE', 'PAmPJPE']
        num_values = len(stats_names)

        # 打印表头
        print(' '.join(['| {}'.format(name) for name in stats_names]) + ' |')
        print('|---' * (num_values + 1) + '|')

        # 打印数值
        print(' '.join([
            '| {:.3f}'.format(abs(value))
            for value in self.eval_results['pose3d']
        ]) + ' |')

    def get_results(self):
        """
        获取评估结果

        Returns:
            评估结果字典
        """
        return self.eval_results