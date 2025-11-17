#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :ema.py
@Author :CodeCat
@Date   :2025/11/17 10:20
"""
import torch
import math
from torch import nn
from collections import OrderedDict
import weakref

from det.optimizer.utils import get_bn_running_state_names

__all__ = ['ModelEMA', 'SimpleModelEMA']


class ModelEMA(object):
    """
    深度神经网络的指数加权平均
    Args:
        model (nn.Module): 模型
        decay (float): 更新ema参数的衰减系数。
            ema参数通过以下公式更新：
            `ema_param = decay * ema_param + (1 - decay) * cur_param`。
            默认值为0.9998。
        ema_decay_type (str): 类型在['threshold', 'normal', 'exponential']中，
            默认为'threshold'。
        cycle_epoch (int): 重置ema_param和step的间隔周期。
            默认值为-1，表示不重置。其功能是为ema添加正则化效果，
            根据经验设置，在总训练周期较大时有效。
        ema_black_list (set|list|tuple, optional): 自定义EMA黑名单。
            不参与EMA计算的权重名称黑名单。默认：None。
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1,
                 ema_black_list=None,
                 ema_filter_no_grad=False):
        # 初始化计数器和参数
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch

        # 匹配EMA黑名单
        self.ema_black_list = self._match_ema_black_list(
            model.state_dict().keys(), ema_black_list)

        # 获取BN层的运行状态名称
        bn_states_names = get_bn_running_state_names(model)

        # 如果过滤无梯度参数
        if ema_filter_no_grad:
            for n, p in model.named_parameters():
                if not p.requires_grad and n not in bn_states_names:
                    self.ema_black_list.add(n)

        # 初始化EMA状态字典
        self.state_dict = dict()
        for k, v in model.state_dict().items():
            if k in self.ema_black_list:
                # 黑名单中的参数直接复制
                self.state_dict[k] = v.clone()
            else:
                # 非黑名单参数初始化为零
                self.state_dict[k] = torch.zeros_like(v, dtype=torch.float32)

        # 保存模型参数的弱引用
        self._model_state = {
            k: weakref.ref(p)
            for k, p in model.state_dict().items()
        }

    def reset(self):
        """
        重置EMA状态
        将step和epoch计数器归零，并重置EMA参数
        """
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            if k in self.ema_black_list:
                # 黑名单参数保持不变
                self.state_dict[k] = v.clone()
            else:
                # 非黑名单参数重置为零
                self.state_dict[k] = torch.zeros_like(v)

    def resume(self, state_dict, step=0):
        """
        恢复EMA状态
        Args:
            state_dict: 要恢复的状态字典
            step: 恢复的步数
        """
        for k, v in state_dict.items():
            if k in self.state_dict:
                if self.state_dict[k].dtype == v.dtype:
                    # 数据类型相同直接赋值
                    self.state_dict[k] = v.clone()
                else:
                    # 数据类型不同时转换
                    self.state_dict[k] = v.to(self.state_dict[k].dtype)
        self.step = step

    def update(self, model=None):
        """
        更新EMA参数
        根据不同的衰减类型计算新的衰减系数，并更新EMA参数
        """
        # 根据衰减类型计算实际衰减系数
        if self.ema_decay_type == 'threshold':
            # 阈值类型：在训练初期使用较小衰减，逐渐增加到设定值
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == 'exponential':
            # 指数类型：使用指数衰减公式
            decay = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        else:
            # 正常类型：使用固定衰减系数
            decay = self.decay
        self._decay = decay

        # 获取当前模型参数
        if model is not None:
            model_dict = model.state_dict()
        else:
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

        # 更新EMA参数
        for k, v in self.state_dict.items():
            if k not in self.ema_black_list:
                # 非黑名单参数使用EMA公式更新
                updated_v = decay * v + (1 - decay) * model_dict[k].to(torch.float32)
                self.state_dict[k] = updated_v
        self.step += 1

    def apply(self):
        """
        应用EMA参数
        返回当前EMA参数，并处理周期性重置
        """
        if self.step == 0:
            # 如果没有更新过，直接返回当前状态
            return self.state_dict

        state_dict = dict()
        model_dict = {k: p() for k, p in self._model_state.items()}

        for k, v in self.state_dict.items():
            if k in self.ema_black_list:
                # 黑名单参数直接使用原值
                state_dict[k] = v.clone()
            else:
                # 非黑名单参数根据衰减类型进行偏差校正
                if self.ema_decay_type != 'exponential':
                    # 非指数类型需要进行偏差校正
                    corrected_v = v / (1 - self._decay ** self.step)
                    corrected_v = corrected_v.to(model_dict[k].dtype)
                    state_dict[k] = corrected_v
                else:
                    # 指数类型直接使用
                    state_dict[k] = v.clone()

        # 增加epoch计数
        self.epoch += 1

        # 检查是否需要周期性重置
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict

    @staticmethod
    def _match_ema_black_list(weight_name, ema_black_list=None):
        """
        匹配EMA黑名单
        根据给定的黑名单模式匹配实际的权重名称
        Args:
            weight_name: 所有权重名称
            ema_black_list: 黑名单模式列表
        Returns:
            匹配到的黑名单集合
        """
        out_list = set()
        if ema_black_list:
            for name in weight_name:
                for key in ema_black_list:
                    if key in name:
                        out_list.add(name)
        return out_list


class SimpleModelEMA(object):
    """
    模型指数移动平均，来自 https://github.com/rwightman/pytorch-image-models
    保持模型state_dict中所有参数的移动平均值（参数和缓冲区）。
    这是为了实现类似 https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage 的功能
    某些训练方案需要平滑版本的权重才能表现良好。
    这个类对模型初始化、GPU分配和分布式训练包装器的顺序很敏感。
    """

    def __init__(self, model=None, decay=0.9996):
        """
        初始化EMA对象
        Args:
            model (nn.Module): 要应用EMA的模型
            decay (float): EMA衰减率
        """
        self.model = deepcopy(model)
        self.decay = decay

    def update(self, model, decay=None):
        """
        更新EMA参数
        使用指数移动平均公式更新EMA模型的参数

        Args:
            model: 当前训练模型
            decay: 可选的衰减率，如果为None则使用self.decay
        """
        if decay is None:
            decay = self.decay

        with torch.no_grad():  # 禁用梯度计算以节省内存
            state = {}
            msd = model.state_dict()  # 获取当前模型的状态字典
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:  # 只对浮点类型的参数进行EMA更新
                    # EMA公式：ema_param = decay * ema_param + (1 - decay) * current_param
                    v *= decay
                    v += (1.0 - decay) * msd[k].detach()  # detach()断开梯度连接
                state[k] = v
            # 更新EMA模型的状态
            self.model.load_state_dict(state)

    def resume(self, state_dict, step=0):
        """
        从给定的状态字典恢复EMA状态

        Args:
            state_dict: 要恢复的状态字典
            step: 步数（这里保持与原代码一致，但未在类中定义step属性）
        """
        state = {}
        msd = state_dict  # 输入的状态字典
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:  # 只对浮点类型的参数进行恢复
                # 直接从输入状态字典复制参数值
                v = msd[k].detach()
            state[k] = v
        # 更新EMA模型的状态
        self.model.load_state_dict(state)
        # 注意：原代码中使用了self.step但未定义，这里保持原样
        # 如果需要，可以添加 self.step = step