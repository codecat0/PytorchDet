#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :fuse_utils.py
@Author :CodeCat
@Date   :2025/11/21 16:29
"""
import copy
import torch
import torch.nn as nn


__all__ = ['fuse_conv_bn']

def find_parent_layer_and_sub_name(model, name):
    """
    给定模型和层的名称，找到父层和子名称。
    例如，如果名称是 'block_1/convbn_1/conv_1'，父层是
    'block_1/convbn_1'，子名称是 `conv_1`。
    Args:
        model(torch.nn.Module): 要量化的模型。
        name(string): 层的名称

    Returns:
        parent_layer, subname
    """
    assert isinstance(model, torch.nn.Module), \
        "The model must be the instance of torch.nn.Module."
    assert len(name) > 0, "The input (name) should not be empty."

    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == '.':
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


class Identity(nn.Module):
    """a layer to replace bn or relu layers"""

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """fuse weights and bias of conv and bn"""
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    conv_w = conv_w * \
             (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return conv_w, conv_b


def _fuse_conv_bn_eval(conv, bn):
    """fuse conv and bn for eval"""
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_weight, fused_bias = _fuse_conv_bn_weights(
        fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps,
        bn.weight, bn.bias)
    fused_conv.weight.data = fused_weight
    if fused_conv.bias is None:
        fused_conv.bias = torch.nn.Parameter(
            torch.zeros(fused_conv.out_channels, dtype=bn.bias.dtype))
    fused_conv.bias.data = fused_bias
    return fused_conv


def _fuse_conv_bn(conv, bn):
    """fuse conv and bn for train or eval"""
    assert (conv.training == bn.training), \
        "Conv and BN both must be in the same mode (train or eval)."
    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        raise NotImplementedError
    else:
        return _fuse_conv_bn_eval(conv, bn)


types_to_fusion_method = {(torch.nn.Conv2d, torch.nn.BatchNorm2d): _fuse_conv_bn, }


def _fuse_func(layer_list):
    """选择融合方法并融合层"""
    types = tuple(type(m) for m in layer_list)
    fusion_method = types_to_fusion_method.get(types, None)
    if fusion_method is None:
        # 如果没有找到对应的融合方法，返回原层列表
        return layer_list

    new_layers = [None] * len(layer_list)
    fused_layer = fusion_method(*layer_list)

    # 转移第一个层的前置hook
    for handle_id, pre_hook_fn in layer_list[0]._forward_pre_hooks.items():
        fused_layer.register_forward_pre_hook(pre_hook_fn)
        del layer_list[0]._forward_pre_hooks[handle_id]

    # 转移最后一个层的后置hook
    for handle_id, hook_fn in layer_list[-1]._forward_post_hooks.items():
        fused_layer.register_forward_post_hook(hook_fn)
        del layer_list[-1]._forward_post_hooks[handle_id]

    new_layers[0] = fused_layer
    # 用Identity层替换后续层
    for i in range(1, len(layer_list)):
        identity = Identity()
        identity.training = layer_list[0].training
        new_layers[i] = identity
    return new_layers


def fuse_layers(model, layers_to_fuse, inplace=False):
    """
    融合模型中的指定层（例如Conv+BN融合）

    Args:
        model (torch.nn.Module): 需要进行层融合的模型
        layers_to_fuse (list): 需要融合的层名列表。例如:
            fuse_list = [["conv1", "bn1"], ["conv2", "bn2"]]
            表示将conv1和bn1融合，以及conv2和bn2融合。
            如果设置了fuse=True但fuse_list为None，则会引发TypeError。
            默认: None
        inplace (bool): 是否直接对输入模型进行融合操作。
                       默认: False (返回融合后的新模型)

    Returns:
        fused_model (torch.nn.Module): 融合后的模型
    """
    # 如果不是就地融合，则创建模型的深拷贝以避免修改原始模型
    if not inplace:
        model = copy.deepcopy(model)

    # 遍历每一组需要融合的层
    for layers_list in layers_to_fuse:
        layer_list = []  # 存储当前组需要融合的层实例

        # 获取当前组中每个层的实例
        for layer_name in layers_list:
            parent_layer, sub_name = find_parent_layer_and_sub_name(model,
                                                                    layer_name)
            layer_instance = getattr(parent_layer, sub_name)
            layer_list.append(layer_instance)

        # 使用融合函数对当前组的层进行融合
        new_layers = _fuse_func(layer_list)

        # 将融合后的层替换回模型中
        for i, item in enumerate(layers_list):
            parent_layer, sub_name = find_parent_layer_and_sub_name(model, item)
            setattr(parent_layer, sub_name, new_layers[i])

    return model


def fuse_conv_bn(model):
    """
    自动查找模型中的Conv2D和BatchNorm2D层对，并将它们融合成一个卷积层。
    融合通常用于推理阶段以提高计算效率和减少内存使用。

    Args:
        model (torch.nn.Module): 需要进行Conv-BN融合的PyTorch模型。

    Returns:
        torch.nn.Module: 融合了Conv-BN层的新模型。如果模型在调用此函数前处于训练模式，
                         返回的模型将恢复到训练模式；否则保持评估模式。
    """
    # 保存原始的训练状态
    is_train = False
    if model.training:
        # 如果模型处于训练模式，则临时切换到评估模式
        # 这是因为BN层在eval模式下有固定的均值和方差，才能被融合
        model.eval()
        is_train = True

    # 初始化用于存储需要融合的层名对的列表
    fuse_list = []
    # 临时配对列表，用于存储当前找到的Conv和BN层的名称
    # tmp_pair[0] 存Conv层名, tmp_pair[1] 存BN层名
    tmp_pair = [None, None]

    # 遍历模型的所有子层
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            tmp_pair[0] = name
        elif isinstance(layer, torch.nn.BatchNorm2d):
            tmp_pair[1] = name

        # 如果当前配对中的Conv和BN都找到了
        if tmp_pair[0] is not None and tmp_pair[1] is not None:
            # 将这对层名添加到融合列表中
            fuse_list.append([tmp_pair[0], tmp_pair[1]])
            # 重置临时配对列表，准备寻找下一组Conv-BN对
            tmp_pair = [None, None]

    # 执行层融合
    model = fuse_layers(model, fuse_list)

    # 如果原始模型处于训练模式，则恢复训练模式
    if is_train:
        model.train()

    return model