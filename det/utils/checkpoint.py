#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :checkpoint.py
@Author :CodeCat
@Date   :2025/11/17 10:41
"""
import os
import torch
import numpy as np
from loguru import logger


def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    else:
        return obj


def match_state_dict(model_state_dict, weight_state_dict, mode='default'):
    """
    匹配模型状态字典和预训练权重状态字典。
    返回匹配后的状态字典。

    该方法假设预训练权重状态字典中的所有名称都是模型名称的子串，
    如果去掉预训练权重键中的前缀'backbone.'。我们可以为每个模型键
    获取候选匹配项。然后选择匹配长度最长的名称作为最终匹配结果。
    例如，模型状态字典有名称'backbone.res2.res2a.branch2a.conv.weight'
    而预训练权重有名称'res2.res2a.branch2a.conv.weight'和'branch2a.conv.weight'。
    我们将'res2.res2a.branch2a.conv.weight'匹配到模型键。
    """

    model_keys = sorted(model_state_dict.keys())
    weight_keys = sorted(weight_state_dict.keys())

    def teacher_match(a, b):
        # 跳过学生参数
        if b.startswith('modelStudent'):
            return False
        return a == b or a.endswith("." + b) or b.endswith("." + a)

    def student_match(a, b):
        # 跳过教师参数
        if b.startswith('modelTeacher'):
            return False
        return a == b or a.endswith("." + b) or b.endswith("." + a)

    def match(a, b):
        if b.startswith('backbone.res5'):
            b = b[9:]
        return a == b or a.endswith("." + b)

    if mode == 'student':
        match_op = student_match
    elif mode == 'teacher':
        match_op = teacher_match
    else:
        match_op = match

    # 创建匹配矩阵，存储匹配长度
    match_matrix = np.zeros([len(model_keys), len(weight_keys)])
    for i, m_k in enumerate(model_keys):
        for j, w_k in enumerate(weight_keys):
            if match_op(m_k, w_k):
                match_matrix[i, j] = len(w_k)

    # 找到每个模型键的最佳匹配权重键
    max_id = match_matrix.argmax(1)
    max_len = match_matrix.max(1)
    max_id[max_len == 0] = -1  # 没有匹配的设为-1

    # 获取所有被匹配的权重键索引
    load_id = set(max_id)
    load_id.discard(-1)

    not_load_weight_name = []

    if (weight_keys and
            (weight_keys[0].startswith('modelStudent') or weight_keys[0].startswith('modelTeacher'))):
        # 处理教师-学生模型的情况
        for match_idx in range(len(max_id)):
            if max_id[match_idx] == -1:
                not_load_weight_name.append(model_keys[match_idx])
        if len(not_load_weight_name) > 0:
            logger.info('{} in model is not matched with pretrained weights, '
                        'and it will be trained from scratch'.format(not_load_weight_name))

    else:
        # 处理普通模型的情况
        for idx in range(len(weight_keys)):
            if idx not in load_id:
                not_load_weight_name.append(weight_keys[idx])

        if len(not_load_weight_name) > 0:
            logger.info('{} in pretrained weight is not used in the model, '
                        'and it will not be loaded'.format(not_load_weight_name))

    matched_keys = {}
    result_state_dict = {}

    # 构建最终的匹配状态字典
    for model_id, weight_id in enumerate(max_id):
        if weight_id == -1:
            continue
        model_key = model_keys[model_id]
        weight_key = weight_keys[weight_id]
        weight_value = weight_state_dict[weight_key]
        model_value_shape = list(model_state_dict[model_key].shape)

        # 检查权重形状是否匹配
        if list(weight_value.shape) != model_value_shape:
            logger.info(
                'The shape {} in pretrained weight {} is unmatched with '
                'the shape {} in model {}. And the weight {} will not be '
                'loaded'.format(weight_value.shape, weight_key,
                                model_value_shape, model_key, weight_key))
            continue

        # 确保模型键没有重复
        assert model_key not in result_state_dict
        result_state_dict[model_key] = weight_value

        # 检查是否存在模糊匹配（一个权重键匹配多个模型键）
        if weight_key in matched_keys:
            raise ValueError('Ambiguity weight {} loaded, it matches at least '
                             '{} and {} in the model'.format(weight_key, model_key, matched_keys[weight_key]))
        matched_keys[weight_key] = model_key

    return result_state_dict


def _strip_postfix(path):
    """移除路径的后缀"""
    # 例如，移除.pth, .pt等
    for ext in ['.pth', '.pt']:
        if path.endswith(ext):
            return path[:-len(ext)]
    return path


def load_weight(model, weight, optimizer=None, ema=None, exchange=True):
    """
    加载模型权重

    Args:
        model: PyTorch模型
        weight: 权重文件路径或URL
        optimizer: 优化器对象（可选）
        ema: EMA对象（可选）
        exchange: 是否交换模型和EMA权重加载顺序

    Returns:
        last_epoch: 最后一个epoch数
    """
    path = _strip_postfix(weight)
    param_path = path + '.pth'
    # 也可以根据需要使用.pt后缀
    # param_path = path + '.pt'

    if not os.path.exists(param_path):
        raise ValueError("Model pretrain path {} does not exist.".format(param_path))

    # 检查是否有EMA权重文件 (.ema.pth 或 .ema.pt)
    ema_state_dict = None
    param_state_dict = None

    if ema is not None and os.path.exists(path + '.ema.pth'):  # 根据实际EMA文件格式调整
        if exchange:
            # 交换模型和EMA模型来加载权重
            logger.info('Exchange model and ema_model to load:')
            # 加载原始权重文件作为EMA状态
            ema_state_dict = torch.load(param_path, map_location='cpu')
            logger.info('Loading ema_model weights from {}'.format(param_path))
            # 加载EMA文件作为模型状态
            param_state_dict = torch.load(path + '.ema.pth', map_location='cpu')
            logger.info('Loading model weights from {}'.format(path + '.ema.pth'))
        else:
            # 正常加载：EMA文件作为EMA状态，原始文件作为模型状态
            ema_state_dict = torch.load(path + '.ema.pth', map_location='cpu')
            logger.info('Loading ema_model weights from {}'.format(path + '.ema.pth'))
            param_state_dict = torch.load(param_path, map_location='cpu')
            logger.info('Loading model weights from {}'.format(param_path))
    else:
        ema_state_dict = None
        param_state_dict = torch.load(param_path, map_location='cpu')

    # 检查是否为教师-学生框架模型
    if hasattr(model, 'modelTeacher') and hasattr(model, 'modelStudent'):
        print('Loading pretrain weights for Teacher-Student framework.')
        print('Loading pretrain weights for Student model.')

        student_model_dict = model.modelStudent.state_dict()
        student_param_state_dict = match_state_dict(
            student_model_dict, param_state_dict, mode='student')
        # 在PyTorch中使用load_state_dict，可以设置strict=False以允许部分匹配
        model.modelStudent.load_state_dict(student_param_state_dict, strict=False)

        print('Loading pretrain weights for Teacher model.')
        teacher_model_dict = model.modelTeacher.state_dict()
        teacher_param_state_dict = match_state_dict(
            teacher_model_dict, param_state_dict, mode='teacher')
        model.modelTeacher.load_state_dict(teacher_param_state_dict, strict=False)

    else:
        # 普通模型加载
        model_dict = model.state_dict()
        model_weight = {}
        incorrect_keys = 0

        for key in model_dict.keys():
            if key in param_state_dict.keys():
                # 检查形状是否匹配
                if model_dict[key].shape == param_state_dict[key].shape:
                    model_weight[key] = param_state_dict[key]
                else:
                    logger.warning(f"Shape mismatch for key {key}: "
                                   f"model {model_dict[key].shape} vs param {param_state_dict[key].shape}")
            else:
                logger.info('Unmatched key: {}'.format(key))
                incorrect_keys += 1

        if incorrect_keys > 0:
            logger.warning(f"Load weight {weight} with {incorrect_keys} unmatched keys. "
                           f"This might be expected for some architectures (e.g., classifier layers).")

        # 加载权重，允许部分匹配
        model.load_state_dict(model_weight, strict=False)
        logger.info('Finish resuming model weights: {}'.format(param_path))

    last_epoch = 0

    # 加载优化器状态
    if optimizer is not None and os.path.exists(path + '.opt.pth'):
        optim_state_dict = torch.load(path + '.opt.pth', map_location='cpu')

        for key in optimizer.state_dict().keys():
            if key not in optim_state_dict:
                logger.warning(f"Key {key} not found in optimizer checkpoint, keeping current state.")

        if 'last_epoch' in optim_state_dict:
            last_epoch = optim_state_dict.pop('last_epoch')
        elif 'epoch' in optim_state_dict:  # 一些优化器可能使用'epoch'而不是'last_epoch'
            last_epoch = optim_state_dict.pop('epoch')

        optimizer.load_state_dict(optim_state_dict)

        # 如果有EMA状态字典，恢复EMA状态
        if ema_state_dict is not None:
            # 获取调度器信息以获取epoch数
            scheduler_epoch = 0
            if 'LR_Scheduler' in optim_state_dict and 'last_epoch' in optim_state_dict['LR_Scheduler']:
                scheduler_epoch = optim_state_dict['LR_Scheduler']['last_epoch']
            # 或者直接从optim_state_dict中获取
            elif 'last_epoch' in optim_state_dict:
                scheduler_epoch = optim_state_dict['last_epoch']

            ema.resume(ema_state_dict, step=scheduler_epoch)
    elif ema_state_dict is not None:
        # 如果没有优化器状态，直接恢复EMA，步数设为0
        ema.resume(ema_state_dict)

    return last_epoch


def load_pretrain_weight(model, pretrain_weight, ARSL_eval=False):
    """
    加载预训练权重

    Args:
        model: PyTorch模型
        pretrain_weight: 预训练权重路径或URL
        ARSL_eval: 是否为ARSL评估模式
    """

    path = _strip_postfix(pretrain_weight)
    # 检查路径是否存在（目录、文件或权重文件）
    if not (os.path.isdir(path) or os.path.isfile(path) or
            os.path.exists(path + '.pth')):  # 修改文件扩展名
        raise ValueError("Model pretrain path `{}` does not exists. "
                         "If you don't want to load pretrain model, "
                         "please delete `pretrain_weights` field in "
                         "config file.".format(path))

    teacher_student_flag = False

    if not ARSL_eval:
        # 非ARSL评估模式
        if hasattr(model, 'modelTeacher') and hasattr(model, 'modelStudent'):
            print('Loading pretrain weights for Teacher-Student framework.')
            print(
                'Assert Teacher model has the same structure with Student model.'
            )
            model_dict = model.modelStudent.state_dict()
            teacher_student_flag = True
        else:
            model_dict = model.state_dict()

        # 加载权重文件
        weights_path = path + '.pth'  # 修改文件扩展名
        param_state_dict = torch.load(weights_path, map_location='cpu')

        # 匹配状态字典
        param_state_dict = match_state_dict(model_dict, param_state_dict)

        # 确保张量类型匹配
        for k, v in param_state_dict.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if model_dict[k].dtype != v.dtype:
                param_state_dict[k] = v.to(model_dict[k].dtype)

        # 加载权重到模型
        if teacher_student_flag:
            model.modelStudent.load_state_dict(param_state_dict, strict=False)
            model.modelTeacher.load_state_dict(param_state_dict, strict=False)
        else:
            model.load_state_dict(param_state_dict, strict=False)
        logger.info('Finish loading model weights: {}'.format(weights_path))

    else:
        # ARSL评估模式
        weights_path = path + '.pth'  # 修改文件扩展名
        param_state_dict = torch.load(weights_path, map_location='cpu')

        # 加载学生模型权重
        student_model_dict = model.modelStudent.state_dict()
        student_param_state_dict = match_state_dict(
            student_model_dict, param_state_dict, mode='student')
        model.modelStudent.load_state_dict(student_param_state_dict, strict=False)

        print('Loading pretrain weights for Teacher model.')
        teacher_model_dict = model.modelTeacher.state_dict()

        # 加载教师模型权重
        teacher_param_state_dict = match_state_dict(
            teacher_model_dict, param_state_dict, mode='teacher')
        model.modelTeacher.load_state_dict(teacher_param_state_dict, strict=False)

        logger.info('Finish loading model weights: {}'.format(weights_path))


def save_model(model,
               optimizer,
               save_dir,
               save_name,
               last_epoch,
               ema_model=None):
    """
    将模型保存到磁盘。

    Args:
        model: 模型状态字典或PyTorch模型实例，用于保存参数。
        optimizer (torch.optim.Optimizer): 优化器实例，用于保存优化器状态。
        save_dir (str): 要保存的目录。
        save_name (str): 要保存的路径。
        last_epoch (int): epoch索引。
        ema_model: EMA模型状态字典或PyTorch模型实例，用于保存参数。
    """
    # 在分布式训练中，只让rank 0进程保存模型
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    save_dir = os.path.normpath(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_name == "best_model":
        best_model_path = os.path.join(save_dir, 'best_model')
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)

    save_path = os.path.join(save_dir, save_name)

    # 保存模型
    if isinstance(model, torch.nn.Module):
        # 如果model是PyTorch模型实例，获取其状态字典
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, save_path + ".pth")
        best_model = model_state_dict
    else:
        # 如果model已经是状态字典
        assert isinstance(model, dict), 'model is not an instance of nn.Module or dict'
        if ema_model is None:
            torch.save(model, save_path + ".pth")
            best_model = model
        else:
            # 如果提供了EMA模型，则交换模型和EMA模型来保存
            # EMA模型保存为 .pth (主权重文件)
            # 原始模型保存为 .ema.pth (原始权重文件)
            assert isinstance(ema_model, dict), ("ema_model is not an instance of dict, "
                                                 "please call model.state_dict() to get.")
            torch.save(ema_model, save_path + ".pth")
            torch.save(model, save_path + ".ema.pth")
            best_model = ema_model

    if save_name == 'best_model':
        best_model_path = os.path.join(best_model_path, 'model')
        torch.save(best_model, best_model_path + ".pth")

    # 保存优化器
    state_dict = optimizer.state_dict()
    state_dict['last_epoch'] = last_epoch
    torch.save(state_dict, save_path + ".opt.pth")  # 优化器状态通常用.opt.pth扩展名

    logger.info("Save checkpoint: {}".format(save_dir))


def save_semi_model(teacher_model, student_model, optimizer, save_dir,
                    save_name, last_epoch, last_iter):
    """
    将教师模型和学生模型保存到磁盘。

    Args:
        teacher_model (dict): 教师模型的 state_dict，用于保存参数。
        student_model (dict): 学生模型的 state_dict，用于保存参数。
        optimizer (torch.optim.Optimizer): 优化器实例，用于保存优化器状态。
        save_dir (str): 保存目录。
        save_name (str): 保存文件的基础名称。
        last_epoch (int): 当前 epoch 索引。
        last_iter (int): 当前 iter 索引。
    """
    # 仅主进程（rank 0）执行保存操作
    if torch.distributed.get_rank() != 0:
        return

    assert isinstance(teacher_model, dict), (
        "teacher_model is not a instance of dict, "
        "please call teacher_model.state_dict() to get.")
    assert isinstance(student_model, dict), (
        "student_model is not a instance of dict, "
        "please call student_model.state_dict() to get.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)

    # 保存模型
    torch.save(teacher_model, save_path + str(last_epoch) + "epoch_t.pth")
    torch.save(student_model, save_path + str(last_epoch) + "epoch_s.pth")

    # 保存优化器状态
    state_dict = optimizer.state_dict()
    state_dict['last_epoch'] = last_epoch
    state_dict['last_iter'] = last_iter
    torch.save(state_dict, save_path + str(last_epoch) + "epoch.opt.pth")
    print("Save checkpoint: {}".format(save_dir))


def save_model_info(model_info, save_path, prefix):
    """
    将模型信息保存到指定路径

    Args:
        model_info (dict): 要保存的模型信息字典
        save_path (str): 保存目录路径
        prefix (str): 文件名前缀
    """
    save_path = os.path.join(save_path, prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{prefix}.info.json'), 'w') as f:
        json.dump(model_info, f)
    print("Already save model info in {}".format(save_path))


def update_train_results(config,
                         prefix,
                         metric_info,
                         done_flag=False,
                         last_num=5,
                         ema=False):
    """
    更新训练结果记录文件（train_result.json），用于保存模型、最优模型及最近几次 checkpoint 的信息。

    Args:
        config (dict): 配置字典，需包含保存路径等信息。
        prefix (str): 当前保存模型的前缀（如 "epoch_10" 或 "best_model"）。
        metric_info (dict): 包含评估指标的字典，至少包含 "metric" 字段。
        done_flag (bool): 训练是否已完成。
        last_num (int): 保留最近 last_num 个 checkpoint 的信息。
        ema (bool): 是否使用 EMA（指数移动平均）模型。
    """
    # 仅主进程（rank 0）执行
    if torch.distributed.get_rank() != 0:
        return

    assert last_num >= 1

    # 定义训练结果文件路径
    train_results_path = os.path.join(config["save_dir"], "train_result.json")

    # 定义模型文件后缀标签
    save_model_tag = ["pth", "opt.pth", "states"]
    save_inference_tag = [
        "inference_config", "pt", "pt", "pt.info"
    ]
    if ema:
        save_model_tag.append("ema.pth")  # EMA 模型文件

    # 读取已有训练结果或初始化
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {
            "model_name": config.get("pdx_model_name", ""),
            "label_dict": "",
            "visualdl_log": "",
            "train_log": "train.log",
            "config": "config.yaml",
            "models": {}
        }
        # 初始化 last_{1..last_num} 和 best
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}

    # 更新训练完成标志
    train_results["done_flag"] = done_flag

    # 判断是否为最优模型
    if prefix == "best_model":
        train_results["models"]["best"]["score"] = metric_info["metric"]
        for tag in save_model_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, f"{prefix}.{tag.replace('ema.pth', 'ema.pth')}"
            )
        # 推理模型路径
        for i, tag in enumerate(save_inference_tag):
            filename = "inference.yml" if tag == "inference_config" else f"inference.{save_inference_tag[i] if save_inference_tag[i] != 'pt.info' else 'pt'}"
            train_results["models"]["best"][tag] = os.path.join(prefix, "inference", filename)

    else:
        # 滚动更新 last checkpoints：last_1 ← 当前, last_2 ← last_1, ..., last_n ← last_{n-1}
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][f"last_{i}"].copy()

        # 更新 last_1
        train_results["models"][f"last_1"]["score"] = metric_info["metric"]
        for tag in save_model_tag:
            train_results["models"][f"last_1"][tag] = os.path.join(prefix, f"{prefix}.{tag}")
        for i, tag in enumerate(save_inference_tag):
            filename = "inference.yml" if tag == "inference_config" else f"inference.{save_inference_tag[i] if save_inference_tag[i] != 'pt.info' else 'pt'}"
            train_results["models"][f"last_1"][tag] = os.path.join(prefix, "inference", filename)

    # 保存更新后的结果
    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)
