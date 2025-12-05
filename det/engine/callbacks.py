#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :callbacks.py
@Author :CodeCat
@Date   :2025/11/24 16:41
"""
import os
import gc
import sys
import datetime
from datetime import timedelta
import six
import copy
import json

import torch
import torch.distributed as dist

from det.utils.checkpoint import save_model, save_semi_model, save_model_info, update_train_results
from det.metrics.coco_utils import get_infer_results
from det.utils.logger import setup_logger

logger = setup_logger('det.engine')


class Callback(object):
    def __init__(self, model):
        """
        回调基类，用于在训练过程的不同阶段插入自定义逻辑。

        Args:
            model: 训练所用的模型对象，应包含配置信息 cfg
        """
        self.model = model
        # 从模型配置中获取日志输出的 rank 列表（用于多卡训练时控制哪些卡打印日志）
        log_ranks = self.model.cfg.get("log_ranks", '0')
        if isinstance(log_ranks, str):
            self.log_ranks = [int(i) for i in log_ranks.split(',')]
        elif isinstance(log_ranks, int):
            self.log_ranks = [log_ranks]
        # 初始化日志记录器
        self.logger = setup_logger('det.engine.callbacks', log_ranks=self.log_ranks)

    def on_step_begin(self, status):
        """每个训练 step 开始时调用"""
        pass

    def on_step_end(self, status):
        """每个训练 step 结束时调用"""
        pass

    def on_epoch_begin(self, status):
        """每个 epoch 开始时调用"""
        pass

    def on_epoch_end(self, status):
        """每个 epoch 结束时调用"""
        pass

    def on_train_begin(self, status):
        """整个训练开始时调用"""
        pass

    def on_train_end(self, status):
        """整个训练结束时调用"""
        pass


class ComposeCallback(object):
    def __init__(self, callbacks):
        """
        组合多个回调函数的容器类，用于统一管理训练过程中的多个回调逻辑。

        Args:
            callbacks (list): 回调函数列表，会自动过滤掉 None 值。
        """
        # 过滤掉 None 的回调
        callbacks = [c for c in list(callbacks) if c is not None]
        # 确保所有回调都是 Callback 的子类
        for c in callbacks:
            assert isinstance(
                c, Callback), "callback should be subclass of Callback"
        self._callbacks = callbacks

    def on_step_begin(self, status):
        """在每个训练 step 开始时，依次调用所有回调的 on_step_begin 方法"""
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status):
        """在每个训练 step 结束时，依次调用所有回调的 on_step_end 方法"""
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status):
        """在每个 epoch 开始时，依次调用所有回调的 on_epoch_begin 方法"""
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status):
        """在每个 epoch 结束时，依次调用所有回调的 on_epoch_end 方法"""
        for c in self._callbacks:
            c.on_epoch_end(status)

    def on_train_begin(self, status):
        """在训练开始时，依次调用所有回调的 on_train_begin 方法"""
        for c in self._callbacks:
            c.on_train_begin(status)

    def on_train_end(self, status):
        """在训练结束时，依次调用所有回调的 on_train_end 方法"""
        for c in self._callbacks:
            c.on_train_end(status)


class LogPrinter(Callback):
    def __init__(self, model):
        super(LogPrinter, self).__init__(model)
    
    def _is_main_process(self):
        """判断是否为主进程（rank 0）或单卡模式"""
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _should_log(self):
        """判断当前进程是否应该打印日志"""
        if not dist.is_available() or not dist.is_initialized():
            return True  # 单卡模式：总是打印
        
        world_size = dist.get_world_size()
        if world_size < 2:
            return True  # 单卡：打印
        
        # 多卡：只在指定 rank 打印（默认 log_ranks 应包含 0）
        current_rank = dist.get_rank()
        log_ranks = getattr(self, 'log_ranks', [0])  # 默认只在 rank 0 打印
        return current_rank in log_ranks


    def on_step_end(self, status):
        # 在单卡或指定 rank 上打印日志
        if self._should_log():
            mode = status['mode']
            if mode == 'train':
                epoch_id = status['epoch_id']
                step_id = status['step_id']
                steps_per_epoch = status['steps_per_epoch']
                training_staus = status['training_staus']
                batch_time = status['batch_time']
                data_time = status['data_time']

                epoches = self.model.cfg.epoch
                batch_size = self.model.cfg['batch_size']

                logs = training_staus.log()
                space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
                if step_id % self.model.cfg.log_iter == 0:
                    # 计算预计剩余时间
                    eta_steps = (epoches - epoch_id) * steps_per_epoch - step_id
                    eta_sec = eta_steps * batch_time.global_avg
                    eta_str = str(timedelta(seconds=int(eta_sec)))

                    # 计算每秒处理图像数
                    ips = float(batch_size) / batch_time.avg

                    # GPU内存信息（如果启用）
                    max_mem_reserved_str = ""
                    max_mem_allocated_str = ""
                    print_mem_info = self.model.cfg.get("print_mem_info", True)
                    if torch.cuda.is_available() and print_mem_info:
                        max_mem_reserved_str = f", max_mem_reserved: {torch.cuda.max_memory_reserved() // (1024 ** 2)} MB"
                        max_mem_allocated_str = f", max_mem_allocated: {torch.cuda.max_memory_allocated() // (1024 ** 2)} MB"

                    # 格式化日志输出
                    fmt = ' '.join([
                        'Epoch: [{}]',
                        '[{' + space_fmt + '}/{}]',
                        'learning_rate: {lr:.6f}',
                        '{meters}',
                        'eta: {eta}',
                        'batch_cost: {btime}',
                        'data_cost: {dtime}',
                        'ips: {ips:.4f} images/s'
                        '{max_mem_reserved_str}'
                        '{max_mem_allocated_str}'
                    ])
                    fmt = fmt.format(
                        epoch_id,
                        step_id,
                        steps_per_epoch,
                        lr=status['learning_rate'],
                        meters=logs,
                        eta=eta_str,
                        btime=str(batch_time),
                        dtime=str(data_time),
                        ips=ips,
                        max_mem_reserved_str=max_mem_reserved_str,
                        max_mem_allocated_str=max_mem_allocated_str)
                    self.logger.info(fmt)

            # 验证阶段日志
            if mode == 'eval':
                step_id = status['step_id']
                if step_id % 100 == 0:
                    self.logger.info("Eval iter: {}".format(step_id))
        

    def on_epoch_end(self, status):
        # 只在主进程（rank 0）打印汇总信息
        if self._is_main_process():
            mode = status['mode']
            if mode == 'eval':
                sample_num = status['sample_num']
                cost_time = status['cost_time']
                self.logger.info('Total sample number: {}, average FPS: {}'.format(
                    sample_num, sample_num / cost_time))


class Checkpointer(Callback):
    def __init__(self, model):
        """
        检查点回调类，用于在训练/验证结束时保存模型权重。

        Args:
            model: 训练模型实例
        """
        super(Checkpointer, self).__init__(model)
        self.best_ap = -1000.  # 最佳评估指标（初始化为极小值）
        self.save_dir = self.model.cfg.save_dir
        self.uniform_output_enabled = self.model.cfg.get("uniform_output_enabled", False)

        # 判断是否为半监督模型（如使用 EMA 的 DenseTeacher）
        if hasattr(self.model.model, 'student_model'):
            self.weight = self.model.model.student_model
        else:
            self.weight = self.model.model

    def on_epoch_end(self, status):
        """
        在每个 epoch 结束时执行检查点保存逻辑。
        """
        # Checkpointer 仅在训练期间执行
        mode = status['mode']
        epoch_id = status['epoch_id']
        weight = None
        save_name = None

        # 仅主进程（rank 0）执行保存逻辑
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            end_epoch = self.model.cfg.epoch
            save_name = str(epoch_id) if epoch_id != end_epoch - 1 else "model_final"

            if mode == 'train':
                # 按 snapshot_epoch 配置定期保存或保存最后一个 epoch
                if (epoch_id + 1) % self.model.cfg.snapshot_epoch == 0 or epoch_id == end_epoch - 1:
                    save_name = str(epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                    weight = self.weight.state_dict()

            elif mode == 'eval':
                # 遍历所有评估指标
                for metric in self.model._metrics:
                    map_res = metric.get_results()
                    eval_func = "ap"

                    # 自动识别评估指标类型
                    if 'pose3d' in map_res:
                        key = 'pose3d'
                        eval_func = "mpjpe"
                    elif 'bbox' in map_res:
                        key = 'bbox'
                    elif 'keypoint' in map_res:
                        key = 'keypoint'
                    else:
                        key = 'mask'

                    key = self.model.cfg.get('target_metrics', key)

                    # 处理空评估结果的情况（如训练步数太少）
                    if key not in map_res:
                        logger.warning("Evaluation results empty, this may be due to " \
                                       "training iterations being too few or not " \
                                       "loading the correct weights.")
                        key = ''
                        epoch_ap = 0.0
                    else:
                        epoch_ap = map_res[key][0]

                    epoch_metric = {
                        'metric': abs(epoch_ap),
                        'epoch': epoch_id + 1
                    }

                    # 保存当前 epoch 的评估指标
                    save_path = os.path.join(
                        os.path.join(self.save_dir, save_name) if self.uniform_output_enabled else self.save_dir,
                        f"{save_name}.states"
                    )
                    torch.save(epoch_metric, save_path)

                    if self.uniform_output_enabled:
                        save_model_info(epoch_metric, self.save_dir, save_name)
                        update_train_results(self.model.cfg, save_name, epoch_metric,
                                             done_flag=epoch_id + 1 == self.model.cfg.epoch,
                                             ema=self.model.use_ema)

                    # 保存最佳模型（如果启用）
                    if 'save_best_model' in status and status['save_best_model']:
                        if epoch_ap >= self.best_ap:
                            self.best_ap = epoch_ap
                            save_name = 'best_model'
                            weight = self.weight.state_dict()
                            best_metric = {
                                'metric': abs(self.best_ap),
                                'epoch': epoch_id + 1
                            }
                            save_path = os.path.join(
                                os.path.join(self.save_dir,
                                             save_name) if self.uniform_output_enabled else self.save_dir,
                                "best_model.states"
                            )
                            torch.save(best_metric, save_path)

                            if self.uniform_output_enabled:
                                save_model_info(best_metric, self.save_dir, save_name)
                                update_train_results(self.model.cfg, save_name, best_metric,
                                                     done_flag=epoch_id + 1 == self.model.cfg.epoch,
                                                     ema=self.model.use_ema)

                            logger.info("Best test {} {} is {:0.3f}.".format(
                                key, eval_func, abs(self.best_ap)))

            # 执行模型权重保存
            if weight:
                if self.model.use_ema:
                    exchange_save_model = status.get('exchange_save_model', False)
                    if not exchange_save_model:
                        # 正常保存：model + ema_model
                        save_model(
                            status['weight'],
                            self.model.optimizer,
                            os.path.join(self.save_dir, save_name) if self.uniform_output_enabled else self.save_dir,
                            save_name,
                            epoch_id + 1,
                            ema_model=weight
                        )
                        if self.uniform_output_enabled:
                            self.model.export(output_dir=os.path.join(self.save_dir, save_name, "inference"),
                                              for_fd=True)
                            gc.collect()
                    else:
                        # 交换保存：在 DenseTeacher 中，教师模型性能更高
                        student_model = status['weight']  # 普通模型
                        teacher_model = weight  # EMA 模型
                        save_model(
                            teacher_model,
                            self.model.optimizer,
                            self.save_dir,
                            save_name,
                            epoch_id + 1,
                            ema_model=student_model
                        )
                        del teacher_model
                        del student_model
                else:
                    # 无 EMA 的普通保存
                    save_model(
                        weight,
                        self.model.optimizer,
                        os.path.join(self.save_dir, save_name) if self.uniform_output_enabled else self.save_dir,
                        save_name,
                        epoch_id + 1
                    )
                    if self.uniform_output_enabled:
                        self.model.export(output_dir=os.path.join(self.save_dir, save_name, "inference"), for_fd=True)
                        gc.collect()


class WiferFaceEval(Callback):
    def __init__(self, model):
        """
        WiderFace 评估回调类，用于在评估阶段执行模型评估并立即退出。

        Args:
            model: 模型实例
        """
        super(WiferFaceEval, self).__init__(model)

    def on_epoch_begin(self, status):
        """
        在评估 epoch 开始时执行 WiderFace 评估。
        """
        assert self.model.mode == 'eval', \
            "WiferFaceEval can only be set during evaluation"

        # 遍历所有评估指标并更新（执行评估）
        for metric in self.model._metrics:
            metric.update(self.model.model)

        # 评估完成后立即退出程序（通常用于 standalone eval 模式）
        sys.exit()


class VisualDLWriter(Callback):
    """
    使用 VisualDL 记录训练过程中的标量（如损失、mAP）和图像（如可视化检测结果）。
    注意：VisualDL 仅支持 Python 3.5+。
    """

    def __init__(self, model):
        super(VisualDLWriter, self).__init__(model)

        # 检查 Python 版本
        assert six.PY3, "VisualDL requires Python >= 3.5"
        try:
            from visualdl import LogWriter
        except Exception as e:
            print('visualdl not found, please install visualdl. '
                  'For example: `pip install visualdl`.')
            raise e

        # 初始化 VisualDL 的日志写入器
        vdl_log_dir = model.cfg.get('vdl_log_dir', 'vdl_log_dir/scalar')
        self.vdl_writer = LogWriter(vdl_log_dir)

        # 初始化各模块的 step 计数器
        self.vdl_loss_step = 0  # 损失记录步数
        self.vdl_mAP_step = 0  # mAP 记录步数
        self.vdl_image_step = 0  # 图像记录步数（每帧内）
        self.vdl_image_frame = 0  # 图像帧计数器（每10张图换一帧）
    
    def _is_main_process(self):
        """判断是否为主进程（rank 0）或单卡模式"""
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def on_step_end(self, status):
        """
        在每个训练或测试 step 结束时记录数据。
        """
        
        mode = status['mode']
        # 仅主进程（rank 0）记录
        if self._is_main_process():
            if mode == 'train':
                # 记录训练损失
                training_staus = status['training_staus']
                for loss_name, loss_value in training_staus.get().items():
                    self.vdl_writer.add_scalar(loss_name, loss_value,
                                               self.vdl_loss_step)
                self.vdl_loss_step += 1

            elif mode == 'test':
                # 记录原始图像和检测结果图像
                ori_image = status['original_image']
                result_image = status['result_image']

                # VisualDL 的 add_image 要求图像为 (H, W, C) 形式且值在 [0, 255]
                self.vdl_writer.add_image(
                    "original/frame_{}".format(self.vdl_image_frame),
                    ori_image,
                    self.vdl_image_step
                )
                self.vdl_writer.add_image(
                    "result/frame_{}".format(self.vdl_image_frame),
                    result_image,
                    self.vdl_image_step
                )
                self.vdl_image_step += 1

                # 每帧最多显示 10 张图像，超过则新建一帧
                if self.vdl_image_step % 10 == 0:
                    self.vdl_image_step = 0
                    self.vdl_image_frame += 1

    def on_epoch_end(self, status):
        """
        在每个 epoch 结束时（评估阶段）记录 mAP 等指标。
        """
        mode = status['mode']
        if self._is_main_process():
            if mode == 'eval':
                # 遍历所有评估指标并记录 mAP
                for metric in self.model._metrics:
                    for key, map_value in metric.get_results().items():
                        # map_value 通常为列表，取第一个值（如 [mAP]）
                        self.vdl_writer.add_scalar(
                            "{}-mAP".format(key),
                            map_value[0],
                            self.vdl_mAP_step
                        )
                self.vdl_mAP_step += 1


class WandbCallback(Callback):
    def __init__(self, model):
        super(WandbCallback, self).__init__(model)

        try:
            import wandb
            self.wandb = wandb
        except Exception as e:
            logger.error('wandb not found, please install wandb. '
                         'Use: `pip install wandb`.')
            raise e

        # 解析 W&B 配置
        self.wandb_params = model.cfg.get('wandb', None)
        self.save_dir = self.model.cfg.save_dir
        if self.wandb_params is None:
            self.wandb_params = {}
        for k, v in model.cfg.items():
            if k.startswith("wandb_"):
                self.wandb_params.update({k.lstrip("wandb_"): v})

        self._run = None
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            _ = self.run
            self.run.config.update(self.model.cfg)
            self.run.define_metric("epoch")
            self.run.define_metric("eval/*", step_metric="epoch")

        self.best_ap = -1000.
        self.fps = []

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                logger.info(
                    "There is an ongoing wandb run which will be used "
                    "for logging. Please use `wandb.finish()` to end that "
                    "if the behaviour is not intended.")
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self.wandb_params)
        return self._run

    def save_model(self,
                   optimizer,
                   save_dir,
                   save_name,
                   last_epoch,
                   ema_model=False,
                   ap=None,
                   fps=None,
                   tags=None):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            model_path = os.path.join(save_dir, save_name)
            metadata = {"last_epoch": last_epoch}
            if ap is not None:
                metadata["ap"] = ap
            if fps is not None:
                metadata["fps"] = fps

            model_artifact = self.wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
                metadata=metadata
            )
            model_artifact.add_file(model_path + ".pth", name="model")

            if ema_model:
                ema_artifact = self.wandb.Artifact(
                    name=f"ema_model-{self.run.id}",
                    type="model",
                    metadata=metadata
                )
                ema_artifact.add_file(model_path + ".ema.pth", name="model_ema")
                self.run.log_artifact(ema_artifact, aliases=tags)

            self.run.log_artifact(model_artifact, aliases=tags)

    def on_step_end(self, status):
        mode = status['mode']
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if mode == 'train':
                training_status = status['training_staus'].get()
                training_status = {k: float(v) for k, v in training_status.items()}

                batch_time = status['batch_time']
                data_time = status['data_time']
                batch_size = self.model.cfg['{}Reader'.format(mode.capitalize())]['batch_size']

                ips = float(batch_size) / float(batch_time.avg)
                data_cost = float(data_time.avg)
                batch_cost = float(batch_time.avg)

                metrics = {"train/" + k: v for k, v in training_status.items()}
                metrics.update({
                    "train/ips": ips,
                    "train/data_cost": data_cost,
                    "train/batch_cost": batch_cost
                })

                self.fps.append(ips)
                self.run.log(metrics)

    def on_epoch_end(self, status):
        mode = status['mode']
        epoch_id = status['epoch_id']
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if mode == 'train':
                if self.fps:
                    fps = sum(self.fps) / len(self.fps)
                    self.fps = []
                else:
                    fps = 0.0

                end_epoch = self.model.cfg.epoch
                if (epoch_id + 1) % self.model.cfg.snapshot_epoch == 0 or epoch_id == end_epoch - 1:
                    save_name = str(epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                    tags = ["latest", f"epoch_{epoch_id}"]
                    self.save_model(
                        self.model.optimizer,
                        self.save_dir,
                        save_name,
                        epoch_id + 1,
                        ema_model=self.model.use_ema,
                        fps=fps,
                        tags=tags)

            if mode == 'eval':
                sample_num = status['sample_num']
                cost_time = status['cost_time']
                fps = sample_num / cost_time

                merged_dict = {"epoch": epoch_id, "eval/fps": fps}
                for metric in self.model._metrics:
                    results = metric.get_results()
                    for key, value in results.items():
                        merged_dict[f"eval/{key}-mAP"] = float(value[0])

                self.run.log(merged_dict)

                if 'save_best_model' in status and status['save_best_model']:
                    for metric in self.model._metrics:
                        map_res = metric.get_results()
                        if 'bbox' in map_res:
                            key = 'bbox'
                        elif 'pose3d' in map_res:
                            key = 'pose3d'
                        elif 'keypoint' in map_res:
                            key = 'keypoint'
                        else:
                            key = 'mask'

                        if key not in map_res:
                            logger.warning("Evaluation results empty...")
                            return

                        current_ap = map_res[key][0]
                        if current_ap >= self.best_ap:
                            self.best_ap = current_ap
                            save_name = 'best_model'
                            tags = ["best", f"epoch_{epoch_id}"]
                            self.save_model(
                                self.model.optimizer,
                                self.save_dir,
                                save_name,
                                last_epoch=epoch_id + 1,
                                ema_model=self.model.use_ema,
                                ap=abs(self.best_ap),
                                fps=fps,
                                tags=tags)

    def on_train_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            self.run.finish()


class SniperProposalsGenerator(Callback):
    def __init__(self, model):
        """
        SNIPER 提议生成器回调类。
        在训练结束后，使用训练好的模型在切片（chips）级别上推理，
        并聚合结果生成用于下一轮半监督训练的 proposal 文件（JSON 格式）。
        """
        super(SniperProposalsGenerator, self).__init__(model)
        ori_dataset = self.model.dataset
        self.dataset = self._create_new_dataset(ori_dataset)
        self.loader = self.model.loader
        self.cfg = self.model.cfg
        self.infer_model = self.model.model

    def _create_new_dataset(self, ori_dataset):
        """
        创建用于推理的 SNIPER 切片数据集。
        - 深拷贝原始数据集
        - 初始化 AnnoCropper
        - 生成切片级 roidbs（仅用于推理，不含 gt）
        """
        dataset = copy.deepcopy(ori_dataset)
        # 初始化切片生成器
        dataset.init_anno_cropper()
        # 获取原始 roidbs（未切片）
        ori_roidbs = dataset.get_ori_roidbs()
        # 生成推理用的切片 roidbs
        roidbs = dataset.anno_cropper.crop_infer_anno_records(ori_roidbs)
        # 替换数据集的 roidbs
        dataset.set_roidbs(roidbs)
        return dataset

    def _eval_with_loader(self, loader):
        """
        使用给定的数据加载器进行推理，返回原始输出结果列表。
        """
        results = []
        self.infer_model.eval()
        with torch.no_grad():
            for step_id, data in enumerate(loader):
                outs = self.infer_model(data)
                # 附加上下文信息（用于后续后处理）
                for key in ['im_shape', 'scale_factor', 'im_id']:
                    outs[key] = data[key]
                # 将张量转为 numpy 数组（兼容后续处理）
                for key, value in outs.items():
                    if isinstance(value, torch.Tensor):
                        outs[key] = value.cpu().numpy()
                results.append(outs)
        return results

    def on_train_end(self, status):
        """
        在训练结束时执行：
        1. 替换数据加载器的数据集为切片推理数据集
        2. 执行推理
        3. 聚合切片检测结果到原始图像级别
        4. 保存为 proposals.json 文件
        """
        # 替换为切片数据集
        self.loader.dataset = self.dataset
        # 执行推理
        results = self._eval_with_loader(self.loader)
        # 聚合切片检测结果到原始图像
        results = self.dataset.anno_cropper.aggregate_chips_detections(results)

        # 构建 proposals 列表（COCO 格式）
        proposals = []
        clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()}
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            start = 0
            for i, im_id in enumerate(outs['im_id']):
                bbox_num = outs['bbox_num']
                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] if 'bbox' in batch_res else None
                if bbox_res:
                    proposals.extend(bbox_res)
                start = end

        # 保存 proposal 文件
        logger.info("save proposals in {}".format(self.cfg.proposals_path))
        with open(self.cfg.proposals_path, 'w') as f:
            json.dump(proposals, f)


class SemiLogPrinter(LogPrinter):
    def __init__(self, model):
        """
        半监督训练专用的日志打印回调，相比普通 LogPrinter 增加了全局 iter_id 显示。
        """
        super(SemiLogPrinter, self).__init__(model)

    def on_step_end(self, status):
        """
        在每个训练 step 结束时打印日志（仅主进程执行）。
        """
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            mode = status['mode']
            if mode == 'train':
                epoch_id = status['epoch_id']
                step_id = status['step_id']      # 当前 epoch 内的 step
                iter_id = status['iter_id']      # 全局 iteration（跨 epoch）
                steps_per_epoch = status['steps_per_epoch']
                training_staus = status['training_staus']
                batch_time = status['batch_time']
                data_time = status['data_time']

                epoches = self.model.cfg.epoch
                batch_size = self.model.cfg['{}Reader'.format(mode.capitalize())]['batch_size']
                total_iters = epoches * steps_per_epoch  # 总训练迭代次数

                logs = training_staus.log()
                # 格式化字符串：对齐总 iter 数字宽度
                iter_space_fmt = ':' + str(len(str(total_iters))) + 'd'
                space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'

                if step_id % self.model.cfg.log_iter == 0:
                    # 预计剩余时间（基于全局 avg batch time）
                    eta_steps = (epoches - epoch_id) * steps_per_epoch - step_id
                    eta_sec = eta_steps * batch_time.global_avg
                    eta_str = str(timedelta(seconds=int(eta_sec)))

                    # 每秒处理图像数（images per second）
                    ips = float(batch_size) / batch_time.avg

                    # 构建日志格式
                    fmt = ' '.join([
                        '{' + iter_space_fmt + '}/{} iters',
                        'Epoch: [{}]',
                        '[{' + space_fmt + '}/{}]',
                        'learning_rate: {lr:.6f}',
                        '{meters}',
                        'eta: {eta}',
                        'batch_cost: {btime}',
                        'data_cost: {dtime}',
                        'ips: {ips:.4f} images/s',
                    ])
                    fmt = fmt.format(
                        iter_id,
                        total_iters,
                        epoch_id,
                        step_id,
                        steps_per_epoch,
                        lr=status['learning_rate'],
                        meters=logs,
                        eta=eta_str,
                        btime=str(batch_time),
                        dtime=str(data_time),
                        ips=ips
                    )
                    logger.info(fmt)

            if mode == 'eval':
                step_id = status['step_id']
                if step_id % 100 == 0:
                    logger.info("Eval iter: {}".format(step_id))


class SemiCheckpointer(Checkpointer):
    def __init__(self, model):
        """
        半监督训练专用检查点回调类，用于保存 Teacher 和 Student 双模型。
        要求 model.model 同时包含 'teacher' 和 'student' 子模块。
        """
        super(SemiCheckpointer, self).__init__(model)
        cfg = self.model.cfg
        self.best_ap = 0.0  # 最佳评估指标（用于保存 best_model）
        # 构建保存路径：save_dir/filename
        self.save_dir = os.path.join(self.model.cfg.save_dir, self.model.cfg.filename)

        # 验证模型结构是否符合半监督要求
        if hasattr(self.model.model, 'student') and hasattr(self.model.model, 'teacher'):
            self.weight = (self.model.model.teacher, self.model.model.student)
        elif hasattr(self.model.model, 'student') or hasattr(self.model.model, 'teacher'):
            raise AttributeError("Model has only one of 'student' or 'teacher', but both are required.")
        else:
            raise AttributeError("Model must have both 'student' and 'teacher' attributes.")

    def every_n_iters(self, iter_id, n):
        """
        判断当前迭代是否是第 n 次迭代的倍数。

        Args:
            iter_id (int): 当前迭代步数（从 0 开始）
            n (int): 间隔步数

        Returns:
            bool: 是否满足间隔条件
        """
        return (iter_id + 1) % n == 0 if n > 0 else False

    def on_step_end(self, status):
        """
        在每个训练 step 结束时，按 save_interval 保存最新模型。
        """
        mode = status['mode']
        save_interval = status['save_interval']
        iter_id = status['iter_id']
        epoch_id = status['epoch_id']

        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if self.every_n_iters(iter_id, save_interval) and mode == 'train':
                save_name = "last_epoch"
                t_weight = self.weight[0].state_dict()  # Teacher 模型权重
                s_weight = self.weight[1].state_dict()  # Student 模型权重

                save_semi_model(
                    t_weight, s_weight,
                    self.model.optimizer,
                    self.save_dir,
                    save_name,
                    epoch_id + 1,
                    iter_id + 1
                )

    def on_epoch_end(self, status):
        """
        在每个评估 epoch 结束时，判断是否保存最佳模型（基于 AP 指标）。
        """
        mode = status['mode']
        eval_interval = status['eval_interval']
        iter_id = status['iter_id']
        epoch_id = status['epoch_id']

        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if self.every_n_iters(iter_id, eval_interval) and mode == 'eval':
                if 'save_best_model' in status and status['save_best_model']:
                    current_ap = None
                    key = None

                    # 获取评估结果
                    for metric in self.model._metrics:
                        map_res = metric.get_results()
                        if 'bbox' in map_res:
                            key = 'bbox'
                        elif 'keypoint' in map_res:
                            key = 'keypoint'
                        else:
                            key = 'mask'

                        if key not in map_res:
                            logger.warning(
                                "Evaluation results empty, this may be due to "
                                "training iterations being too few or not "
                                "loading the correct weights."
                            )
                            return

                        current_ap = map_res[key][0]
                        break  # 假设只有一个主要指标

                    # 如果当前 AP 更优，则保存为 best_model
                    if current_ap is not None and current_ap > self.best_ap:
                        self.best_ap = current_ap
                        save_name = 'best_model'
                        t_weight = self.weight[0].state_dict()
                        s_weight = self.weight[1].state_dict()

                        logger.info("Best teacher test {} AP is {:0.3f}.".format(key, self.best_ap))

                        save_semi_model(
                            t_weight, s_weight,
                            self.model.optimizer,
                            self.save_dir,
                            save_name,
                            epoch_id + 1,
                            iter_id + 1
                        )