#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :trainer.py
@Author :CodeCat
@Date   :2025/11/17 10:14
"""
import os
import sys
import copy
import time
import yaml
import shutil
from tqdm import tqdm
from hydra.utils import instantiate
import numpy as np
import typing
from packaging import version
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from det.optimizer.ema import ModelEMA
from det.utils.checkpoint import load_pretrain_weight, load_weight, convert_to_dict
from det.utils.visualizer import visualize_results, save_result
from det.metrics.coco_utils import get_infer_results
from det.metrics.keypoint_metrics import KeyPointTopDownCOCOEval, KeyPointTopDownCOCOWholeBadyHandEval, \
    KeyPointTopDownMPIIEval
from det.metrics.pose3d_metrics import Pose3DEval
from det.metrics.metrics import COCOMetric, VOCMetric, WiderFaceMetric, RBoxMetric, SNIPERCOCOMetric
from det.metrics.mot_metrics import JDEDetMetric
from det.metrics.culane_metrics import CULaneMetric
from det.data.source.sniper_coco import SniperCOCODataSet
from det.data.source.category import get_categories
from det.utils import stats as stats
from det.utils.fuse_utils import fuse_conv_bn
from det.utils import profiler
from det.modeling.post_process import multiclass_nms
from det.modeling.lane_utils import imshow_lanes
from det.engine.callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, \
    SniperProposalsGenerator, WandbCallback, SemiCheckpointer, SemiLogPrinter
from det.engine.naive_sync_bn import convert_bn, convert_syncbn
from det.utils.logger import setup_logger

logger = setup_logger("det.engine")

__all__ = ['Trainer']

MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg.copy()
        assert mode.lower() in ['train', 'eval', 'test'], \
            "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)
        self.use_master_grad = self.cfg.get('use_master_grad', False)
        self.uniform_output_enabled = self.cfg.get('uniform_output_enabled', False)
        log_ranks = self.cfg.get('log_ranks', '0')

        if dist.is_initialized():
            self._nranks = dist.get_world_size()
            self._local_rank = dist.get_rank()
            self._device = torch.device(f'cuda')
        else:
            self._nranks = 1
            self._local_rank = 0
            self._device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(log_ranks, str):
            self.log_ranks = [int(rank) for rank in log_ranks.split(',')]
        elif isinstance(log_ranks, int):
            self.log_ranks = [log_ranks]
        train_results_path = os.path.abspath(os.path.join(self.cfg.save_dir, "train_result.json"))
        if self.uniform_output_enabled:
            if os.path.exists(train_results_path) and self.mode == 'train':
                try:
                    os.remove(train_results_path)
                except:
                    pass
            if not os.path.exists(self.cfg.save_dir):
                os.makedirs(self.cfg.save_dir)
            with open(os.path.join(self.cfg.save_dir, "config.yaml"), 'w') as f:
                config_dict = convert_to_dict(self.cfg)
                config_dict = {k: v for k, v in config_dict.items() if v != {}}
                yaml.dump(config_dict, f)

        if self.mode == 'train':
            self.dataset = instantiate(self.cfg.train_dataset)
            self.loader = instantiate(self.cfg.train_loader)(self.dataset, cfg.worker_num)

        self.model = instantiate(self.cfg.model)

        if self.mode == 'eval':
            self._eval_batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(self.dataset),
                self.cfg.eval_loader.batch_size,
                drop_last=False,
            )
            if cfg.metric == 'VOC':
                self.cfg.eval_loader.collate_batch = False
            self.loader = instantiate(self.cfg.eval_loader)(self.dataset, cfg.worker_num, self._eval_batch_sampler)

        print_params = self.cfg.get('print_params', False)
        if print_params:
            params = sum([
                p.numel() for n, p in self.model.named_parameters()
                if all(x not in n for x in ['_mean', '_variance', 'aux_', 'running_mean', 'running_var'])
            ])
            logger.info('Model Params : {:.2f} M.'.format(params / 1e6))

        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch size, please set samller batch size in train loader.")
            self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
            self.lr_scheduler = instantiate(self.cfg.lr_scheduler, optimizer=self.optimizer)

        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            ema_filter_no_grad = self.cfg.get('ema_filter_no_grad', False)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad
            )

        self.status = {}
        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        self._init_callbacks()

        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        if self.mode == 'train':
            if self.cfg.get('ssod_method', False) and self.cfg['ssod_method'] == 'Semi_RTDETR':
                self._callbacks = [SemiLogPrinter(self), SemiCheckpointer(self)]
            else:
                self._callbacks = [LogPrinter(self), Checkpointer(self)]

            if self.cfg.get('use_vdl', False):
                self._callbacks.append(VisualDLWriter(self))
            if self.cfg.get('save_proposals', False):
                self._callbacks.append(SniperProposalsGenerator(self))
            if self.cfg.get('wandb', False) or 'wandb' in self.cfg:
                self._callbacks.append(WandbCallback(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg.get('use_vdl', False):
            self._callbacks = [VisualDLWriter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg.get('classwise', False)
        if self.cfg.metric == 'COCO' or self.cfg.metric == 'SNIPERCOCO':
            bias = 1 if self.cfg.get('bias', False) else 0
            output_eval = self.cfg.get('output_eval', None)
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            clsid2catid = {
                v: k for k, v in self.dataset.catid2clsid.items()
            } if self.mode == 'eval' else None
            save_threshold = self.cfg.get('save_threshold', 0)

            if self.mode == 'train' and validate:
                eval_dataset = self.cfg.eval_dataset
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()

            IoUType = self.cfg.get('IoUType', 'bbox')
            if self.cfg.metric == 'COCO':
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IoUType=IoUType,
                        save_prediction_only=save_prediction_only,
                        save_threshold=save_threshold
                    )
                ]
            elif self.cfg.metric == 'SNIPERCOCO':
                self._metrics = [
                    SNIPERCOCOMetric(
                        anno_file=anno_file,
                        dataset=dataset,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IoUType=IoUType,
                        save_prediction_only=save_prediction_only,
                        save_threshold=save_threshold
                    )
                ]
        elif self.cfg.metric == 'RBOX':
            bias = self.cfg.get('bias', 0)
            output_eval = self.cfg.get('output_eval', None)
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            imid2path = self.cfg.get('imid2path', None)

            anno_file = self.dataset.get_anno()
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg.eval_dataset
                anno_file = eval_dataset.get_anno()

            self._metrics = [
                RBoxMetric(
                    anno_file=anno_file,
                    classwise=classwise,
                    output_eval=output_eval,
                    bias=bias,
                    imid2path=imid2path,
                    save_prediction_only=save_prediction_only
                )
            ]
        elif self.cfg.metric == 'VOC':
            output_eval = self.cfg.get('output_eval', None)
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise,
                    output_eval=output_eval,
                    save_prediction_only=save_prediction_only
                )
            ]
        elif self.cfg.metric == 'WiderFace':
            self._metrics = [
                WiderFaceMetric()
            ]
        elif self.cfg.metric == 'KeyPointTopDownCOCOEval':
            eval_dataset = self.cfg.eval_dataset
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOEval(
                    anno_file=anno_file,
                    num_samples=len(eval_dataset),
                    num_joints=self.cfg.num_joints,
                    output_eval=self.cfg.save_dir,
                    save_prediction_only=save_prediction_only
                )
            ]
        elif self.cfg.metric == 'KeyPointTopDownCOCOWholeBadyHandEval':
            eval_dataset = self.cfg.eval_dataset
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOWholeBadyHandEval(
                    anno_file=anno_file,
                    num_samples=len(eval_dataset),
                    num_joints=self.cfg.num_joints,
                    output_eval=self.cfg.save_dir,
                    save_prediction_only=save_prediction_only
                )
            ]
        elif self.cfg.metric == 'KeyPointTopDownMPIIEval':
            eval_dataset = self.cfg.eval_dataset
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownMPIIEval(
                    anno_file=anno_file,
                    num_samples=len(eval_dataset),
                    num_joints=self.cfg.num_joints,
                    output_eval=self.cfg.save_dir,
                    save_prediction_only=save_prediction_only
                )
            ]
        elif self.cfg.metric == 'Pose3DEval':
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                Pose3DEval(
                    output_eval=self.cfg.save_dir,
                    save_prediction_only=save_prediction_only
                )
            ]
        elif self.cfg.metric == 'MOTDet':
            self._metrics = [
                JDEDetMetric()
            ]
        elif self.cfg.metric == 'CULaneMetric':
            output_eval = self.cfg.get('output_eval', None)
            self._metrics = [
                CULaneMetric(
                    cfg=self.cfg,
                    output_eval=output_eval,
                    split=self.dataset.split,
                    dataset_dir=self.cfg.dataset_dir
                )
            ]
        else:
            logger.warning("Metric not supported for metric type {}".format(self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def load_weights(self, weights, ARSL_eval=False):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights, ARSL_eval=ARSL_eval)
        logger.debug("Load weights {} to start training.".format(weights))

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            if self.model.reid and reid_weights:
                load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights)

    def resume_weights(self, weights):
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights, self.optimizer)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer, self.ema if self.use_ema else None)

        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode."
        Init_mark = False
        if validate:
            self.cfg.eval_dataset = instantiate(self.cfg.eval_dataset)

        model = self.model.to(self._device)
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler(
                enabled=self.cfg.use_gpu,
                init_scale=self.cfg.get('init_loss_scaling', 1024)
            )
        if self._nranks > 1:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=find_unused_parameters
            )

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}'
        )
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}'
        )
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = instantiate(self.cfg.train_loader)(self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)

        use_fused_allreduce_gradients = self.cfg.get('use_fused_allreduce_gradients', False)

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                def deep_pin(blob, non_blocking=True):
                    if isinstance(blob, torch.Tensor):
                        return blob.cuda(non_blocking=non_blocking)
                    elif isinstance(blob, dict):
                        return {k: deep_pin(v, non_blocking) for k, v in blob.items()}
                    elif isinstance(blob, (list, tuple)):
                        return type(blob)([deep_pin(x, non_blocking) for x in blob])
                    return blob

                if torch.cuda.is_available():
                    data = deep_pin(data, non_blocking=True)

                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id
                if self.cfg.get('to_static', False) and 'image_file' in data.keys():
                    data.pop('image_file')

                if self.use_amp:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel) and use_fused_allreduce_gradients:
                        with model.no_sync():
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                outputs = model(data)
                                loss = outputs['loss']
                            scaler.scale(loss).backward()

                            for param in model.parameters():
                                if param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                    param.grad.div_(self._nranks)
                    else:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = model(data)
                            loss = outputs['loss']
                        scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel) and use_fused_allreduce_gradients:
                        with model.no_sync():
                            outputs = model(data)
                            loss = outputs['loss']
                            loss.backward()

                        for param in model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                param.grad.div_(self._nranks)
                    else:
                        outputs = model(data)
                        loss = outputs['loss']
                        loss.backward()

                    self.optimizer.step()

                cur_lr = self.optimizer.param_groups[0]['lr']
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.status['learning_rate'] = cur_lr

                if self._nranks < 2 or self._local_rank in self.log_ranks:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                if self.use_ema:
                    self.ema.update()
                iter_tic = time.time()

            is_snapshot = (self._nranks < 2 or (self._local_rank == 0 or self.cfg.metric == 'Pose3DEval')) and (
                    (epoch_id + 1) % self.cfg.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1)
            if is_snapshot and self.use_ema:
                weight = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self.ema.apply())
                self.status['weight'] = weight

            self._compose_callback.on_epoch_end(self.status)

            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    self._eval_dataset = self.cfg.eval_dataset
                    self._eval_batch_sampler = torch.utils.data.BatchSampler(
                        torch.utils.data.SequentialSampler(self._eval_dataset),
                        self.cfg.eval_loader['batch_size'],
                        drop_last=False,
                    )
                    if self.cfg.metric == 'VOC':
                        self.cfg.eval_loader['collate_batch'] = False
                    if self.cfg.metric == 'Pose3DEval':
                        self._eval_loader = instantiate(self.cfg.eval_loader)(self._eval_dataset, self.cfg.worker_num)
                    else:
                        self._eval_loader = instantiate(self.cfg.eval_loader)(self._eval_dataset, self.cfg.worker_num,
                                                                              self._eval_batch_sampler)
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with torch.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

            if is_snapshot and self.use_ema:
                self.model.load_state_dict(weight)
                self.status.pop('weight')

        self._compose_callback.on_train_end(self.status)

    def _eval_with_loader(self, loader):
        """
            使用给定的数据加载器进行模型评估。

            Args:
                loader: 评估数据加载器
        """
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'

        self.model.eval()
        if self.cfg.get('print_flops', False):
            self._flops(loader)

        for setp_id, data in enumerate(loader):
            self.status['step_id'] = setp_id
            self._compose_callback.on_step_begin(self.status)
            if self.use_amp:
                with torch.cuda.amp.autocast(
                        enabled=self.cfg.use_gpu,
                        dtype=torch.float16):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            for metric in self._metrics:
                metric.update_state(data, outs)

            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        for metric in self._metrics:
            metric.accumulate()
            metric.log()

        self._compose_callback.on_epoch_end(self.status)
        self._reset_metrics()

    def evaluate(self):
        if self._nranks > 1:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            self.model = self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=find_unused_parameters
            )
        with torch.no_grad():
            self._eval_with_loader(self.loader)

    def _eval_with_loader_slice(self,
                                loader,
                                slice_size=[640, 640],
                                overlap_ratio=[0.25, 0.25],
                                combine_method='nms',
                                match_threshold=0.6,
                                match_metric='iou'):
        """
            对切片检测结果进行评估，适用于大图滑窗检测后合并预测框的场景。

            Args:
                loader: 数据加载器，每次返回一个图像切片及其元信息
                slice_size (list): 切片大小 [H, W]
                overlap_ratio (list): 切片重叠比例 [H_ratio, W_ratio]
                combine_method (str): 合并方式，'nms' 或 'concat'
                match_threshold (float): NMS 匹配阈值
                match_metric (str): 匹配度量方式，如 'iou'
        """
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            self._flops(loader)

        merged_bboxes = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            if self.use_amp:
                with torch.cuda.amp.autocast(
                        enabled=self.cfg.use_gpu,
                        dtype=torch.float16):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount
            merged_bboxes.append(outs['bbox'])

            if data['is_last'] > 0:
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        bboxs=np.concatenate(merged_bboxes),
                        num_classes=self.cfg.num_classes,
                        match_threshold=match_threshold,
                        match_metric=match_metric
                    )
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxes)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )

                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array([len(merged_results['bbox'])])

                merged_bboxes = []
                data['im_id'] = data['ori_im_id']
                for metric in self._metrics:
                    metric.update_state(data, merged_results)

                if isinstance(data, typing.Sequence):
                    sample_num += data[0]['im_id'].numpy().shape[0]
                else:
                    sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        for metric in self._metrics:
            metric.accumulate()
            metric.log()

        self._compose_callback.on_epoch_end(self.status)
        self._reset_metrics()

    def evaluate_slice(self,
                       slice_size=[640, 640],
                       overlap_ratio=[0.25, 0.25],
                       combine_method='nms',
                       match_threshold=0.6,
                       match_metric='iou'):
        with torch.no_grad():
            self._eval_with_loader_slice(self.loader, slice_size, overlap_ratio, combine_method, match_threshold,
                                         match_metric)

    def slice_predict(self,
                      images,
                      slice_size=[640, 640],
                      overlap_ratio=[0.25, 0.25],
                      combine_method='nms',
                      match_threshold=0.6,
                      match_metric='iou',
                      draw_threshold=0.5,
                      output_dir='output',
                      save_results=False,
                      visualize=True):
        """
            对大图像进行切片预测（sliding window），再合并检测结果。

            Args:
                images (list[str]): 图像路径列表
                slice_size (list): 切片大小 [H, W]
                overlap_ratio (list): 切片重叠比例 [H_ratio, W_ratio]
                combine_method (str): 合并方式 'nms' 或 'concat'
                match_threshold (float): NMS 阈值
                match_metric (str): 匹配度量 'iou' 或 'ios'
                draw_threshold (float): 可视化置信度阈值
                output_dir (str): 输出目录
                save_results (bool): 是否保存预测结果
                visualize (bool): 是否可视化结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_slice_images(images, slice_size, overlap_ratio)
        loader = instantiate(self.cfg.test_loader)(self.dataset, 0)
        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            output_eval = self.cfg.get('output_eval', None)

            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')
            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file
        )

        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            self._flops(loader)

        results = []
        merged_bboxes = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            outs = self.model(data)

            outs['bbox'] = outs['bbox'].cpu().numpy()
            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount.cpu().numpy()
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount.cpu().numpy()
            merged_bboxes.append(outs['bbox'])

            if data['is_last'] > 0:
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        bboxs=np.concatenate(merged_bboxes),
                        num_classes=self.cfg.num_classes,
                        match_threshold=match_threshold,
                        match_metric=match_metric
                    )
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxes)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )

                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array([len(merged_results['bbox'])])

                merged_bboxes = []
                data['im_id'] = data['ori_im_id']

                for _m in metrics:
                    _m.update_state(data, merged_results)

                for key in ['im_shape', 'scale_factor', 'im_id']:
                    if isinstance(data, typing.Sequence):
                        merged_results[key] = data[0][key]
                    else:
                        merged_results[key] = data[key]

                for key, value in merged_results.items():
                    if hasattr(value, 'numpy'):
                        merged_results[key] = value.numpy()

                results.append(merged_results)

            for _m in metrics:
                _m.accumulate()
                _m.reset()

            if visualize:
                for outs in results:
                    batch_res = get_infer_results(outs, clsid2catid)
                    bbox_num = outs['bbox_num']

                    start = 0
                    for i, im_id in enumerate(outs['im_id']):
                        image_path = imid2path[int(im_id)]
                        image = Image.open(image_path).convert('RGB')
                        image = ImageOps.exif_transpose(image)
                        self.status['original_image'] = np.array(image.copy())

                        end = start + bbox_num[i]
                        bbox_res = batch_res['bbox'][start:end] if 'bbox' in batch_res else None
                        mask_res = batch_res['mask'][start:end] if 'mask' in batch_res else None
                        segm_res = batch_res['segm'][start:end] if 'segm' in batch_res else None
                        keypoint_res = batch_res['keypoint'][start:end] if 'keypoint' in batch_res else None
                        pose3d_res = batch_res['pose3d'][start:end] if 'pose3d' in batch_res else None
                        image = visualize_results(
                            image=image,
                            bbox_res=bbox_res,
                            mask_res=mask_res,
                            segm_res=segm_res,
                            keypoint_res=keypoint_res,
                            pose3d_res=pose3d_res,
                            im_id=int(im_id),
                            catid2name=catid2name,
                            threshold=draw_threshold
                        )
                        self.status['result_image'] = np.array(image.copy())
                        if self._compose_callback:
                            self._compose_callback.on_step_end(self.status)

                        save_name = self._get_save_image_name(output_dir, image_path)
                        logger.info("Detection bbox results save in {}".format(save_name))
                        image.save(save_name, quality=95)
                        start = end
        return results

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True,
                save_threshold=0,
                do_eval=False):
        """
            对输入图像进行推理预测，支持结果保存、可视化和评估模式。

            Args:
                images (list[str]): 待预测的图像路径列表
                draw_threshold (float): 绘制检测框的置信度阈值
                output_dir (str): 输出目录（用于保存可视化图像或预测结果）
                save_results (bool): 是否保存预测结果（用于后续评估）
                visualize (bool): 是否可视化检测结果
                save_threshold (float): 保存预测的置信度阈值（若 do_eval=True 则设为 0）
                do_eval (bool): 是否以评估模式运行（影响阈值和输出格式）
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if do_eval:
            save_threshold = 0.0

        self.dataset.set_images(images)
        loader = instantiate(self.cfg.test_loader)(self.dataset, 0)
        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            output_eval = self.cfg.get('output_eval', None)

            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self.cfg['save_threshold'] = save_threshold
            self._init_metrics()

            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')
            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file
        )

        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            self._flops(loader)

        results = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            if hasattr(self.model, 'modelTeacher'):
                outs = self.model.modelTeacher(data)
            else:
                outs = self.model(data)

            for _m in metrics:
                _m.update_state(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]

            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()

            results.append(outs)

        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image=image,
                        bbox_res=bbox_res,
                        mask_res=mask_res,
                        segm_res=segm_res,
                        keypoint_res=keypoint_res,
                        pose3d_res=pose3d_res,
                        im_id=int(im_id),
                        catid2name=catid2name,
                        threshold=draw_threshold
                    )
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)

                    save_name = self._get_save_image_name(output_dir, image_path)
                    logger.info("Detection bbox results save in {}".format(save_name))
                    image.save(save_name, quality=95)
                    start = end
        return results

    @staticmethod
    def _get_save_image_name(output_dir, image_path):
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    @staticmethod
    def parse_mot_images(cfg):
        import glob
        dataset_dir = cfg['EvalMOTDataset'].dataset_dir
        data_root = cfg['EvalMOTDataset'].data_root
        data_root = '{}/{}'.format(dataset_dir, data_root)
        seqs = os.listdir(data_root)
        seqs.sort()
        all_images = []
        for seq in seqs:
            infer_dir = os.path.join(data_root, seq)
            assert infer_dir is None or os.path.isdir(infer_dir), "{} is not a directory".format(infer_dir)
            images = set()
            exts = ['jpg', 'jpeg', 'png', 'bmp']
            exts += [ext.upper() for ext in exts]
            for ext in exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            images.sort()
            assert len(images) > 0, "No images found in {}".format(infer_dir)
            all_images.extend(images)
            logger.info("Found {} inference images in {}.".format(len(images), infer_dir))
        return all_images

    def predict_culane(self,
                       images,
                       output_dir='output',
                       save_results=False,
                       visualize=True):
        """
           对 CULane 数据集图像进行车道线预测，并支持保存结果和可视化。

           Args:
               images (list[str]): 待预测的图像路径列表
               output_dir (str): 输出目录，用于保存可视化结果或预测文件
               save_results (bool): 是否保存预测结果（用于后续评估）
               visualize (bool): 是否可视化车道线并保存图像
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = instantiate(self.cfg.test_loader)(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            output_eval = self.cfg.get('output_eval', None)

            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')
            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            self._flops(loader)
        results = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            outs = self.model(data)

            for _m in metrics:
                _m.update_state(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]

            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()

            results.append(outs)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            import cv2

            for outs in results:
                for i in range(len(outs['img_path'])):
                    lanes = outs['lanes'][i]
                    img_path = outs['img_path'][i]
                    img = cv2.imread(img_path)
                    out_file = os.path.join(output_dir, os.path.basename(img_path))
                    lanes = [
                        lane.to_array(
                            sample_y_range=[
                                self.cfg['sample_y']['start'],
                                self.cfg['sample_y']['end'],
                                self.cfg['sample_y']['step']
                            ],
                            img_w=self.cfg.ori_img_w,
                            img_h=self.cfg.ori_img_h
                        ) for lane in lanes
                    ]
                    imshow_lanes(img, lanes, out_file)
        return results

    def _flops(self, loader):
        if hasattr(self.model, 'aux_neck'):
            delattr(self.model, 'aux_neck')
        if hasattr(self.model, 'aux_head'):
            delattr(self.model, 'aux_head')

        self.model.eval()

        try:
            from thop import profile
        except ImportError:
            logger.warning(
                "Unable to calculate FLOPs, please install thop: `pip install thop`"
            )
            return

        input_data = None
        for data in loader:
            input_data = data
            break

        if input_data is None:
            logger.warning("Empty dataloader, skip FLOPs calculation.")
            return

        if self.cfg.use_gpu:
            image = input_data['image'][0].unsqueeze(0).cuda()
        else:
            image = input_data['image'][0].unsqueeze(0)

        if hasattr(self.model, 'forward') and 'image' in input_data:
            try:
                flops, params = profile(self.model, inputs=(image,), verbose=False)
            except Exception as e:
                dummy_input = {
                    'image': image,
                    'im_shape': input_data['im_shape'][0].unsqueeze(0) if not self.cfg.use_gpu else
                    input_data['im_shape'][0].unsqueeze(0).cuda(),
                    'scaler_factor': input_data['scaler_factor'][0].unsqueeze(0) if not self.cfg.use_gpu else
                    input_data['scaler_factor'][0].unsqueeze(0).cuda()
                }
                try:
                    flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
                except Exception as e:
                    logger.warning(f"Failed to compute FLOPs: {e}")
                    return
        else:
            logger.warning("Model input format not supported for FLOPs calculation.")
            return

        flops_g = flops / (1000 ** 3)
        logger.info("Model FLOPs: {:.6f}G. (image shape is {})".format(
            flops_g, tuple(image.shape)))

    def reset_norm_param_attr(self, layer, **kwargs):
        """
        递归遍历模型，将所有归一化层（BatchNorm/LayerNorm/GroupNorm）的参数属性（如 weight_attr、bias_attr）重置为新值。
        Args:
            layer (torch.nn.Module): 要处理的层
            **kwargs: 传递给新层构造函数的参数（如 `affine`, `track_running_stats` 等）

        Returns:
            torch.nn.Module: 处理后的层（可能已被替换）
        """

        if isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            # 保存原始状态字典（参数和 buffers）
            src_state_dict = layer.state_dict()

            # 根据类型重建归一化层（保留原始配置 + 应用新 kwargs）
            if isinstance(layer, nn.BatchNorm2d):
                new_layer = nn.BatchNorm2d(
                    num_features=layer.num_features,
                    eps=layer.eps,
                    momentum=layer.momentum,
                    affine=layer.affine,
                    track_running_stats=layer.track_running_stats,
                    **kwargs
                )
            elif isinstance(layer, nn.LayerNorm):
                new_layer = nn.LayerNorm(
                    normalized_shape=layer.normalized_shape,
                    eps=layer.eps,
                    elementwise_affine=layer.elementwise_affine,
                    **kwargs
                )
            else:  # GroupNorm
                new_layer = nn.GroupNorm(
                    num_groups=layer.num_groups,
                    num_channels=layer.num_channels,
                    eps=layer.eps,
                    affine=layer.affine,
                    **kwargs
                )

            # 加载原始参数和 buffers（确保数值不变）
            new_layer.load_state_dict(src_state_dict, strict=False)
            return new_layer

        else:
            # 递归处理子模块
            for name, sublayer in layer.named_children():
                new_sublayer = self.reset_norm_param_attr(sublayer, **kwargs)
                if new_sublayer is not sublayer:
                    setattr(layer, name, new_sublayer)
            return layer
