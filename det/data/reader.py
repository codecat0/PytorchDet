#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :reader.py
@Author :CodeCat
@Date   :2025/11/5 16:43
"""
import copy
import os
import traceback
import six
import sys
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from . import transform
from .utils import default_collate_fn
from loguru import logger


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        """
        Args:
            transforms (list): 包含一系列变换操作的列表，每个变换操作是一个字典，键为操作名，值为操作参数。
            num_classes (int, optional): 类别数，默认为80。用于初始化一些需要类别数的变换操作。
        Returns:
            None
        """
        self.transforms = transforms
        self.transforms_cls = []
        for f in self.transforms:
            if hasattr(f, 'num_classes'):
                f.num_classes = num_classes

            self.transforms_cls.append(f)

    def _update_transforms_cls(self, data):
        if 'transform_schedulers' in data:
            def is_valid(op):
                op_name = op.__class__.__name__
                for t in data['transform_schedulers']:
                    for k, v in t.items():
                        if op_name == k:
                            # [start_epoch, stop_epoch)
                            start_epoch = v.get('start_epoch', 0)
                            if start_epoch > data['curr_epoch']:
                                return False
                            stop_epoch = v.get('stop_epoch', float('inf'))
                            if stop_epoch <= data['curr_epoch']:
                                return False
                return True

            return filter(is_valid, self.transforms_cls)
        else:
            return self.transforms_cls

    def __call__(self, data):
        transforms_cls = self._update_transforms_cls(data)
        for f in transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map sample transform [{}] "
                               "with error: {} and stack:\n{}".format(
                    f, e, str(stack_info)))
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        transforms_cls = self._update_transforms_cls(data[0])
        for f in transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                    f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped', 'transform_schedulers']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BaseDataLoader(object):
    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        self.dataset.parse_dataset()
        self.dataset.set_transform(self._sample_transforms)
        self.dataset.set_kwargs(**self.kwargs)
        if batch_sampler is None:
            if self._is_distributed():
                self._batch_sampler = DistributedSampler(
                    dataset,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last)
            else:
                if self.shuffle:
                    self._batch_sampler = RandomSampler(dataset)
                else:
                    self._batch_sampler = SequentialSampler(dataset)
        else:
            self._batch_sampler = batch_sampler

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            pin_memory=self.use_shared_memory,
            drop_last=self.drop_last,
            shuffle=False,
        )
        self.loader = iter(self.dataloader)

        return self

    def _is_distributed(self):
        """判断是否在分布式训练环境中"""
        if not torch.distributed.is_available():
            return False
        if not torch.distributed.is_initialized():
            return False
        return True

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())


class TrainReader(BaseDataLoader):
    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(TrainReader, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)


class EvalReader(BaseDataLoader):
    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(EvalReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


class TestReader(BaseDataLoader):
    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(TestReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)
        
        
if __name__ == "__main__":
    from det.data.source.coco import COCODataset
    from det.data.transform.operators import Decode, RandomFlip, RandomSelect,\
                            RandomShortSideResize, RandomSizeCrop,\
                            NormalizeImage, NormalizeBox, BboxXYXY2XYWH, Permute
    from det.data.transform.batch_operators import PadMaskBatch
    from det.data.reader import TrainReader
    
    dataset = COCODataset(
        dataset_dir='/data0/helizhi/works/PytorchDet/data',
        image_dir='train',
        anno_path='annotations/instance_train.json',
        data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd']
    )
    
    sample_transforms = [
            Decode(),
            RandomFlip(prob=0.5),
            RandomSelect(
                transforms1=[
                    RandomShortSideResize(
                        short_side_sizes=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                        max_size=1333,
                    )
                ],
                transforms2=[
                    RandomShortSideResize(short_side_sizes=[400, 500, 600]),
                    RandomSizeCrop(min_size=384, max_size=600),
                    RandomShortSideResize(
                        short_side_sizes=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                        max_size=1333,
                    )
                ],
            ),
            NormalizeImage(
                is_scale=True,
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            NormalizeBox(),
            BboxXYXY2XYWH(),
            Permute()
        ]
    
    batch_transforms = [
        PadMaskBatch(pad_to_stride=-1, return_pad_mask=True)
    ]
    
    train_loader = TrainReader(
        sample_transforms=sample_transforms,
        batch_transforms=batch_transforms,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        collate_batch=False
    )(dataset, worker_num=0)
    for data in train_loader:
        for key, value in data.items():
            print('{}: {}'.format(key, value))
        break

    # from omegaconf import OmegaConf
    # from hydra.utils import instantiate
    # cfg = OmegaConf.load('config/base/dataset/coco_dataloader.yaml')
    # dataset = instantiate(cfg.dataset)
    # dataloader = instantiate(cfg.dataloader, _convert_="all")(dataset, cfg.worker_num)
    # for data in dataloader:
    #     for key, value in data.items():
    #         print('{}: {}'.format(key, value))
    #     break