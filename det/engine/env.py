#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :env.py
@Author :CodeCat
@Date   :2025/12/2 11:55
"""
import os
import random
import numpy as np

import torch
import torch.distributed as dist

def init_parallel_env():
    """
    初始化分布式训练环境，并为每个 rank 设置独立的随机种子，
    以确保多卡训练时数据加载和初始化的可重现性。
    """
    env = os.environ

    torch_dist_env = 'RANK' in env or 'LOCAL_RANK' in env

    if torch_dist_env:
        if 'RANK' in env:
            trainer_id = int(env['RANK'])
        else:
            trainer_id = int(env.get('LOCAL_RANK', 0))

        # 为每个 rank 设置独立的随机种子
        local_seed = 99 + trainer_id
        random.seed(local_seed)
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed(local_seed)
        torch.cuda.manual_seed_all(local_seed)

        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
            dist.barrier()

    else:
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(99)

def init_distributed_mode():
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    RANK = int(os.getenv('RANK', -1))
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    if WORLD_SIZE > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=RANK)
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.barrier()
        if RANK == 0:
            logger.info("Initialized distributed training with {} processes".format(WORLD_SIZE))
    else:
        logger.info("Initialized single-GPU training")


def set_random_seed(seed):
    """
    设置所有随机数生成器的种子，以确保训练可复现。

    Args:
        seed (int): 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)