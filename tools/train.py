#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :train.py
@Author :CodeCat
@Date   :2025/11/17 10:07
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
from omegaconf import OmegaConf
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import warnings

warnings.filterwarnings('ignore')

import cv2

cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

import torch
from det.engine.trainer import Trainer
from det.engine.trainer_cot import TrainerCot
from det.engine.env import init_distributed_mode, set_random_seed
import det.utils.check as check
from det.utils.cli import ArgsParser, merge_args
from det.utils.logger import setup_logger

logger = setup_logger('train')


def parse_args():
    """
    解析命令行参数，用于配置训练/评估行为。

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = ArgsParser()

    # 是否在训练过程中进行验证
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation during training."
    )

    # 断点恢复路径
    parser.add_argument(
        "-r", "--resume",
        default=None,
        help="Path to weights for resume training."
    )

    # 模型压缩配置（如 QAT、蒸馏等）
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file for model slimming (e.g., QAT, distillation)."
    )

    # 内部 CI 使用的连续评估标志（可忽略）
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="Enable continuous evaluation (for internal CI only)."
    )

    # 混合精度训练
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable automatic mixed precision (AMP) training."
    )

    # 可视化日志（VisualDL）
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Record scalars to VisualDL."
    )
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='Logging directory for scalar.'
    )

    # W&B 支持
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Log metrics to Weights & Biases (W&B)."
    )

    # 仅保存预测结果（用于提交评测）
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Save evaluation results only (no metric computation).'
    )

    # 性能分析器选项
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help="Profiler options in format 'key1=value1;key2=value2'. "
             "Used by PyTorch Profiler (see docs for details)."
    )

    # SNIPER：保存提议框（用于半监督）
    parser.add_argument(
        '--save_proposals',
        action='store_true',
        default=False,
        help='Save training proposals (for SNIPER semi-supervised training).'
    )
    parser.add_argument(
        '--proposals_path',
        type=str,
        default="sniper/proposals.json",
        help='Path to save proposals JSON file.'
    )

    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    init_distributed_mode()

    if FLAGS.enable_ce:
        set_random_seed(0)

    if cfg.get('use_cot', False):
        trainer = TrainerCot(cfg, mode='train')
    else:
        trainer = Trainer(cfg, mode='train')

    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()
    cfg = OmegaConf.load(FLAGS.config)
    cfg = merge_args(cfg, FLAGS)

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
