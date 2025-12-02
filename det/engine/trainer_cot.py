#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :trainer_cot.py
@Author :CodeCat
@Date   :2025/12/2 11:44
"""
from det.utils.logger import setup_logger
from det.engine.trainer import Trainer

logger = setup_logger('det.engine')


class TrainerCot(Trainer):
    """
        用于 Label Co-tuning 训练的 Trainer。
        该方法通过预训练模型学习基础类（base_classes）与新类（novel_classes）之间的关系，
        并基于此关系初始化一个辅助分类头（co-tuning head），用于少样本微调。
    """
    def __init__(self, cfg, mode='train'):
        super(TrainerCot, self).__init__(cfg, mode)
        self.cotuning_init()

    def cotuning_init(self):
        num_classes_novel = self.cfg['num_classes']

        self.load_weights(self.cfg.pretrain_weights)

        self.model.eval()
        relationship = self.model.relationship_learning(self.loader, num_classes_novel)

        self.model.init_cot_head(relationship)
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())