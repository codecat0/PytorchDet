#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :keypoint_metrics.py
@Author :CodeCat
@Date   :2025/11/18 14:49
"""
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from det.modeling.keypoint_utils import oks_nms, keypoint_pck_accuracy, keypoint_auc, keypoint_epe
from scipy.io import loadmat, savemat
from loguru import logger


class KeyPointTopDownCOCOEval(object):
    """
    用于自顶向下（Top-Down）关键点检测模型的 COCO 风格评估类。

    该类负责收集模型的预测结果，将其格式化为 COCO 格式的 JSON 文件，
    并使用 pycocotools 库进行评估，最终计算 AP (Average Precision) 和 AR (Average Recall) 等指标。

    该实现参考了  https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    """

    def __init__(self,
                 anno_file,              # 真实标注文件的路径 (例如: person_keypoints_val2017.json)
                 num_samples,            # 数据集中的样本总数
                 num_joints,             # 关键点的数量 (例如 COCO 是 17 个)
                 output_eval,            # 评估结果（JSON 文件）的输出目录
                 iou_type='keypoints',   # COCOeval 使用的类型，这里固定为 'keypoints'
                 in_vis_thre=0.2,        # 用于计算 OKS 和 PCK 等指标的可见性阈值
                 oks_thre=0.9,           # 用于 OKS NMS 的阈值
                 save_prediction_only=False): # 如果为 True，则只保存预测结果，不进行 COCOeval 评估
        super(KeyPointTopDownCOCOEval, self).__init__()
        # 初始化 COCO API 的真实标注对象
        self.coco = COCO(anno_file)
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.iou_type = iou_type
        self.in_vis_thre = in_vis_thre
        self.oks_thre = oks_thre
        self.output_eval = output_eval
        # 定义保存结果的 JSON 文件路径
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        self.save_prediction_only = save_prediction_only
        # 重置内部存储结果的结构
        self.reset()

    def reset(self):
        """
        重置评估器的内部状态，清空之前收集的所有结果。
        """
        # 初始化用于存储所有预测结果、包围框信息和图像路径的结构
        self.results = {
            # 存储所有样本的所有关键点预测 [N, K, 3] (x, y, score)
            'all_preds': np.zeros(
                (self.num_samples, self.num_joints, 3), dtype=np.float32),
            # 存储与预测相关的信息 [N, 6] (center_x, center_y, scale_x, scale_y, area, score)
            'all_boxes': np.zeros((self.num_samples, 6)),
            # 存储图像 ID 或路径列表
            'image_path': []
        }
        # 存储最终的评估统计结果
        self.eval_results = {}
        # 当前已处理的样本索引计数器
        self.idx = 0

    def update(self, inputs, outputs):
        """
        更新评估器，将一批新的预测结果添加到内部存储结构中。

        Args:
            inputs (dict): 包含输入信息的字典，如 'center', 'scale', 'score', 'im_id'。
            outputs (dict): 包含模型输出的字典，其中 'keypoint' 包含预测的关键点。
        """
        # 从模型输出中提取关键点 (kpts) 和其他可能的输出 (这里忽略)
        # 假设 outputs['keypoint'][0] 是 (kpts, other_info)
        kpts, _ = outputs['keypoint'][0]

        # 获取当前批次的图像数量
        num_images = inputs['image'].shape[0]

        # 将当前批次的预测关键点存储到 results['all_preds'] 中
        # 只取前 3 个维度 (x, y, score)
        self.results['all_preds'][self.idx:self.idx + num_images, :, 0:3] = kpts[:, :, 0:3]

        # 将当前批次的中心点信息存储到 results['all_boxes'] 的前两列
        # 检查 inputs['center'] 是否为 PaddlePaddle 张量，如果是则转换为 numpy
        self.results['all_boxes'][self.idx:self.idx + num_images, 0:2] = inputs[
            'center'].numpy()[:, 0:2] if isinstance(
                inputs['center'], paddle.Tensor) else inputs['center'][:, 0:2]

        # 将当前批次的尺度信息存储到 results['all_boxes'] 的 3,4 列
        self.results['all_boxes'][self.idx:self.idx + num_images, 2:4] = inputs[
            'scale'].numpy()[:, 0:2] if isinstance(
                inputs['scale'], paddle.Tensor) else inputs['scale'][:, 0:2]

        # 计算并存储包围框面积 (scale * 200 的乘积，这是一个常见的转换方式)
        # 存储到 results['all_boxes'] 的第 5 列
        self.results['all_boxes'][self.idx:self.idx + num_images, 4] = np.prod(
            inputs['scale'].numpy() * 200, 1) if isinstance(inputs['scale'], paddle.Tensor) else np.prod(
                inputs['scale'] * 200, 1)

        # 将当前批次的置信度分数存储到 results['all_boxes'] 的第 6 列
        self.results['all_boxes'][self.idx:self.idx + num_images, 5] = np.squeeze(inputs['score'].numpy()) if isinstance(
            inputs['score'], paddle.Tensor) else np.squeeze(inputs['score'])

        # 将当前批次的图像 ID 添加到 results['image_path'] 列表中
        if isinstance(inputs['im_id'], paddle.Tensor):
            self.results['image_path'].extend(inputs['im_id'].numpy())
        else:
            self.results['image_path'].extend(inputs['im_id'])

        # 更新索引计数器
        self.idx += num_images

    def _write_coco_keypoint_results(self, keypoints):
        """
        将格式化后的关键点结果写入 JSON 文件，以便 COCO API 读取。

        Args:
            keypoints (list): 格式化后的关键点列表，每个元素对应一张图像的所有实例。
        """
        # 创建一个数据包，包含类别 ID、类别名称、注释类型和关键点数据
        data_pack = [{
            'cat_id': 1,  # COCO 中 person 类别的 ID 是 1
            'cls': 'person',
            'ann_type': 'keypoints',
            'keypoints': keypoints
        }]

        # 调用内核函数将数据包转换为 COCO 格式的最终结果列表
        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        # 确保输出目录存在
        if not os.path.exists(self.output_eval):
            os.makedirs(self.output_eval)

        # 将结果写入 JSON 文件
        with open(self.res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')

        # 尝试加载 JSON 文件以验证其有效性
        try:
            json.load(open(self.res_file))
        except Exception:
            # 如果加载失败，可能是因为文件格式问题（例如，手动添加了逗号）
            # 这段代码尝试修复文件末尾的 ']'
            content = []
            with open(self.res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'  # 修正最后一行
            with open(self.res_file, 'w') as f:
                for c in content:
                    f.write(c)

    @staticmethod
    def _coco_keypoint_results_one_category_kernel(data_pack):
        """
        将数据包转换为 COCO 格式的最终结果列表的核心函数。

        Args:
            data_pack (dict): 包含类别 ID、关键点等信息的数据包。

        Returns:
            list: COCO 格式的字典列表，每个字典代表一个检测实例。
        """
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        # 遍历每张图像的关键点预测
        for img_kpts in keypoints:
            if len(img_kpts) == 0: # 如果该图像没有预测结果，则跳过
                continue

            # 提取所有实例的关键点坐标，并重塑为 [N, K*3] 的形状
            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            _key_points = _key_points.reshape(_key_points.shape[0], -1)

            # 为该图像的每个预测实例创建 COCO 格式的字典
            result = [{
                'image_id': img_kpts[k]['image'],        # 图像 ID
                'category_id': cat_id,                   # 类别 ID
                'keypoints': _key_points[k].tolist(),    # 关键点坐标列表 [x1,y1,v1,x2,y2,v2,...]
                'score': img_kpts[k]['score'],           # 实例置信度
                'center': list(img_kpts[k]['center']),   # 中心点 (可能用于评估)
                'scale': list(img_kpts[k]['scale'])      # 尺度 (可能用于评估)
            } for k in range(len(img_kpts))]
            cat_results.extend(result) # 将该图像的所有实例结果添加到总列表

        return cat_results

    def get_final_results(self, preds, all_boxes, img_path):
        """
        对收集到的预测结果进行最终处理，包括重新评分和 OKS NMS。

        Args:
            preds (np.ndarray): 所有样本的预测关键点 [N, K, 3]。
            all_boxes (np.ndarray): 所有样本的包围框信息 [N, 6]。
            img_path (list): 所有样本的图像 ID 列表。
        """
        _kpts = []
        # 将预测、包围框、图像路径等信息打包成一个列表，每个元素对应一个样本
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,          # [K, 3] (x, y, score)
                'center': all_boxes[idx][0:2], # [2] (cx, cy)
                'scale': all_boxes[idx][2:4],  # [2] (sx, sy)
                'area': all_boxes[idx][4],     # float
                'score': all_boxes[idx][5],    # float
                'image': int(img_path[idx])    # int
            })

        # 按图像 ID 分组，将同一图像的所有预测实例放在一起
        # kpts 是一个字典，键是图像 ID，值是该图像的所有预测实例列表
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # 对每张图像进行重新评分和 OKS NMS
        num_joints = preds.shape[1]
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img] # 获取当前图像的所有预测实例

            # 1. 重新评分：结合关键点置信度和包围框置信度
            for n_p in img_kpts:
                box_score = n_p['score'] # 原始包围框置信度
                kpt_score = 0
                valid_num = 0
                # 计算该实例所有可见关键点的平均置信度
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2] # 第 n_jt 个关键点的置信度
                    if t_s > in_vis_thre: # 如果关键点可见
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # 新的实例分数 = 关键点平均置信度 * 包围框置信度
                n_p['score'] = kpt_score * box_score

            # 2. OKS NMS：根据 OKS (Object Keypoint Similarity) 去除重复检测
            # 提取当前图像的所有预测实例用于 NMS
            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)

            # 根据 NMS 结果保留实例
            if len(keep) == 0:
                # 如果 NMS 后没有保留任何实例，则保留所有原始实例（这可能是个 bug，通常应为 []）
                oks_nmsed_kpts.append(img_kpts)
            else:
                # 保留 NMS 后的实例
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        # 将经过处理的最终结果写入 JSON 文件
        self._write_coco_keypoint_results(oks_nmsed_kpts)

    def accumulate(self):
        """
        执行最终的评估流程：处理结果、保存 JSON、调用 COCO API 评估。
        """
        # 处理收集到的所有预测结果
        self.get_final_results(self.results['all_preds'],
                               self.results['all_boxes'],
                               self.results['image_path'])

        # 如果只保存预测结果而不评估
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return

        # 使用 COCO API 加载预测结果
        coco_dt = self.coco.loadRes(self.res_file)
        # 创建 COCOeval 对象进行评估
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None # 确保只评估关键点，不评估分割
        # 执行评估
        coco_eval.evaluate()
        # 汇总评估结果
        coco_eval.accumulate()
        # 打印评估摘要 (AP, AR 等)
        coco_eval.summarize()

        # 提取评估统计结果
        keypoint_stats = []
        for ind in range(len(coco_eval.stats)):
            keypoint_stats.append((coco_eval.stats[ind]))
        self.eval_results['keypoint'] = keypoint_stats

    def log(self):
        """
        打印评估结果到控制台。
        """
        if self.save_prediction_only:
            return # 如果只保存预测，不打印评估结果
        # 定义 COCO 评估指标的名称
        stats_names = [
            'AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        num_values = len(stats_names)
        # 打印表头
        print(' '.join(['| {}'.format(name) for name in stats_names]) + ' |')
        print('|---' * (num_values + 1) + '|')
        # 打印评估值
        print(' '.join([
            '| {:.3f}'.format(value) for value in self.eval_results['keypoint']
        ]) + ' |')

    def get_results(self):
        """
        获取最终的评估结果。

        Returns:
            dict: 包含评估统计信息的字典。
        """
        return self.eval_results


class KeyPointTopDownCOCOWholeBadyHandEval(object):
    """
    用于自顶向下（Top-Down）手部关键点检测模型的 COCO 风格评估类。

    该类专门针对 COCO 数据集中的左手和右手关键点进行评估。
    它会解析 COCO 格式的标注文件，提取手部相关的标注，
    收集模型预测结果，并使用 PCK, AUC, EPE 等指标进行评估。
    """

    def __init__(self,
                 anno_file,              # 真实标注文件的路径 (例如: person_keypoints_val2017.json)
                 num_samples,            # 数据集中的样本总数
                 num_joints,             # 关键点的数量 (例如，对于手部，通常是 21 个)
                 output_eval,            # 评估结果（JSON 文件）的输出目录
                 save_prediction_only=False): # 如果为 True，则只保存预测结果，不进行评估
        super(KeyPointTopDownCOCOWholeBadyHandEval, self).__init__()
        # 初始化 COCO API 的真实标注对象
        self.coco = COCO(anno_file)
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.output_eval = output_eval
        # 定义保存结果的 JSON 文件路径
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        self.save_prediction_only = save_prediction_only
        # 解析数据集，构建包含手部标注的数据库
        self.parse_dataset()
        # 重置内部存储结果的结构
        self.reset()

    def parse_dataset(self):
        """
        解析 COCO 格式的标注文件，提取有效的左手和右手标注，
        并构建一个包含边界框、真实关键点和可见性掩码的数据库 (self.db)。
        """
        gt_db = [] # 初始化数据库列表
        num_joints = self.num_joints
        coco = self.coco
        # 获取所有图像的 ID
        img_ids = coco.getImgIds()

        # 遍历每张图像
        for img_id in img_ids:
            # 获取该图像的所有标注 ID (排除 crowd 标注)
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            # 加载这些标注
            objs = coco.loadAnns(ann_ids)

            # 遍历每个标注对象
            for obj in objs:
                # 遍历左手和右手 ('left', 'right')
                for type in ['left', 'right']:
                    # 检查该手是否有效 (valid) 且关键点坐标存在 (max > 0)
                    # 例如 obj['lefthand_valid'] 为 True 且 obj['lefthand_kpts'] 中的最大值 > 0
                    if (obj[f'{type}hand_valid'] and
                            max(obj[f'{type}hand_kpts']) > 0):

                        # 初始化存储该手部真实关键点的数组 [K, 3] (x, y, v)
                        joints = np.zeros((num_joints, 3), dtype=np.float32)
                        # 初始化存储该手部关键点可见性的数组 [K, 3] (x_vis, y_vis, v_vis)
                        joints_vis = np.zeros((num_joints, 3), dtype=np.float32)

                        # 提取关键点坐标并重塑为 [K, 3]
                        keypoints = np.array(obj[f'{type}hand_kpts'])
                        keypoints = keypoints.reshape(-1, 3)
                        # 存储 x, y 坐标
                        joints[:, :2] = keypoints[:, :2]
                        # 存储可见性标志 (将置信度 v 转换为 0 或 1)
                        joints_vis[:, :2] = np.minimum(1, keypoints[:, 2:3])

                        # 将该手部的标注信息添加到数据库中
                        gt_db.append({
                            'bbox': obj[f'{type}hand_box'], # 手部边界框 [x, y, w, h]
                            'gt_joints': joints,             # 真实关键点坐标 [K, 3]
                            'joints_vis': joints_vis,        # 关键点可见性 [K, 3]
                        })
        # 将构建好的数据库赋值给实例变量
        self.db = gt_db

    def reset(self):
        """
        重置评估器的内部状态，清空之前收集的所有预测结果。
        """
        # 初始化用于存储所有预测结果的结构
        self.results = {
            # 存储所有样本的所有关键点预测 [N, K, 3] (x, y, score)
            'preds': np.zeros(
                (self.num_samples, self.num_joints, 3), dtype=np.float32),
        }
        # 存储最终的评估统计结果
        self.eval_results = {}
        # 当前已处理的样本索引计数器
        self.idx = 0

    def update(self, inputs, outputs):
        """
        更新评估器，将一批新的预测结果添加到内部存储结构中。

        Args:
            inputs (dict): 包含输入信息的字典（此评估器中未使用）。
            outputs (dict): 包含模型输出的字典，其中 'keypoint' 包含预测的关键点。
        """
        # 从模型输出中提取关键点 (kpts) 和其他可能的输出 (这里忽略)
        # 假设 outputs['keypoint'][0] 是 (kpts, other_info)
        kpts, _ = outputs['keypoint'][0]

        # 获取当前批次的图像数量
        num_images = inputs['image'].shape[0]

        # 将当前批次的预测关键点存储到 results['preds'] 中
        # 只取前 3 个维度 (x, y, score)
        self.results['preds'][self.idx:self.idx + num_images, :, 0:3] = kpts[:, :, 0:3]

        # 更新索引计数器
        self.idx += num_images

    def accumulate(self):
        """
        执行最终的评估流程：处理预测结果、保存 JSON、调用评估函数。
        """
        # 处理收集到的所有预测结果
        self.get_final_results(self.results['preds'])

        # 如果只保存预测结果而不评估
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return

        # 调用 evaluate 函数进行评估，计算 PCK, AUC, EPE 指标
        self.eval_results = self.evaluate(self.res_file, ('PCK', 'AUC', 'EPE'))

    def get_final_results(self, preds):
        """
        将预测的关键点格式化为 COCO 风格的 JSON 结构并保存。

        Args:
            preds (np.ndarray): 所有样本的预测关键点 [N, K, 3]。
        """
        kpts = []
        # 遍历每个样本的预测关键点
        for idx, kpt in enumerate(preds):
            # 将关键点坐标转换为列表格式并添加到 kpts 列表
            kpts.append({'keypoints': kpt.tolist()})

        # 调用函数将结果写入 JSON 文件
        self._write_keypoint_results(kpts)

    def _write_keypoint_results(self, keypoints):
        """
        将格式化后的关键点结果写入 JSON 文件。

        Args:
            keypoints (list): 格式化后的关键点列表。
        """
        # 确保输出目录存在
        if not os.path.exists(self.output_eval):
            os.makedirs(self.output_eval)

        # 将结果写入 JSON 文件
        with open(self.res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')

        # 尝试加载 JSON 文件以验证其有效性（与 KeyPointTopDownCOCOEval 类似）
        try:
            json.load(open(self.res_file))
        except Exception:
            # 如果加载失败，尝试修复文件末尾的 ']'
            content = []
            with open(self.res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(self.res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def log(self):
        """
        打印评估结果到控制台。
        """
        if self.save_prediction_only:
            return # 如果只保存预测，不打印评估结果
        # 遍历并打印每个评估指标及其值
        for item, value in self.eval_results.items():
            print("{} : {}".format(item, value))

    def get_results(self):
        """
        获取最终的评估结果。

        Returns:
            dict: 包含评估统计信息的字典。
        """
        return self.eval_results

    def evaluate(self, res_file, metrics, pck_thr=0.2, auc_nor=30):
        """
        执行关键点评估。

        Args:
            res_file (str): 存储预测结果的 Json 文件路径。
            metrics (str | list[str]): 要执行的评估指标。
                                    选项: 'PCK', 'AUC', 'EPE'。
            pck_thr (float, optional): PCK 阈值，默认为 0.2。
            auc_nor (float, optional): AUC 归一化因子（像素），默认为 30。

        Returns:
            OrderedDict: 一个有序字典，包含评估结果，例如 {'PCK': value, 'AUC': value, 'EPE': value}。
        """
        # 初始化存储评估结果的列表
        info_str = []

        # 从 JSON 文件加载预测结果
        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        # 确保预测结果的数量与数据库中的真实标注数量一致
        assert len(preds) == len(self.db)

        # 初始化用于评估的列表
        outputs = []  # 存储预测的关键点
        gts = []      # 存储真实的（Groundtruth）关键点
        masks = []    # 存储关键点可见性掩码
        threshold_bbox = [] # 存储用于 PCK 计算的归一化因子（边界框尺寸）

        # 遍历预测结果和数据库中的真实标注
        for pred, item in zip(preds, self.db):
            # 提取预测关键点的 x, y 坐标 (去掉置信度 v)
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            # 提取真实关键点的 x, y 坐标 (去掉可见性 v)
            gts.append(np.array(item['gt_joints'])[:, :-1])
            # 提取关键点可见性掩码 (检查 x 的可见性是否为 True)
            masks.append((np.array(item['joints_vis'])[:, 0]) > 0)

            # 如果需要计算 PCK 指标
            if 'PCK' in metrics:
                # 获取手部边界框 [x, y, w, h]
                bbox = np.array(item['bbox'])
                # 使用边界框的宽度和高度中的最大值作为归一化因子
                bbox_thr = np.max(bbox[2:])
                # 将归一化因子扩展为 [max_size, max_size] 的形式
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

        # 将列表转换为 numpy 数组，便于批量计算
        outputs = np.array(outputs) # 预测关键点 [N, K, 2]
        gts = np.array(gts)         # 真实关键点 [N, K, 2]
        masks = np.array(masks)     # 可见性掩码 [N, K]
        threshold_bbox = np.array(threshold_bbox) # 归一化因子 [N, 2]

        # 根据指定的指标进行计算
        if 'PCK' in metrics:
            # 计算 PCK 准确率
            # keypoint_pck_accuracy 返回 (acc_per_kpt, avg_acc, valid_count)
            # 我们只关心平均准确率 avg_acc
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            # 将 PCK 结果添加到 info_str
            info_str.append(('PCK', pck))

        if 'AUC' in metrics:
            # 计算 AUC
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks, auc_nor)))

        if 'EPE' in metrics:
            # 计算 EPE
            info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        # 将结果列表转换为有序字典
        name_value = OrderedDict(info_str)

        # 返回评估结果字典
        return name_value


# 导入必要的库 (通常在文件开头)
import os
import json
import numpy as np
from collections import OrderedDict
import logging
# 假设这些库是可用的
from scipy.io import loadmat, savemat

logger = logging.getLogger(__name__)

class KeyPointTopDownMPIIEval(object):
    """
    用于自顶向下（Top-Down）关键点检测模型的 MPII 数据集风格评估类。

    该类负责收集模型的预测结果，将其格式化并保存，然后使用 PCKh (Percentage of Correct Keypoints w.r.t. Head)
    指标对 MPII 数据集进行评估。PCKh 是一种常用的 MPII 评估指标，它使用头部尺寸作为归一化因子来衡量关键点预测的准确性。
    """

    def __init__(self,
                 anno_file,              # 真实标注文件的路径 (例如: mpii_annotations.json 或 .mat 文件路径信息)
                 num_samples,            # 数据集中的样本总数
                 num_joints,             # 关键点的数量 (MPII 是 16 个)
                 output_eval,            # 评估结果（JSON 文件）的输出目录
                 oks_thre=0.9,           # OKS 阈值
                 save_prediction_only=False): # 如果为 True，则只保存预测结果，不进行评估
        super(KeyPointTopDownMPIIEval, self).__init__()
        # 保存标注文件路径（用于加载 MPII 的真实值 .mat 文件）
        self.ann_file = anno_file
        # 定义保存预测结果的 JSON 文件路径
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        # 保存是否只保存预测的标志
        self.save_prediction_only = save_prediction_only
        # 重置内部存储结果的结构
        self.reset()

    def reset(self):
        """
        重置评估器的内部状态，清空之前收集的所有结果。
        """
        # 初始化用于存储所有预测结果的列表
        # 与 COCO 评估器不同，这里直接存储每个批次的结果字典
        self.results = []
        # 存储最终的评估统计结果
        self.eval_results = {}
        # 当前已处理的样本索引计数器 (此评估器中未严格按索引累加)
        self.idx = 0

    def update(self, inputs, outputs):
        """
        更新评估器，将一批新的预测结果添加到内部存储结构中。

        Args:
            inputs (dict): 包含输入信息的字典，如 'center', 'scale', 'score', 'image_file'。
            outputs (dict): 包含模型输出的字典，其中 'keypoint' 包含预测的关键点。
        """
        # 从模型输出中提取关键点 (kpts) 和其他可能的输出 (这里忽略)
        kpts, _ = outputs['keypoint'][0]

        # 获取当前批次的图像数量
        num_images = inputs['image'].shape[0]

        # 创建一个字典来存储当前批次的所有信息
        results = {}
        # 存储预测的关键点坐标 [N, K, 3] (x, y, score)
        results['preds'] = kpts[:, :, 0:3]
        # 初始化包围框信息数组 [N, 6] (center_x, center_y, scale_x, scale_y, area, score)
        results['boxes'] = np.zeros((num_images, 6))
        # 存储中心点信息
        results['boxes'][:, 0:2] = inputs['center'].numpy()[:, 0:2]
        # 存储尺度信息
        results['boxes'][:, 2:4] = inputs['scale'].numpy()[:, 0:2]
        # 计算并存储包围框面积 (scale * 200 的乘积)
        results['boxes'][:, 4] = np.prod(inputs['scale'].numpy() * 200, 1)
        # 存储置信度分数
        results['boxes'][:, 5] = np.squeeze(inputs['score'].numpy())
        # 存储图像文件路径
        results['image_path'] = inputs['image_file']

        # 将当前批次的结果字典添加到总列表中
        self.results.append(results)

    def accumulate(self):
        """
        执行最终的评估流程：保存预测结果、调用评估函数。
        """
        # 将收集到的所有预测结果保存到 JSON 文件
        self._mpii_keypoint_results_save()

        # 如果只保存预测结果而不评估
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return

        # 调用 evaluate 函数进行评估，计算 PCKh 指标
        self.eval_results = self.evaluate(self.results)

    def _mpii_keypoint_results_save(self):
        """
        将格式化后的预测结果写入 JSON 文件。
        """
        results = []
        # 遍历 self.results 列表中的每个批次结果字典
        for res in self.results:
            if len(res) == 0: # 如果该批次结果为空，则跳过
                continue
            # 为该批次的每个样本创建一个结果字典
            # res['preds'], res['boxes'], res['image_path'] 都是 N 个元素的数组
            result = [{
                'preds': res['preds'][k].tolist(),      # [K, 3] 预测坐标转列表
                'boxes': res['boxes'][k].tolist(),      # [6] 包围框信息转列表
                'image_path': res['image_path'][k],     # 字符串 图像路径
            } for k in range(len(res['preds']))] # 遍历该批次的 N 个样本
            results.extend(result) # 将该批次的所有样本结果添加到总列表

        # 将结果写入 JSON 文件
        with open(self.res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')

    def log(self):
        """
        打印评估结果到控制台。
        """
        if self.save_prediction_only:
            return # 如果只保存预测，不打印评估结果
        # 遍历并打印每个评估指标及其值
        for item, value in self.eval_results.items():
            print("{} : {}".format(item, value))

    def get_results(self):
        """
        获取最终的评估结果。

        Returns:
            dict: 包含评估统计信息的字典。
        """
        return self.eval_results

    def evaluate(self, outputs, savepath=None):
        """
        评估 MPII 数据集的 PCKh 指标。
        参考了 https://github.com/leoxiaobin/deep-high-resolution-net.pytorch评估代码。

        Args:
            outputs(list(dict)): 包含预测结果的列表。
                每个字典包含:
                * 'preds' (np.ndarray[N,K,3]): 前两个维度是坐标，第三个维度是分数。
                * 'boxes' (np.ndarray[N,6]): [center[0], center[1], scale[0], scale[1], area, score]
                * 'image_path' (list): 图像路径列表
            savepath (str, optional): 如果提供，将预测坐标保存为 .mat 文件。

        Returns:
            dict: 每个关节的 PCKh 指标。
        """
        # 将预测结果从列表格式转换为更适合评估的格式
        kpts = []
        for output in outputs: # 遍历每个批次的结果
            preds = output['preds'] # [N, K, 3]
            batch_size = preds.shape[0]
            for i in range(batch_size): # 遍历该批次的每个样本
                kpts.append({'keypoints': preds[i]}) # 将每个样本的 [K, 3] 关键点添加到列表

        # 将所有样本的关键点堆叠成一个 [Total_N, K, 3] 的数组
        preds = np.stack([kpt['keypoints'] for kpt in kpts])

        # 将 0-based 索引转换为 1-based 索引，并提取 x, y 坐标
        # MPII 数据集的坐标通常基于 1-based 索引
        preds = preds[..., :2] + 1.0

        # 如果提供了 savepath，则将预测坐标保存为 .mat 文件
        if savepath is not None:
            pred_file = os.path.join(savepath, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        # 定义用于 PCKh 计算的参数
        SC_BIAS = 0.6 # 缩放偏置，用于调整头部尺寸
        threshold = 0.5 # 默认 PCKh 阈值

        # 加载 MPII 的真实值 .mat 文件
        # 通常在标注文件同目录下有一个 mpii_gt_val.mat 文件
        gt_file = os.path.join(
            os.path.dirname(self.ann_file), 'mpii_gt_val.mat')
        gt_dict = loadmat(gt_file)
        # 从真实值文件中提取必要的数据
        dataset_joints = gt_dict['dataset_joints'] # 关节名称列表
        jnt_missing = gt_dict['jnt_missing']       # 标记哪些关节缺失 (1=缺失, 0=存在)
        pos_gt_src = gt_dict['pos_gt_src']         # 真实关键点坐标 [K, 2, N]
        headboxes_src = gt_dict['headboxes_src']   # 头部边界框 [2, 2, N] (左上角和右下角)

        # 转换预测坐标格式，使其与真实值格式匹配 [K, 2, Total_N]
        pos_pred_src = np.transpose(preds, [1, 2, 0])

        # 根据关节名称找到对应的索引
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]
        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        # 创建可见性掩码 (1=可见, 0=不可见或缺失)
        jnt_visible = 1 - jnt_missing

        # 计算预测坐标与真实坐标之间的误差 [K, Total_N]
        uv_error = pos_pred_src - pos_gt_src
        # 计算欧几里得距离误差 [K, Total_N]
        uv_err = np.linalg.norm(uv_error, axis=1)

        # 计算头部尺寸 [2, N] -> [N]
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        # 应用缩放偏置
        headsizes *= SC_BIAS
        # 创建归一化因子数组 [Total_N, K] (通过广播)
        scale = headsizes * np.ones((len(uv_err[0]), len(uv_err)), dtype=np.float32).T # 调整形状以匹配 uv_err

        # 计算归一化后的误差 [K, Total_N]
        scaled_uv_err = uv_err / scale
        # 应用可见性掩码 [K, Total_N]
        scaled_uv_err = scaled_uv_err * jnt_visible
        # 计算每个样本可见的关节数量 [Total_N]
        jnt_count = np.sum(jnt_visible, axis=1)
        # 计算低于阈值的误差数量 [K, Total_N]
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        # 计算 PCKh 百分比 [K]
        PCKh = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # 计算不同阈值下的 PCKh 曲线
        # rng 从 0 到 0.5，步长 0.01，共 51 个阈值
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        # pckAll 存储每个阈值下每个关节的 PCK [51, K]
        pckAll = np.zeros((len(rng), len(PCKh)), dtype=np.float32)

        for r, threshold in enumerate(rng):
            # 对每个阈值计算 PCK
            less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
            pckAll[r, :] = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # 使用 MaskedArray 处理不需要评估的关节 (MPII 通常不评估左右耳)
        # 这里假设索引 6 和 7 是左右耳 (根据 MPII 数据集定义，通常是 8 和 9，但代码写 6:8)
        # 根据标准 MPII 评估，通常忽略 'leye' (索引 14) 和 'reye' (索引 15)
        # 但此代码标记的是 6:8，可能对应特定的关节点定义或是一个旧的/特定的约定
        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True # 掩盖索引 6 和 7 的 PCKh 值

        # 同样处理关节数量，用于加权平均
        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        # 计算每个关节在总可见关节中的比例 [K]，用于加权平均
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        # 组织最终的评估结果
        # 包括单个关节的 PCKh、部分关节的平均 PCKh、总体加权平均 PCKh 和 PCKh@0.1
        name_value = [  # 使用 OrderedDict 存储结果
            ('Head', PCKh[head]), # 头部 PCKh
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])), # 肩膀平均 PCKh
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),   # 肘部平均 PCKh
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),   # 腕部平均 PCKh
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),     # 髋部平均 PCKh
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),    # 膝盖平均 PCKh
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),   # 脚踝平均 PCKh
            ('PCKh', np.sum(PCKh * jnt_ratio)),           # 总体加权平均 PCKh
            ('PCKh@0.1', np.sum(pckAll[11, :] * jnt_ratio)) # 阈值为 0.1 时的加权 PCKh (rng[11] = 0.1)
        ]
        name_value = OrderedDict(name_value)

        # 返回评估结果字典
        return name_value

    @staticmethod
    def _sort_and_unique_bboxes(kpts, key='bbox_id'):
        """
        对关键点结果进行排序并移除重复项。
        这个函数在当前类中定义但未被调用，可能是从其他评估器类复制过来的。

        Args:
            kpts (list): 关键点结果列表。
            key (str): 用于排序和去重的键名。

        Returns:
            list: 排序并去重后的关键点结果列表。
        """
        # 根据指定的键对列表进行排序
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        # 从后往前遍历，移除相邻的重复项
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts