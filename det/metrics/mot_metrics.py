#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :mot_metrics.py
@Author :CodeCat
@Date   :2025/11/21 11:05
"""
import os
import copy
import sys
import math
from collections import defaultdict
import numpy as np
from loguru import logger

from det.modeling.bbox_utils import bbox_iou_np_expand
from det.metrics.map_utils import ap_per_class
from det.metrics.metrics import Metric
from det.metrics.munkres import Munkres

try:
    import motmetrics as mm

    mm.lap.default_solver = 'lap'
except:
    print(
        'Warning: Unable to use MOT metric, please install motmetrics, for example: `pip install motmetrics`, see https://github.com/longcw/py-motmetrics'
    )
    pass

__all__ = ['MOTEvaluator', 'MOTMetric', 'JDEDetMetric', 'KITTIMOTMetric']


def read_mot_results(filename, is_gt=False, is_ignore=False):
    """
    读取MOT（Multiple Object Tracking）结果文件

    Args:
        filename (str): MOT结果文件路径
        is_gt (bool): 是否为真值文件，默认为False
        is_ignore (bool): 是否为忽略区域文件，默认为False

    Returns:
        dict: 包含结果的字典，格式为 {frame_id: [(tlwh, target_id, score), ...]}
              其中tlwh为边界框坐标(x, y, w, h)，target_id为目标ID，score为置信度
    """
    # 定义有效的标签（通常在MOT16/17数据集中使用1表示有效目标）
    valid_label = [1]
    # 定义忽略标签（在MOT挑战数据集中，这些标签表示需要忽略的目标）
    ignore_labels = [2, 7, 8, 12]

    if is_gt:
        print(
            "In MOT16/17 dataset the valid_label of ground truth is '{}', "
            "in other dataset it should be '0' for single classs MOT.".format(
                valid_label[0]))

    # 初始化结果字典
    results_dict = dict()

    # 检查文件是否存在
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                # 按逗号分割每一行
                linelist = line.split(',')

                # 如果分割后的元素少于7个，则跳过此行
                if len(linelist) < 7:
                    continue

                # 获取帧ID
                fid = int(linelist[0])

                # 如果帧ID小于1，则跳过此行
                if fid < 1:
                    continue

                # 为当前帧ID设置默认空列表
                results_dict.setdefault(fid, list())

                if is_gt:
                    # 处理真值文件
                    label = int(float(linelist[7]))  # 获取标签
                    mark = int(float(linelist[6]))  # 获取标记
                    # 如果标记为0或标签不在有效标签中，则跳过
                    if mark == 0 or label not in valid_label:
                        continue
                    score = 1  # 地面真值的置信度设为1
                elif is_ignore:
                    # 处理忽略区域文件
                    if 'MOT16-' in filename or 'MOT17-' in filename or 'MOT15-' in filename or 'MOT20-' in filename:
                        label = int(float(linelist[7]))  # 获取标签
                        vis_ratio = float(linelist[8])  # 获取可见比例
                        # 如果标签不在忽略标签中且可见比例大于等于0，则跳过
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1  # 忽略区域的置信度设为1
                else:
                    # 处理检测结果文件，获取置信度
                    score = float(linelist[6])

                # 获取边界框坐标（top-left x, top-left y, width, height）
                tlwh = tuple(map(float, linelist[2:6]))
                # 获取目标ID
                target_id = int(linelist[1])

                # 将结果添加到对应帧ID的列表中
                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


"""
MOT dataset label list, see in https://motchallenge.net
labels={'ped', ...			    % 1
        'person_on_vhcl', ...	% 2
        'car', ...				% 3
        'bicycle', ...			% 4
        'mbike', ...			% 5
        'non_mot_vhcl', ...		% 6
        'static_person', ...	% 7
        'distractor', ...		% 8
        'occluder', ...			% 9
        'occluder_on_grnd', ...	% 10
        'occluder_full', ...	% 11
        'reflection', ...		% 12
        'crowd' ...			    % 13
};
"""


def unzip_objs(objs):
    """
    解压对象列表，将其分解为边界框、ID和分数

    Args:
        objs: 对象列表，每个元素为 (tlwh, target_id, score) 元组

    Returns:
        tuple: (tlwhs, ids, scores)
               tlwhs: 边界框数组，形状为 (N, 4)
               ids: 目标ID列表
               scores: 分数列表
    """
    if len(objs) > 0:
        # 将对象列表解压为三个元组
        tlwhs, ids, scores = zip(*objs)
    else:
        # 如果对象列表为空，初始化为空列表
        tlwhs, ids, scores = [], [], []

    # 将边界框转换为numpy数组并重塑为 (N, 4) 形状
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores


class MOTEvaluator(object):
    def __init__(self, data_root, seq_name, data_type):
        """
        MOT（多目标跟踪）评估器类

        Args:
            data_root: 数据根目录
            seq_name: 序列名称
            data_type: 数据类型，应为'mot'
        """
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        try:
            import motmetrics as mm
            mm.lap.default_solver = 'lap'
        except Exception as e:
            raise RuntimeError(
                'Unable to use MOT metric, please install motmetrics, for example: `pip install motmetrics`, see https://github.com/longcw/py-motmetrics'
            )
        self.reset_accumulator()

    def load_annotations(self):
        """
        加载注释文件
        """
        assert self.data_type == 'mot'
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt',
                                   'gt.txt')
        if not os.path.exists(gt_filename):
            print(
                "gt_filename '{}' of MOTEvaluator is not exist, so the MOTA will be -INF."
            )
        self.gt_frame_dict = read_mot_results(gt_filename, is_gt=True)
        self.gt_ignore_frame_dict = read_mot_results(
            gt_filename, is_ignore=True)

    def reset_accumulator(self):
        """
        重置累积器
        """
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        """
        评估单个帧

        Args:
            frame_id: 帧ID
            trk_tlwhs: 跟踪结果的边界框坐标列表 (top-left x, top-left y, width, height)
            trk_ids: 跟踪结果的ID列表
            rtn_events: 是否返回事件

        Returns:
            events: 事件列表（如果rtn_events为True）
        """
        # 复制跟踪结果以避免修改原始数据
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # 获取地面真值
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # 获取忽略区域
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # 移除被忽略的结果
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(
            ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # 获取距离矩阵
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # 更新累积器
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc,
                                                            'last_mot_events'):
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    def eval_file(self, filename):
        """
        评估整个文件

        Args:
            filename: 结果文件路径

        Returns:
            累积器对象
        """
        self.reset_accumulator()

        result_frame_dict = read_mot_results(filename, is_gt=False)
        frames = sorted(list(set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs,
                    names,
                    metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1',
                             'precision', 'recall')):
        """
        获取评估摘要

        Args:
            accs: 累积器列表
            names: 名称列表
            metrics: 指标列表

        Returns:
            摘要数据框
        """
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs, metrics=metrics, names=names, generate_overall=True)
        return summary

    @staticmethod
    def save_summary(summary, filename):
        """
        保存摘要到文件

        Args:
            summary: 摘要数据框
            filename: 保存文件路径
        """
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


class MOTMetric(Metric):
    def __init__(self, save_summary=False):
        """
        MOT（多目标跟踪）指标类

        Args:
            save_summary (bool): 是否保存评估摘要到Excel文件
        """
        super(MOTMetric, self).__init__()
        self.save_summary = save_summary
        self.MOTEvaluator = MOTEvaluator
        self.result_root = None
        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.accs = []  # 存储累积器列表
        self.seqs = []  # 存储序列名称列表

    def update(self, data_root, seq, data_type, result_root, result_filename):
        """
        更新指标统计值

        Args:
            data_root (str): 数据根目录
            seq (str): 序列名称
            data_type (str): 数据类型
            result_root (str): 结果根目录
            result_filename (str): 结果文件路径
        """
        evaluator = self.MOTEvaluator(data_root, seq, data_type)
        self.accs.append(evaluator.eval_file(result_filename))
        self.seqs.append(seq)
        self.result_root = result_root

    def accumulate(self):
        """
        汇总指标统计值
        """
        # 使用MOTChallenge标准指标
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        # 计算所有序列的汇总结果
        summary = self.MOTEvaluator.get_summary(self.accs, self.seqs, metrics)
        # 格式化摘要结果
        self.strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        # 如果需要保存摘要，则保存到Excel文件
        if self.save_summary:
            self.MOTEvaluator.save_summary(
                summary, os.path.join(self.result_root, 'summary.xlsx'))

    def log(self):
        """
        记录指标结果
        """
        print(self.strsummary)

    def get_results(self):
        """
        获取指标结果

        Returns:
            格式化的字符串摘要结果
        """
        return self.strsummary

    def name(self):
        return self.__class__.__name__


class JDEDetMetric(Metric):
    # 注意，此检测AP指标与COCOMetric或VOCMetric不同，
    # 边界框坐标未缩放到原始图像
    def __init__(self, overlap_thresh=0.5):
        """
        JDE检测指标类

        Args:
            overlap_thresh (float): IOU重叠阈值，默认0.5
        """
        super(JDEDetMetric, self).__init__()
        self.overlap_thresh = overlap_thresh
        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.AP_accum = np.zeros(1)  # AP累积值
        self.AP_accum_count = np.zeros(1)  # AP累积计数

    def update(self, inputs, outputs):
        """
        更新指标统计值

        Args:
            inputs: 输入数据，包含地面真值
            outputs: 输出数据，包含预测结果
        """
        # 提取预测的边界框、置信度和标签
        bboxes = outputs['bbox'][:, 2:].cpu().numpy()
        scores = outputs['bbox'][:, 1].cpu().numpy()
        labels = outputs['bbox'][:, 0].cpu().numpy()
        bbox_lengths = outputs['bbox_num'].cpu().numpy()

        # 如果预测框为空，直接返回
        if bboxes.shape[0] == 1 and bboxes.sum() == 0.0:
            return

        # 获取地面真值
        gt_boxes = inputs['gt_bbox'].cpu().numpy()[0]
        gt_labels = inputs['gt_class'].cpu().numpy()[0]

        # 如果地面真值为空，直接返回
        if gt_labels.shape[0] == 0:
            return

        # 计算预测与地面真值的匹配情况
        correct = []
        detected = []
        for i in range(bboxes.shape[0]):
            obj_pred = 0  # 预测类别（JDE中检测器只预测一个类别）
            pred_bbox = bboxes[i].reshape(1, 4)
            # 计算与地面真值框的IOU
            iou = bbox_iou_np_expand(pred_bbox, gt_boxes, x1y1x2y2=True)[0]
            # 找到最大重叠的索引
            best_i = np.argmax(iou)
            # 如果重叠超过阈值且分类正确且未被检测过，则标记为正确
            if iou[best_i] > self.overlap_thresh and obj_pred == gt_labels[
                best_i] and best_i not in detected:
                correct.append(1)
                detected.append(best_i)
            else:
                correct.append(0)

        # 按类别计算平均精度(AP)
        target_cls = list(gt_labels.T[0])
        AP, AP_class, R, P = ap_per_class(
            tp=correct,
            conf=scores,
            pred_cls=np.zeros_like(scores),
            target_cls=target_cls)
        # 累积AP值和计数
        self.AP_accum_count += np.bincount(AP_class, minlength=1)
        self.AP_accum += np.bincount(AP_class, minlength=1, weights=AP)

    def accumulate(self):
        """
        汇总指标统计值
        """
        print("Accumulating evaluatation results...")
        # 计算平均精度
        self.map_stat = self.AP_accum[0] / (self.AP_accum_count[0] + 1E-16)

    def log(self):
        """
        记录指标结果
        """
        map_stat = 100. * self.map_stat
        print("mAP({:.2f}) = {:.2f}%".format(self.overlap_thresh,
                                             map_stat))

    def get_results(self):
        """
        获取指标结果

        Returns:
            mAP值
        """
        return self.map_stat

    def name(self):
        return self.__class__.__name__


"""
Following code is borrow from https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/eval_kitti_track/evaluate_tracking.py
"""


class tData:
    """
    数据加载的工具类
    """

    def __init__(self, frame=-1, obj_type="unset", truncation=-1, occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1, \
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, track_id=-1):
        """
        构造函数，使用给定的参数初始化对象

        Args:
            frame: 帧编号，默认-1
            obj_type: 对象类型，默认"unset"
            truncation: 截断程度，默认-1
            occlusion: 遮挡程度，默认-1
            obs_angle: 观测角度，默认-10
            x1, y1: 边界框左上角坐标，默认-1
            x2, y2: 边界框右下角坐标，默认-1
            w, h, l: 宽度、高度、长度，默认-1
            X, Y, Z: 3D位置坐标，默认-1000
            yaw: 偏航角，默认-10
            score: 置信度分数，默认-1000
            track_id: 跟踪ID，默认-1
        """
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.tracker = -1

    def __str__(self):
        """
        返回对象的字符串表示

        Returns:
            对象所有属性的字符串表示
        """
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


class KITTIEvaluation(object):
    """ KITTI跟踪统计 (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	- 多目标跟踪准确率 [0,100]
             MOTP	- 多目标跟踪精度 [0,100] (3D) / [td,100] (2D)
             MOTAL	- 使用log10(id-switches)的多目标跟踪准确率 [0,100]

             id-switches - ID切换次数
             fragments   - 分段次数

             MT, PT, ML	- 大部分跟踪、部分跟踪和大部分丢失轨迹的数量

             recall	        - 召回率 = 检测到的目标百分比
             precision	    - 精确率 = 正确检测到的目标百分比
             FAR		    - 每帧的误报数量
             falsepositives - 误报数量 (FP)
             missed         - 错失的目标数量 (FN)
    """

    def __init__(self, result_path, gt_path, min_overlap=0.5, max_truncation=0, \
                 min_height=25, max_occlusion=2, cls="car", \
                 n_frames=[], seqs=[], n_sequences=0):
        """
        初始化KITTI评估器

        Args:
            result_path: 结果路径
            gt_path: 地面真值路径
            min_overlap: 最小边界框重叠
            max_truncation: 最大截断
            min_height: 最小高度
            max_occlusion: 最大遮挡
            cls: 评估类别
            n_frames: 每个序列的帧数
            seqs: 序列名称列表
            n_sequences: 序列数量
        """
        # 获取序列数量和从测试映射中获取每序列的帧数
        # (在提取基准时创建)
        self.gt_path = os.path.join(gt_path, "../labels")
        self.n_frames = n_frames
        self.sequence_name = seqs
        self.n_sequences = n_sequences

        self.cls = cls  # 评估类别，如行人或汽车

        self.result_path = result_path

        # 用于评估的统计和数字
        self.n_gt = 0  # 地面真值检测数减去被忽略的假阴性和真阳性
        self.n_igt = 0  # 被忽略的地面真值检测数
        self.n_gts = [
        ]  # 地面真值检测数减去被忽略的假阴性和真阳性 每序列
        self.n_igts = [
        ]  # 被忽略的地面真值检测数 每序列
        self.n_gt_trajectories = 0
        self.n_gt_seq = []
        self.n_tr = 0  # 跟踪检测数减去被忽略的跟踪检测
        self.n_trs = [
        ]  # 跟踪检测数减去被忽略的跟踪检测 每序列
        self.n_itr = 0  # 被忽略的跟踪检测数
        self.n_itrs = []  # 被忽略的跟踪检测数 每序列
        self.n_igttr = 0  # 被忽略的地面真值检测，其中相应的关联跟踪检测也被忽略
        self.n_tr_trajectories = 0
        self.n_tr_seq = []
        self.MOTA = 0
        self.MOTP = 0
        self.MOTAL = 0
        self.MODA = 0
        self.MODP = 0
        self.MODP_t = []
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0
        self.total_cost = 0
        self.itp = 0  # 被忽略的真阳性数
        self.itps = []  # 被忽略的真阳性数 每序列
        self.tp = 0  # 真阳性数，包括被忽略的真阳性！
        self.tps = [
        ]  # 真阳性数，包括被忽略的真阳性 每序列
        self.fn = 0  # 假阴性数，不包括被忽略的假阴性
        self.fns = [
        ]  # 假阴性数，不包括被忽略的假阴性 每序列
        self.ifn = 0  # 被忽略的假阴性数
        self.ifns = []  # 被忽略的假阴性数 每序列
        self.fp = 0  # 假阳性数
        # 有点复杂，被忽略的假阴性和被忽略的真阳性数
        # 被减去，但如果跟踪检测和地面真值检测
        # 都被忽略，则再次添加此数字以避免重复计算
        self.fps = []  # 上述每序列
        self.mme = 0
        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.min_overlap = min_overlap  # 第三方指标的最小边界框重叠
        self.max_truncation = max_truncation  # 评估对象的最大截断
        self.max_occlusion = max_occlusion  # 评估对象的最大遮挡
        self.min_height = min_height  # 评估对象的最小高度
        self.n_sample_points = 500

        # 这应该足以容纳所有地面真值轨迹
        # 如果需要会扩展，否则会减少
        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

    def loadGroundtruth(self):
        """
        加载地面真值数据

        Returns:
            是否成功加载
        """
        try:
            self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def loadTracker(self):
        """
        加载跟踪器数据

        Returns:
            是否成功加载
        """
        try:
            if not self._loadData(
                    self.result_path, cls=self.cls, loading_groundtruth=False):
                return False
        except IOError:
            return False
        return True

    def _loadData(self,
                  root_dir,
                  cls,
                  min_score=-1000,
                  loading_groundtruth=False):
        """
        通用加载器，用于加载地面真值和跟踪数据。
        使用loadGroundtruth()或loadTracker()来加载这些数据。
        从文本文件加载KITTI格式的检测。

        Args:
            root_dir: 根目录
            cls: 类别
            min_score: 最小分数
            loading_groundtruth: 是否加载地面真值

        Returns:
            是否成功加载
        """
        # 构造对象检测对象来保存检测数据
        t_data = tData()
        data = []
        eval_2d = True
        eval_3d = True

        seq_data = []
        n_trajectories = 0
        n_trajectories_seq = []
        for seq, s_name in enumerate(self.sequence_name):
            i = 0
            filename = os.path.join(root_dir, "%s.txt" % s_name)
            f = open(filename, "r")

            f_data = [
                [] for x in range(self.n_frames[seq])
            ]  # 当前集合只有1059个条目，充分长度会随时检查
            ids = []
            n_in_seq = 0
            id_frame_cache = []
            for line in f:
                # KITTI跟踪基准数据格式：
                # (帧,轨迹ID,对象类型,截断,遮挡,阿尔法,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
                line = line.strip()
                fields = line.split(" ")
                # 应该加载的类别（忽略相邻类别）
                if "car" in cls.lower():
                    classes = ["car", "van"]
                elif "pedestrian" in cls.lower():
                    classes = ["pedestrian", "person_sitting"]
                else:
                    classes = [cls.lower()]
                classes += ["dontcare"]
                if not any([s for s in classes if s in fields[2].lower()]):
                    continue
                # 从表格获取字段
                t_data.frame = int(float(fields[0]))  # 帧
                t_data.track_id = int(float(fields[1]))  # ID
                t_data.obj_type = fields[
                    2].lower()  # 对象类型 [car, pedestrian, cyclist, ...]
                t_data.truncation = int(
                    float(fields[3]))  # 截断 [-1,0,1,2]
                t_data.occlusion = int(
                    float(fields[4]))  # 遮挡  [-1,0,1,2]
                t_data.obs_angle = float(fields[5])  # 观测角度 [rad]
                t_data.x1 = float(fields[6])  # 左   [px]
                t_data.y1 = float(fields[7])  # 上    [px]
                t_data.x2 = float(fields[8])  # 右  [px]
                t_data.y2 = float(fields[9])  # 下 [px]
                t_data.h = float(fields[10])  # 高度 [m]
                t_data.w = float(fields[11])  # 宽度  [m]
                t_data.l = float(fields[12])  # 长度 [m]
                t_data.X = float(fields[13])  # X [m]
                t_data.Y = float(fields[14])  # Y [m]
                t_data.Z = float(fields[15])  # Z [m]
                t_data.yaw = float(fields[16])  # 偏航角 [rad]
                if not loading_groundtruth:
                    if len(fields) == 17:
                        t_data.score = -1
                    elif len(fields) == 18:
                        t_data.score = float(fields[17])  # 检测分数
                    else:
                        print("file is not in KITTI format")
                        return

                # 不考虑标记为无效的对象
                if t_data.track_id is -1 and t_data.obj_type != "dontcare":
                    continue

                idx = t_data.frame
                # 检查帧数据长度是否足够
                if idx >= len(f_data):
                    print("extend f_data", idx, len(f_data))
                    f_data += [[] for x in range(max(500, idx - len(f_data)))]
                try:
                    id_frame = (t_data.frame, t_data.track_id)
                    if id_frame in id_frame_cache and not loading_groundtruth:
                        print(
                            "track ids are not unique for sequence %d: frame %d"
                            % (seq, t_data.frame))
                        print(
                            "track id %d occurred at least twice for this frame"
                            % t_data.track_id)
                        print("Exiting...")
                        # continue # 这允许评估非唯一结果文件
                        return False
                    id_frame_cache.append(id_frame)
                    f_data[t_data.frame].append(copy.copy(t_data))
                except:
                    print(len(f_data), idx)
                    raise

                if t_data.track_id not in ids and t_data.obj_type != "dontcare":
                    ids.append(t_data.track_id)
                    n_trajectories += 1
                    n_in_seq += 1

                # 检查上传的数据是否提供2D和3D评估信息
                if not loading_groundtruth and eval_2d is True and (
                        t_data.x1 == -1 or t_data.x2 == -1 or t_data.y1 == -1 or
                        t_data.y2 == -1):
                    eval_2d = False
                if not loading_groundtruth and eval_3d is True and (
                        t_data.X == -1000 or t_data.Y == -1000 or
                        t_data.Z == -1000):
                    eval_3d = False

            # 只添加存在的帧
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)
            f.close()

        if not loading_groundtruth:
            self.tracker = seq_data
            self.n_tr_trajectories = n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories == 0:
                return False
        else:
            # 分割地面真值和DontCare区域
            self.dcareas = []
            self.groundtruth = []
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [], []
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g, dc = [], []
                    for gg in all_gt:
                        if gg.obj_type == "dontcare":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq = n_trajectories_seq
            self.n_gt_trajectories = n_trajectories
        return True

    def boxoverlap(self, a, b, criterion="union"):
        """
        boxoverlap 计算KITTI格式中边界框a和b的交并比。
        如果criterion是'union'，重叠 = (a交b) / (a并b)。
        如果criterion是'a'，重叠 = (a交b) / a，其中b应该是DontCare区域。

        Args:
            a: 边界框a
            b: 边界框b
            criterion: 计算标准，默认为'union'

        Returns:
            重叠率
        """
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        w = x2 - x1
        h = y2 - y1

        if w <= 0. or h <= 0.:
            return 0.
        inter = w * h
        aarea = (a.x2 - a.x1) * (a.y2 - a.y1)
        barea = (b.x2 - b.x1) * (b.y2 - b.y1)
        # 交并比重叠
        if criterion.lower() == "union":
            o = inter / float(aarea + barea - inter)
        elif criterion.lower() == "a":
            o = float(inter) / float(aarea)
        else:
            raise TypeError("未知的criterion类型")
        return o

    def compute3rdPartyMetrics(self):
        """
        计算定义的指标
            - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
              MOTA, MOTAL, MOTP
            - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
              MT/PT/ML
        """
        # 构造Munkres对象用于匈牙利方法关联
        hm = Munkres()
        max_cost = 1e9

        # 遍历所有帧并关联地面真值和跟踪器结果
        # groundtruth和tracker包含每帧的列表，包含KITTI格式检测的列表
        fr, ids = 0, 0
        for seq_idx in range(len(self.groundtruth)):
            seq_gt = self.groundtruth[seq_idx]
            seq_dc = self.dcareas[seq_idx]  # 不关心区域
            seq_tracker = self.tracker[seq_idx]
            seq_trajectories = defaultdict(list)
            seq_ignored = defaultdict(list)

            # 当前序列的统计，检查__init__中的相应变量注释以了解其含义
            seqtp = 0
            seqitp = 0
            seqfn = 0
            seqifn = 0
            seqfp = 0
            seqigt = 0
            seqitr = 0

            last_ids = [[], []]
            n_gts = 0
            n_trs = 0

            for f in range(len(seq_gt)):
                g = seq_gt[f]
                dc = seq_dc[f]

                t = seq_tracker[f]
                # 统计地面真值和跟踪器对象总数
                self.n_gt += len(g)
                self.n_tr += len(t)

                n_gts += len(g)
                n_trs += len(t)

                # 使用匈牙利方法关联，使用boxoverlap 0..1作为成本
                # 构建成本矩阵
                cost_matrix = []
                this_ids = [[], []]
                for gg in g:
                    # 保存当前ID
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker = -1
                    gg.id_switch = 0
                    gg.fragmentation = 0
                    cost_row = []
                    for tt in t:
                        # 重叠 == 1 是成本 ==0
                        c = 1 - self.boxoverlap(gg, tt)
                        # 边界框重叠的门控
                        if c <= self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost)  # = 1e9
                    cost_matrix.append(cost_row)
                    # 所有地面真值轨迹最初都没有关联
                    # 扩展地面真值轨迹列表（合并列表）
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)

                if len(g) is 0:
                    cost_matrix = [[]]
                # 关联
                association_matrix = hm.compute(cost_matrix)

                # 用于完整性检查和MODP计算的临时变量
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc = 0  # 这将汇总所有真阳性的重叠
                tmpcs = [0] * len(
                    g)  # 这将保存所有真阳性的重叠
                # 原因是某些真阳性稍后可能被忽略
                # 因此对应重叠可以
                # 从tmpc中减去用于MODP计算

                # 跟踪器ID和地面真值ID的映射
                for row, col in association_matrix:
                    # 应用边界框重叠的门控
                    c = cost_matrix[row][col]
                    if c < max_cost:
                        g[row].tracker = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid = True
                        g[row].distance = c
                        self.total_cost += 1 - c
                        tmpc += 1 - c
                        tmpcs[row] = 1 - c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # 真阳性仅是有效关联
                        self.tp += 1
                        tmptp += 1
                    else:
                        g[row].tracker = -1
                        self.fn += 1
                        tmpfn += 1

                # 关联跟踪器和DontCare区域
                # 忽略相邻类的跟踪器
                nignoredtracker = 0  # 被忽略的跟踪器检测数
                ignoredtrackers = dict()  # 将关联track_id与-1
                # 如果未忽略则为-1，如果忽略则为1
                # 这用于避免双计数被忽略
                # 情况，见下一个循环

                for tt in t:
                    ignoredtrackers[tt.track_id] = -1
                    # 如果检测属于相邻类别或小于等于最小高度则忽略
                    tt_height = abs(tt.y1 - tt.y2)
                    if ((self.cls == "car" and tt.obj_type == "van") or
                        (self.cls == "pedestrian" and
                         tt.obj_type == "person_sitting") or
                        tt_height <= self.min_height) and not tt.valid:
                        nignoredtracker += 1
                        tt.ignored = True
                        ignoredtrackers[tt.track_id] = 1
                        continue
                    for d in dc:
                        overlap = self.boxoverlap(tt, d, "a")
                        if overlap > 0.5 and not tt.valid:
                            tt.ignored = True
                            nignoredtracker += 1
                            ignoredtrackers[tt.track_id] = 1
                            break

                # 检查被忽略的FN/TP（截断或相邻对象类别）
                ignoredfn = 0  # 被忽略的假阴性数
                nignoredtp = 0  # 被忽略的真阳性数
                nignoredpairs = 0  # 被忽略的对数，即一个被忽略的真阳性
                # 其关联的跟踪器检测已经被忽略

                gi = 0
                for gg in g:
                    if gg.tracker < 0:
                        if gg.occlusion > self.max_occlusion or gg.truncation > self.max_truncation \
                                or (self.cls == "car" and gg.obj_type == "van") or (
                                self.cls == "pedestrian" and gg.obj_type == "person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1

                    elif gg.tracker >= 0:
                        if gg.occlusion > self.max_occlusion or gg.truncation > self.max_truncation \
                                or (self.cls == "car" and gg.obj_type == "van") or (
                                self.cls == "pedestrian" and gg.obj_type == "person_sitting"):

                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1

                            # 如果关联的跟踪器检测已经被忽略，
                            # 我们想要避免双计数被忽略的检测
                            if ignoredtrackers[gg.tracker] > 0:
                                nignoredpairs += 1

                            # 为计算MODP，被忽略检测的重叠
                            # 从tmpc中减去
                            tmpc -= tmpcs[gi]
                    gi += 1

                # 下面可能令人困惑，查看__init__中的注释
                # 以了解各个统计代表什么

                # 通过被忽略的TP数量校正TP，由于截断
                # 被忽略的TP在可视化中显示为跟踪
                tmptp -= nignoredtp

                # 统计被忽略的真阳性数
                self.itp += nignoredtp

                # 调整考虑的地面真值对象数量
                self.n_gt -= (ignoredfn + nignoredtp)

                # 统计被忽略的地面真值对象数
                self.n_igt += ignoredfn + nignoredtp

                # 统计被忽略的跟踪器对象数
                self.n_itr += nignoredtracker

                # 统计被忽略的对数，即同时被忽略的关联跟踪器和
                # 地面真值对象
                self.n_igttr += nignoredpairs

                # 假阴性 = 超过关联阈值的关联gt bboxes + 非关联gt bboxes
                tmpfn += len(g) - len(association_matrix) - ignoredfn
                self.fn += len(g) - len(association_matrix) - ignoredfn
                self.ifn += ignoredfn

                # 假阳性 = 跟踪器bboxes - 关联跟踪器bboxes
                # 错配 (mme_t)
                tmpfp += len(
                    t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs
                self.fp += len(
                    t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs

                # 更新序列数据
                seqtp += tmptp
                seqitp += nignoredtp
                seqfp += tmpfp
                seqfn += tmpfn
                seqifn += ignoredfn
                seqigt += ignoredfn + nignoredtp
                seqitr += nignoredtracker

                # 完整性检查
                # - 真阳性数减去被忽略的真阳性数
                #   应该大于或等于0
                # - 假阴性数应该大于或等于0
                # - 假阳性数需要大于或等于0
                #   否则被忽略的检测可能被双计数
                # - 计数的真阳性数（加上被忽略的）
                #   和计数的假阴性数（加上被忽略的）
                #   应该匹配地面真值对象总数
                # - 计数的真阳性数（加上被忽略的）
                #   和计数的假阳性数
                #   加上被忽略的跟踪器检测数应该
                #   匹配跟踪器检测总数；注意nignoredpairs在此处减去
                #   以避免在nignoredtp和nignoredtracker中双计数被忽略的检测
                if tmptp < 0:
                    print(tmptp, nignoredtp)
                    raise NameError("出错了！TP是负数")
                if tmpfn < 0:
                    print(tmpfn,
                          len(g),
                          len(association_matrix), ignoredfn, nignoredpairs)
                    raise NameError("出错了！FN是负数")
                if tmpfp < 0:
                    print(tmpfp,
                          len(t), tmptp, nignoredtracker, nignoredtp,
                          nignoredpairs)
                    raise NameError("出错了！FP是负数")
                if tmptp + tmpfn is not len(g) - ignoredfn - nignoredtp:
                    print("seqidx", seq_idx)
                    print("frame ", f)
                    print("TP    ", tmptp)
                    print("FN    ", tmpfn)
                    print("FP    ", tmpfp)
                    print("nGT   ", len(g))
                    print("nAss  ", len(association_matrix))
                    print("ign GT", ignoredfn)
                    print("ign TP", nignoredtp)
                    raise NameError(
                        "出错了！nGroundtruth不是TP+FN")
                if tmptp + tmpfp + nignoredtp + nignoredtracker - nignoredpairs is not len(
                        t):
                    print(seq_idx, f, len(t), tmptp, tmpfp)
                    print(len(association_matrix), association_matrix)
                    raise NameError(
                        "出错了！nTracker不是TP+FP")

                # 检查ID切换或分段
                for i, tt in enumerate(this_ids[0]):
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]
                        lid = last_ids[1][idx]
                        if tid != lid and lid != -1 and tid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].id_switch = 1
                                ids += 1
                        if tid != lid and lid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].fragmentation = 1
                                fr += 1

                # 保存当前索引
                last_ids = this_ids
                # 计算MOTP_t
                MODP_t = 1
                if tmptp != 0:
                    MODP_t = tmpc / float(tmptp)
                self.MODP_t.append(MODP_t)

            # 移除当前gt轨迹的空列表
            self.gt_trajectories[seq_idx] = seq_trajectories
            self.ign_trajectories[seq_idx] = seq_ignored

            # 收集"每序列"统计的统计数据。
            self.n_gts.append(n_gts)
            self.n_trs.append(n_trs)
            self.tps.append(seqtp)
            self.itps.append(seqitp)
            self.fps.append(seqfp)
            self.fns.append(seqfn)
            self.ifns.append(seqifn)
            self.n_igts.append(seqigt)
            self.n_itrs.append(seqitr)

        # 为所有地面真值轨迹计算MT/PT/ML, fragments, idswitches
        n_ignored_tr_total = 0
        for seq_idx, (
                seq_trajectories, seq_ignored
        ) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
            if len(seq_trajectories) == 0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0] * 5
            n_ignored_tr = 0
            for g, ign_g in zip(seq_trajectories.values(),
                                seq_ignored.values()):
                # 此gt轨迹的所有帧都被忽略
                if all(ign_g):
                    n_ignored_tr += 1
                    n_ignored_tr_total += 1
                    continue
                # 此gt轨迹的所有帧都未分配给任何检测
                if all([this == -1 for this in g]):
                    tmpML += 1
                    self.ML += 1
                    continue
                # 计算轨迹中的跟踪帧数
                last_id = g[0]
                # 第一个检测（必须在gt_trajectories中）始终被跟踪
                tracked = 1 if g[0] >= 0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1, len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt += 1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[
                        f - 1] != -1:
                        tmpId_switches += 1
                        self.id_switches += 1
                    if f < len(g) - 1 and g[f - 1] != g[
                        f] and last_id != -1 and g[f] != -1 and g[f +
                                                                  1] != -1:
                        tmpFragments += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # 处理最后一帧；跟踪状态在for循环中处理(g[f]!=-1)
                if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[
                    f] != -1 and not ign_g[f]:
                    tmpFragments += 1
                    self.fragments += 1

                # 计算MT/PT/ML
                tracking_ratio = tracked / float(len(g) - sum(ign_g))
                if tracking_ratio > 0.8:
                    tmpMT += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML += 1
                    self.ML += 1
                else:  # 0.2 <= tracking_ratio <= 0.8
                    tmpPT += 1
                    self.PT += 1

        if (self.n_gt_trajectories - n_ignored_tr_total) == 0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

        # 精确率/召回率等
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)
        if (self.recall + self.precision) == 0:
            self.F1 = 0.
        else:
            self.F1 = 2. * (self.precision * self.recall) / (
                    self.precision + self.recall)
        if sum(self.n_frames) == 0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp / float(sum(self.n_frames))

        # 计算CLEARMOT
        if self.n_gt == 0:
            self.MOTA = -float("inf")
            self.MODA = -float("inf")
        else:
            self.MOTA = 1 - (self.fn + self.fp + self.id_switches
                             ) / float(self.n_gt)
            self.MODA = 1 - (self.fn + self.fp) / float(self.n_gt)
        if self.tp == 0:
            self.MOTP = float("inf")
        else:
            self.MOTP = self.total_cost / float(self.tp)
        if self.n_gt != 0:
            if self.id_switches == 0:
                self.MOTAL = 1 - (self.fn + self.fp + self.id_switches
                                  ) / float(self.n_gt)
            else:
                self.MOTAL = 1 - (self.fn + self.fp +
                                  math.log10(self.id_switches)
                                  ) / float(self.n_gt)
        else:
            self.MOTAL = -float("inf")
        if sum(self.n_frames) == 0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t) / float(sum(self.n_frames))
        return True

    def createSummary(self):
        """
        创建评估摘要

        Returns:
            摘要字符串
        """
        summary = ""
        summary += "跟踪评估摘要".center(80, "=") + "\n"
        summary += self.printEntry("多目标跟踪准确率 (MOTA)",
                                   self.MOTA) + "\n"
        summary += self.printEntry("多目标跟踪精确率 (MOTP)",
                                   self.MOTP) + "\n"
        summary += self.printEntry("多目标跟踪准确率 (MOTAL)",
                                   self.MOTAL) + "\n"
        summary += self.printEntry("多目标检测准确率 (MODA)",
                                   self.MODA) + "\n"
        summary += self.printEntry("多目标检测精确率 (MODP)",
                                   self.MODP) + "\n"
        summary += "\n"
        summary += self.printEntry("召回率", self.recall) + "\n"
        summary += self.printEntry("精确率", self.precision) + "\n"
        summary += self.printEntry("F1", self.F1) + "\n"
        summary += self.printEntry("误报率", self.FAR) + "\n"
        summary += "\n"
        summary += self.printEntry("大部分跟踪", self.MT) + "\n"
        summary += self.printEntry("部分跟踪", self.PT) + "\n"
        summary += self.printEntry("大部分丢失", self.ML) + "\n"
        summary += "\n"
        summary += self.printEntry("真阳性", self.tp) + "\n"
        # summary += self.printEntry("每序列真阳性", self.tps) + "\n"
        summary += self.printEntry("被忽略的真阳性", self.itp) + "\n"
        # summary += self.printEntry("每序列被忽略的真阳性", self.itps) + "\n"

        summary += self.printEntry("假阳性", self.fp) + "\n"
        # summary += self.printEntry("每序列假阳性", self.fps) + "\n"
        summary += self.printEntry("假阴性", self.fn) + "\n"
        # summary += self.printEntry("每序列假阴性", self.fns) + "\n"
        summary += self.printEntry("ID切换", self.id_switches) + "\n"
        self.fp = self.fp / self.n_gt
        self.fn = self.fn / self.n_gt
        self.id_switches = self.id_switches / self.n_gt
        summary += self.printEntry("假阳性比例", self.fp) + "\n"
        # summary += self.printEntry("每序列假阳性", self.fps) + "\n"
        summary += self.printEntry("假阴性比例", self.fn) + "\n"
        # summary += self.printEntry("每序列假阴性", self.fns) + "\n"
        summary += self.printEntry("被忽略的假阴性比例",
                                   self.ifn) + "\n"

        # summary += self.printEntry("每序列被忽略的假阴性", self.ifns) + "\n"
        summary += self.printEntry("错失目标", self.fn) + "\n"
        summary += self.printEntry("ID切换", self.id_switches) + "\n"
        summary += self.printEntry("分段", self.fragments) + "\n"
        summary += "\n"
        summary += self.printEntry("地面真值对象 (总计)", self.n_gt +
                                   self.n_igt) + "\n"
        # summary += self.printEntry("每序列地面真值对象", self.n_gts) + "\n"
        summary += self.printEntry("被忽略的地面真值对象",
                                   self.n_igt) + "\n"
        # summary += self.printEntry("每序列被忽略的地面真值对象", self.n_igts) + "\n"
        summary += self.printEntry("地面真值轨迹",
                                   self.n_gt_trajectories) + "\n"
        summary += "\n"
        summary += self.printEntry("跟踪器对象 (总计)", self.n_tr) + "\n"
        # summary += self.printEntry("每序列跟踪器对象", self.n_trs) + "\n"
        summary += self.printEntry("被忽略的跟踪器对象", self.n_itr) + "\n"
        # summary += self.printEntry("每序列被忽略的跟踪器对象", self.n_itrs) + "\n"
        summary += self.printEntry("跟踪器轨迹",
                                   self.n_tr_trajectories) + "\n"
        # summary += "\n"
        # summary += self.printEntry("与被忽略的地面真值对象关联的被忽略跟踪器对象", self.n_igttr) + "\n"
        summary += "=" * 80
        return summary

    def printEntry(self, key, val, width=(70, 10)):
        """
        以表格方式漂亮地打印条目。

        Args:
            key: 键
            val: 值
            width: 宽度，默认为(70, 10)

        Returns:
            格式化字符串
        """
        s_out = key.ljust(width[0])
        if type(val) == int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val) == float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out

    def saveToStats(self, save_summary):
        """
        将统计信息保存到空白分隔的文件中。

        Args:
            save_summary: 是否保存摘要

        Returns:
            摘要字符串
        """
        summary = self.createSummary()
        if save_summary:
            filename = os.path.join(self.result_path,
                                    "summary_%s.txt" % self.cls)
            dump = open(filename, "w+")
            dump.write(summary)
            dump.close()
        return summary


class KITTIMOTMetric(Metric):
    def __init__(self, save_summary=True):
        """
        KITTI多目标跟踪指标类

        Args:
            save_summary (bool): 是否保存评估摘要到文件
        """
        super().__init__()
        self.save_summary = save_summary
        self.MOTEvaluator = KITTIEvaluation  # 使用KITTIEvaluation作为评估器
        self.result_root = None
        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.seqs = []  # 序列名称列表
        self.n_sequences = 0  # 序列数量
        self.n_frames = []  # 每个序列的帧数列表
        self.strsummary = ''  # 字符串摘要结果

    def update(self, data_root, seq, data_type, result_root, result_filename):
        """
        更新指标统计值

        Args:
            data_root (str): 数据根目录
            seq (str): 序列名称
            data_type (str): 数据类型，应为'kitti'
            result_root (str): 结果根目录
            result_filename (str): 结果文件路径
        """
        assert data_type == 'kitti', "data_type should 'kitti'"
        self.result_root = result_root
        self.gt_path = data_root
        # 计算当前序列的最大帧数
        gt_path = '{}/../labels/{}.txt'.format(data_root, seq)
        gt = open(gt_path, "r")
        max_frame = 0
        for line in gt:
            line = line.strip()
            line_list = line.split(" ")
            if int(line_list[0]) > max_frame:
                max_frame = int(line_list[0])
        rs = open(result_filename, "r")
        for line in rs:
            line = line.strip()
            line_list = line.split(" ")
            if int(line_list[0]) > max_frame:
                max_frame = int(line_list[0])
        gt.close()
        rs.close()
        # 更新统计信息
        self.n_frames.append(max_frame + 1)  # 帧数加1（因为帧从0开始计数）
        self.seqs.append(seq)
        self.n_sequences += 1

    def accumulate(self):
        """
        汇总指标统计值并计算评估结果
        """
        print("Processing Result for KITTI Tracking Benchmark")
        # 创建KITTIEvaluation评估器实例
        e = self.MOTEvaluator(result_path=self.result_root, gt_path=self.gt_path, \
                              n_frames=self.n_frames, seqs=self.seqs, n_sequences=self.n_sequences)
        try:
            if not e.loadTracker():
                return
            print("Loading Results - Success")
            # 注意：这里的变量c未定义，应该使用self.MOTEvaluator.cls或其他适当值
            # logger.info("Evaluate Object Class: %s" % c.upper())
            print("Evaluate Object Class: CAR")  # 假设评估CAR类别
        except:
            print("Caught exception while loading result data.")
        if not e.loadGroundtruth():
            raise ValueError("Ground truth not found.")
        print("Loading Groundtruth - Success")
        # 完整性检查
        if len(e.groundtruth) is not len(e.tracker):
            print(
                "The uploaded data does not provide results for every sequence.")
            return False
        print("Loaded %d Sequences." % len(e.groundtruth))
        print("Start Evaluation...")

        if e.compute3rdPartyMetrics():
            self.strsummary = e.saveToStats(self.save_summary)
        else:
            print(
                "There seem to be no true positives or false positives at all in the submitted data."
            )

    def log(self):
        """
        记录指标结果
        """
        print(self.strsummary)

    def get_results(self):
        """
        获取指标结果

        Returns:
            字符串摘要结果
        """
        return self.strsummary

    def name(self):
        return self.__class__.__name__
