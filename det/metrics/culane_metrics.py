#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :culane_metrics.py
@Author :CodeCat
@Date   :2025/11/21 15:01
"""
import os
import cv2
import numpy as np
import os.path as osp
from functools import partial
from .metrics import Metric
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from loguru import logger



LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}


def draw_lane(lane, img=None, img_shape=None, width=30):
    """
    在图像上绘制车道线

    Args:
        lane: 车道线点的坐标数组，形状为 (N, 2)，每行包含 [x, y] 坐标
        img: 输入图像，如果为None则根据img_shape创建新图像
        img_shape: 图像形状 (height, width, channels)，当img为None时使用
        width: 绘制线的宽度，默认为30

    Returns:
        绘制了车道线的图像
    """
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    # 遍历车道线上的相邻点对，绘制线段
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(
            img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    """
    计算两组车道线之间的离散交叉IOU（交并比）

    Args:
        xs: 第一组车道线，每个车道线为点坐标数组，形状为 (N, 2)
        ys: 第二组车道线，每个车道线为点坐标数组，形状为 (M, 2)
        width: 绘制车道线时的线宽，默认为30像素
        img_shape: 图像形状，默认为 (590, 1640, 3)

    Returns:
        ious: IOU矩阵，形状为 (len(xs), len(ys))，其中 ious[i, j] 表示 xs[i] 与 ys[j] 的IOU值
    """
    # 将每条车道线渲染为二值掩码图像（True/False）
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    # 初始化IOU矩阵
    ious = np.zeros((len(xs), len(ys)))

    # 计算所有车道线对之间的IOU
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            intersection = (x & y).sum()  # 交集像素数
            union = (x | y).sum()  # 并集像素数
            # 避免除零，当并集为0时IOU定义为0
            ious[i, j] = intersection / union if union > 0 else 0.0
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    """
    计算两组车道线之间的连续交叉IOU（交并比），基于几何图形精确计算

    Args:
        xs: 第一组车道线，每个车道线为点坐标数组，形状为 (N, 2)
        ys: 第二组车道线，每个车道线为点坐标数组，形状为 (M, 2)
        width: 车道线宽度（缓冲区半径为 width / 2），默认为30像素
        img_shape: 图像形状，默认为 (590, 1640, 3)

    Returns:
        ious: IOU矩阵，形状为 (len(xs), len(ys))，其中 ious[i, j] 表示 xs[i] 与 ys[j] 的IOU值
    """
    from shapely.geometry import Polygon, LineString
    import numpy as np

    h, w, _ = img_shape
    # 定义图像边界的多边形（用于裁剪）
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])

    # 将每条车道线转换为带宽度的几何区域（缓冲区），并裁剪到图像边界内
    xs = [
        LineString(lane).buffer(
            distance=width / 2., cap_style=1, join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(
            distance=width / 2., cap_style=1, join_style=2).intersection(image)
        for lane in ys
    ]

    # 初始化IOU矩阵
    ious = np.zeros((len(xs), len(ys)))

    # 计算所有车道线几何区域对之间的IOU
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            intersection_area = x.intersection(y).area  # 交集面积
            union_area = x.union(y).area  # 并集面积
            # 避免除零，当并集面积为0时IOU定义为0
            ious[i, j] = intersection_area / union_area if union_area > 0 else 0.0

    return ious


def interp(points, n=50):
    """
    使用样条插值对点序列进行平滑插值

    Args:
        points: 点序列，形状为 (N, 2)，每行包含 [x, y] 坐标
        n: 插值参数，控制插值点的密度，默认为50

    Returns:
        插值后的点序列，形状为 (M, 2)，其中 M > N
    """
    from scipy.interpolate import splprep, splev
    import numpy as np

    # 分离x和y坐标
    x = [x for x, _ in points]
    y = [y for _, y in points]

    # 使用样条参数化准备插值，s=0表示插值（不过拟合），k是样条阶数
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    # 生成更密集的参数向量
    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)

    # 计算插值点
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(590, 1640, 3)):
    """
    计算CULane数据集的评估指标

    Args:
        pred: 预测的车道线列表，每个元素为点坐标数组 (N, 2)
        anno: 地面真值车道线列表，每个元素为点坐标数组 (M, 2)
        width: 车道线绘制宽度，默认为30像素
        iou_thresholds: IOU阈值列表，默认为[0.5]
        official: 是否使用官方离散IOU计算方法，默认为True
        img_shape: 图像形状，默认为(590, 1640, 3)

    Returns:
        dict: 包含每个IOU阈值对应的[tp, fp, fn]的字典
              tp: 真阳性数量
              fp: 假阳性数量
              fn: 假阴性数量
    """
    # 对预测和地面真值车道线进行插值，使其具有更密集的采样点
    interp_pred = np.array(
        [interp(
            pred_lane, n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
    interp_anno = np.array(
        [interp(
            anno_lane, n=5) for anno_lane in anno], dtype=object)  # (4, 50, 2)

    # 根据official参数选择离散或连续IOU计算方法
    if official:
        ious = discrete_cross_iou(
            interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(
            interp_pred, interp_anno, width=width, img_shape=img_shape)

    # 使用匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    # 计算每个IOU阈值下的指标
    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())  # 真阳性：匹配成功且IOU超过阈值
        fp = len(pred) - tp  # 假阳性：预测的车道线中未匹配成功的数量
        fn = len(anno) - tp  # 假阴性：地面真值中未匹配成功的数量
        _metric[thr] = [tp, fp, fn]
    return _metric


def load_culane_img_data(path):
    """
    加载CULane数据集的图像标注数据

    CULane数据集的标注文件中每行代表一条车道线，格式为 x1 y1 x2 y2 x3 y3 ...
    其中每对连续的数字 (xi, yi) 表示车道线上一个点的坐标。

    Args:
        path (str): 标注文件路径

    Returns:
        list: 包含所有车道线的列表，每个车道线是一个点坐标列表 [(x1,y1), (x2,y2), ...]
              只返回包含至少2个点的车道线
    """
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()

    # 将每行按空格分割成字符串列表
    img_data = [line.split() for line in img_data]
    # 将字符串转换为浮点数
    img_data = [list(map(float, lane)) for lane in img_data]
    # 将连续的x,y坐标配对成点 (x, y)
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]
    # 过滤掉点数少于2的车道线
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_culane_data(data_dir, file_list_path):
    """
    加载CULane数据集的所有标注数据

    Args:
        data_dir (str): CULane数据集的根目录路径
        file_list_path (str): 包含图像文件路径列表的文件路径

    Returns:
        list: 包含所有图像标注数据的列表
              每个元素对应一张图像的车道线标注，格式为 [[(x1,y1), (x2,y2), ...], ...]
              其中外层列表的每个元素是一条车道线，内层列表是该车道线上的点坐标
    """
    with open(file_list_path, 'r') as file_list:
        # 读取文件列表并转换为对应的标注文件路径
        filepaths = [
            os.path.join(data_dir,
                         line[1 if line[0] == '/' else 0:].rstrip().replace(
                             '.jpg', '.lines.txt'))
            for line in file_list.readlines()
        ]

    data = []
    # 为每个标注文件加载数据
    for path in filepaths:
        img_data = load_culane_img_data(path)  # 加载单个图像的车道线数据
        data.append(img_data)

    return data


def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=[0.5],
                     width=30,
                     official=True,
                     sequential=False):
    """
    评估预测结果与地面真值之间的指标

    Args:
        pred_dir (str): 预测结果目录路径
        anno_dir (str): 地面真值标注目录路径
        list_path (str): 包含图像文件路径列表的文件路径
        iou_thresholds (list): IOU阈值列表，默认为[0.5]
        width (int): 车道线绘制宽度，默认为30像素
        official (bool): 是否使用官方离散IOU计算方法，默认为True
        sequential (bool): 是否顺序处理（不使用多进程），默认为False

    Returns:
        dict: 包含每个IOU阈值及平均结果的指标字典
              每个阈值的指标包含TP, FP, FN, Precision, Recall, F1
    """
    print('Calculating metric for List: {}'.format(list_path))

    # 加载预测和地面真值数据
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    img_shape = (590, 1640, 3)  # CULane数据集的图像形状

    if sequential:
        # 顺序处理每个图像对
        from functools import partial
        results = map(partial(
            culane_metric,
            width=width,
            official=official,
            iou_thresholds=iou_thresholds,
            img_shape=img_shape),
            predictions,
            annotations)
    else:
        # 使用多进程并行处理
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric,
                                zip(predictions, annotations,
                                    repeat(width),
                                    repeat(iou_thresholds),
                                    repeat(official), repeat(img_shape)))

    # 初始化统计变量
    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}

    # 计算每个IOU阈值下的指标
    for thr in iou_thresholds:
        # 汇总所有图像在当前阈值下的tp, fp, fn
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)

        # 计算精确率、召回率和F1分数
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0

        print('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
              'precision: {}, recall: {}, f1: {}'.format(
            thr, tp, fp, fn, precision, recall, f1))

        # 累积平均值
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 保存当前阈值的指标
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

    # 如果有多个阈值，计算并输出平均结果
    if len(iou_thresholds) > 2:
        print(
            'mean result, total_tp: {}, total_fp: {}, total_fn: {},'
            'precision: {}, recall: {}, f1: {}'.format(
                total_tp, total_fp, total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret


class CULaneMetric(Metric):
    def __init__(self,
                 cfg,
                 output_eval=None,
                 split="test",
                 dataset_dir="dataset/CULane/"):
        """
        CULane数据集指标类

        Args:
            cfg: 配置对象，包含图像尺寸等参数
            output_eval: 评估结果输出目录
            split: 数据集划分，如"test"
            dataset_dir: CULane数据集根目录
        """
        super(CULaneMetric, self).__init__()
        self.output_eval = "evaluation" if output_eval is None else output_eval
        self.dataset_dir = dataset_dir
        self.split = split
        self.list_path = osp.join(dataset_dir, LIST_FILE[split])
        self.predictions = []  # 存储预测的车道线
        self.img_names = []  # 存储图像名称
        self.lanes = []  # 存储地面真值车道线
        self.eval_results = {}  # 存储评估结果
        self.cfg = cfg
        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.predictions = []
        self.img_names = []
        self.lanes = []
        self.eval_results = {}

    def get_prediction_string(self, pred):
        """
        将预测的车道线转换为CULane格式的字符串

        Args:
            pred: 预测的车道线列表，每个元素是一个函数，接受y坐标返回x坐标

        Returns:
            格式化的字符串，每行代表一条车道线的点坐标
        """
        # 在固定y坐标上采样x坐标
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            # 过滤有效范围内的坐标点
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            # 反转顺序（从下到上）
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            # 格式化为字符串
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def accumulate(self):
        """
        汇总指标统计值并计算评估结果
        """
        loss_lines = [[], [], [], []]  # 存储不同数量丢失车道线的图像
        for idx, pred in enumerate(self.predictions):
            # 创建输出目录
            output_dir = os.path.join(self.output_eval,
                                      os.path.dirname(self.img_names[idx]))
            output_filename = os.path.basename(self.img_names[
                                                   idx])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)

            # 记录丢失车道线的图像
            lanes = self.lanes[idx]
            if len(lanes) - len(pred) in [1, 2, 3, 4]:
                loss_lines[len(lanes) - len(pred) - 1].append(self.img_names[
                                                                  idx])

            # 保存预测结果到文件
            with open(os.path.join(output_dir, output_filename),
                      'w') as out_file:
                out_file.write(output)

        # 保存丢失车道线的图像列表
        for i, names in enumerate(loss_lines):
            with open(
                    os.path.join(output_dir, 'loss_{}_lines.txt'.format(i + 1)),
                    'w') as f:
                for name in names:
                    f.write(name + '\n')

        # 按类别评估
        for cate, cate_file in CATEGORYS.items():
            result = eval_predictions(
                self.output_eval,
                self.dataset_dir,
                os.path.join(self.dataset_dir, cate_file),
                iou_thresholds=[0.5],
                official=True)

        # 全局评估（使用0.5到0.95的IOU阈值）
        result = eval_predictions(
            self.output_eval,
            self.dataset_dir,
            self.list_path,
            iou_thresholds=np.linspace(0.5, 0.95, 10),
            official=True)
        self.eval_results['F1@50'] = result[0.5]['F1']  # IOU阈值为0.5时的F1分数
        self.eval_results['result'] = result

    def update(self, inputs, outputs):
        """
        更新指标统计值

        Args:
            inputs: 输入数据，包含图像名称和地面真值车道线
            outputs: 输出数据，包含预测的车道线
        """
        assert len(inputs['img_name']) == len(outputs['lanes'])
        self.predictions.extend(outputs['lanes'])
        self.img_names.extend(inputs['img_name'])
        self.lanes.extend(inputs['lane_line'])

    def log(self):
        """
        记录指标结果
        """
        print(self.eval_results)

    # 获取指标结果的抽象方法
    def get_results(self):
        """
        获取指标结果

        Returns:
            评估结果字典
        """
        return self.eval_results

    def name(self):
        return self.__class__.__name__