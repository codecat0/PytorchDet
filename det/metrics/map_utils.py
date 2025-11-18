#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :map_utils.py
@Author :CodeCat
@Date   :2025/11/17 17:15
"""
import os
import sys
import numpy as np
import itertools
import torch
from loguru import logger

from det.modeling.rbox_utils import poly2rbox_np, rbox_iou


def draw_pr_curve(precision,
                  recall,
                  iou=0.5,
                  out_dir='pr_curve',
                  file_name='precision_recall_curve.jpg'):
    """
    绘制并保存 Precision-Recall (P-R) 曲线图。

    该函数使用 matplotlib 根据给定的精度（precision）和召回率（recall）数据绘制 P-R 曲线，
    并将图像保存到指定目录。

    Args:
        precision (list or np.ndarray): 精度值列表，通常来自 COCO 评估或其他目标检测评估。
        recall (list or np.ndarray): 召回率值列表，应与 precision 长度一致。
        iou (float, optional): 绘制该 P-R 曲线所对应的 IoU 阈值，默认为 0.5。
        out_dir (str, optional): 保存图像的输出目录，默认为 'pr_curve'。
        file_name (str, optional): 保存图像的文件名，默认为 'precision_recall_curve.jpg'。
    """

    # 创建输出目录（如果不存在）
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)

    # 尝试导入 matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.error('Matplotlib not found, please install matplotlib. '
                     'For example: `pip install matplotlib`.')
        raise e

    # 清除当前图形状态，避免重叠
    plt.cla()
    # 创建新图形窗口并命名
    plt.figure('P-R Curve')
    # 设置图形标题，包含 IoU 阈值
    plt.title('Precision/Recall Curve(IoU={})'.format(iou))
    # 设置 x 轴和 y 轴标签
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # 显示网格线
    plt.grid(True)
    # 绘制 P-R 曲线
    plt.plot(recall, precision)
    # 保存图像到指定路径
    plt.savefig(output_path)
    # 关闭图形以释放内存
    plt.close()


def bbox_area(bbox, is_bbox_normalized):
    """
    计算边界框的面积。

    该函数根据边界框的坐标格式（归一化或非归一化）计算其面积。
    边界框的坐标格式应为 [x_min, y_min, x_max, y_max]。

    Args:
        bbox (list or tuple or numpy.ndarray): 边界框坐标，格式为 [x_min, y_min, x_max, y_max]。
        is_bbox_normalized (bool or int): 指示边界框坐标是否已归一化的标志。
                                          如果为 True (或 1)，则坐标范围在 [0, 1] 之间；
                                          如果为 False (或 0)，则坐标为原始像素值。

    Returns:
        float: 边界框的面积。
    """
    # 根据边界框是否归一化来确定加到宽度和高度上的偏置值
    #   - 未归一化: w = x_max - x_min + 1 (符合像素坐标计算)
    #   - 归一化:   w = x_max - x_min + 0 (符合归一化坐标计算)
    norm = 1. - float(is_bbox_normalized)

    # 计算宽度 (x_max - x_min + norm)
    width = bbox[2] - bbox[0] + norm
    # 计算高度 (y_max - y_min + norm)
    height = bbox[3] - bbox[1] + norm

    # 返回面积 (width * height)
    return width * height


def jaccard_overlap(pred, gt, is_bbox_normalized=False):
    """
    计算两个边界框之间的 Jaccard 重叠率（IoU，交并比）。

    该函数根据两个边界框的坐标（pred 和 gt）计算它们的交集面积与并集面积的比值。
    边界框的坐标格式应为 [x_min, y_min, x_max, y_max]。

    Args:
        pred (list or tuple or numpy.ndarray): 预测框坐标，格式为 [x_min, y_min, x_max, y_max]。
        gt (list or tuple or numpy.ndarray): 真实框（ground truth）坐标，格式为 [x_min, y_min, x_max, y_max]。
        is_bbox_normalized (bool or int, optional): 指示边界框坐标是否已归一化的标志。
                                                    如果为 True (或 1)，则坐标范围在 [0, 1] 之间；
                                                    如果为 False (或 0)，则坐标为原始像素值。
                                                    此标志将传递给 `bbox_area` 函数。

    Returns:
        float: 两个边界框之间的 Jaccard 重叠率 (IoU)，范围在 [0, 1] 之间。
               如果两个框不相交，则返回 0.0。
    """
    # 检查两个边界框是否相交
    # 如果 pred 的左边界 >= gt 的右边界，或 pred 的右边界 <= gt 的左边界，
    # 或 pred 的上边界 >= gt 的下边界，或 pred 的下边界 <= gt 的上边界，
    # 则两个框不相交，重叠率为 0。
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
            pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.

    # 计算交集区域的坐标
    inter_xmin = max(pred[0], gt[0])  # 交集区域的左边界
    inter_ymin = max(pred[1], gt[1])  # 交集区域的上边界
    inter_xmax = min(pred[2], gt[2])  # 交集区域的右边界
    inter_ymax = min(pred[3], gt[3])  # 交集区域的下边界

    # 计算交集区域的面积
    inter_area = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax],
                           is_bbox_normalized)

    # 计算预测框和真实框的面积
    pred_area = bbox_area(pred, is_bbox_normalized)
    gt_area = bbox_area(gt, is_bbox_normalized)

    # 计算并集面积：pred_area + gt_area - inter_area
    union_area = pred_area + gt_area - inter_area

    # 计算 Jaccard 重叠率 (IoU) = 交集面积 / 并集面积
    # 使用 float() 确保返回值为浮点数
    overlap = float(inter_area) / union_area

    return overlap


def calc_rbox_iou(pred, gt_poly):
    """
    计算两个旋转框（以多边形顶点形式给出）之间的IoU（交并比）。

    该函数首先计算两个多边形的轴对齐边界框（AABB）的IoU作为快速预筛选。
    如果AABB的IoU大于0，则进一步将多边形转换为旋转框（rbox）格式，
    并使用外部库（如CUDA实现）计算精确的旋转框IoU。

    Args:
        pred (list or numpy.ndarray): 预测框的多边形顶点坐标，格式为 [x1, y1, x2, y2, ..., x8, y8]。
        gt_poly (list or numpy.ndarray): 真实框（ground truth）的多边形顶点坐标，格式为 [x1, y1, x2, y2, ..., x8, y8]。

    Returns:
        float: 预测框和真实框之间的IoU值。
    """
    # 将输入转换为 numpy 数组并重塑为 N x 2 的顶点坐标格式
    pred = np.array(pred, np.float32).reshape(-1, 2)
    gt_poly = np.array(gt_poly, np.float32).reshape(-1, 2)

    # 计算预测框的轴对齐边界框 [x_min, y_min, x_max, y_max]
    pred_rect = [
        np.min(pred[:, 0]),  # x_min
        np.min(pred[:, 1]),  # y_min
        np.max(pred[:, 0]),  # x_max
        np.max(pred[:, 1])  # y_max
    ]

    # 计算真实框的轴对齐边界框 [x_min, y_min, x_max, y_max]
    gt_rect = [
        np.min(gt_poly[:, 0]),  # x_min
        np.min(gt_poly[:, 1]),  # y_min
        np.max(gt_poly[:, 0]),  # x_max
        np.max(gt_poly[:, 1])  # y_max
    ]

    # 计算轴对齐边界框的IoU，用于快速预筛选
    # jaccard_overlap 函数假设输入是 [x_min, y_min, x_max, y_max] 格式
    iou = jaccard_overlap(pred_rect, gt_rect, is_bbox_normalized=False)

    # 如果AABB的IoU为0或负数，则旋转框的IoU也为0
    if iou <= 0:
        return iou

    # AABB相交，需要计算精确的旋转框IoU
    pred_rbox = pred.reshape(-1, 8)
    gt_rbox = gt_poly.reshape(-1, 8)

    iou_tensor = rbox_iou(gt_rbox, pred_rbox)

    iou_result = iou_tensor.cpu().numpy()
    return iou_result[0][0]


def prune_zero_padding(gt_box, gt_label, difficult=None):
    """
    移除填充的零值边界框、标签和困难样本标记。

    在批处理或数据预处理中，有时会用零值填充数组以使其具有固定长度。
    此函数通过查找第一个全为零的边界框来确定有效数据的边界，
    并返回该边界之前的有效部分。

    Args:
        gt_box (list or numpy.ndarray): 真实边界框数组，形状通常为 [N, 4] 或 [N, 8] 等。
                                        如果某个框的所有坐标都为0，则认为它是填充。
        gt_label (list or numpy.ndarray): 与边界框对应的真实标签数组，长度为 N。
        difficult (list or numpy.ndarray, optional): 可选的困难样本标记数组，长度为 N。
                                                     如果提供，也会被截断。

    Returns:
        tuple: 包含三个元素的元组：
            - gt_box_valid (same type as input): 有效的边界框部分。
            - gt_label_valid (same type as input): 有效的标签部分。
            - difficult_valid (same type as input or None): 有效的困难样本标记部分，如果输入为 None 则返回 None。
    """
    # 初始化有效计数器
    valid_cnt = 0

    # 遍历边界框列表
    for i in range(len(gt_box)):
        # 检查当前边界框是否所有元素都为0
        # 使用 `.all()` 方法检查数组的所有元素
        if (gt_box[i] == 0).all():
            # 如果是全零框，则停止计数，当前索引 i 即为第一个无效框的位置
            break
        # 否则，计数器递增，继续检查下一个框
        valid_cnt += 1

    # 根据有效计数截取数组
    # gt_box[:valid_cnt] 会获取从索引 0 到 valid_cnt-1 的元素
    gt_box_valid = gt_box[:valid_cnt]
    gt_label_valid = gt_label[:valid_cnt]

    # 如果提供了 difficult 数组，则同样截取
    difficult_valid = difficult[:valid_cnt] if difficult is not None else None

    # 返回截取后的有效数据
    return gt_box_valid, gt_label_valid, difficult_valid


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    计算平均精度 (Average Precision, AP)，给定召回率 (Recall) 和精确率 (Precision) 曲线。
    方法最初来自 https://github.com/rafaelpadilla/Object-Detection-Metrics。

    Args:
        tp (list): True positives (真正例) 列表，元素为布尔值或0/1。
        conf (list): 置信度分数列表，范围从 0 到 1。
        pred_cls (list): 预测的目标类别列表。
        target_cls (list): 真实的目标类别列表。
    """
    # 将输入列表转换为 numpy 数组，便于后续操作
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(
        pred_cls), np.array(target_cls)

    # 根据置信度分数对所有预测结果进行降序排序
    # np.argsort(-conf) 返回按 -conf 降序排列的索引，即按 conf 降序排列
    i = np.argsort(-conf)
    # 按照排序后的索引重新排列 tp, conf, pred_cls 数组
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到所有出现过的唯一类别（包括预测的和真实标签中的）
    # np.concatenate 合并两个数组，np.unique 去除重复项并排序
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # 为每个类别创建 Precision-Recall 曲线并计算 AP
    # 初始化存储 AP, Precision, Recall 的列表
    ap, p, r = [], [], []
    # 遍历每一个唯一的类别
    for c in unique_classes:
        # 创建一个布尔掩码，标记 pred_cls 中哪些元素等于当前类别 c
        i = pred_cls == c
        # 计算当前类别 c 的真实标签数量（即该类别的总目标数）
        n_gt = sum(target_cls == c)
        # 计算当前类别 c 的预测数量
        n_p = sum(i)

        # 如果该类别既没有预测也没有真实目标，则跳过该类别
        if (n_p == 0) and (n_gt == 0):
            continue
        # 如果该类别没有预测或没有真实目标，则 AP, R, P 都为 0
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)  # AP 为 0
            r.append(0)  # Recall 为 0
            p.append(0)  # Precision 为 0
        else:
            # 计算累积的假正例 (False Positives) 和真正例 (True Positives)
            # 1 - tp[i] 将 True/1 转为 False/0，False/0 转为 True/1，即得到 FP
            # np.cumsum 计算累积和
            fpc = np.cumsum(1 - tp[i])  # 累积 FP
            tpc = np.cumsum(tp[i])  # 累积 TP

            # 计算召回率 (Recall) 曲线
            # recall = TP / (TP + FN) = TP / 总真实数量
            # 为了避免除以零，分母加上一个极小值 1e-16
            recall_curve = tpc / (n_gt + 1e-16)
            # 记录该类别的最终召回率（对应最高置信度阈值）
            r.append(tpc[-1] / (n_gt + 1e-16))

            # 计算精确率 (Precision) 曲线
            # precision = TP / (TP + FP)
            # 同样避免除以零
            precision_curve = tpc / (tpc + fpc)
            # 记录该类别的最终精确率（对应最高置信度阈值）
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # 根据召回率-精确率曲线计算平均精度 (AP)
            # 这里调用了一个外部函数 compute_ap (通常实现为 11-point interpolation 或 VOC07-style)
            ap.append(compute_ap(recall_curve, precision_curve))

    # 返回计算得到的 AP 数组、类别数组（转为 int32 类型）、Recall 数组、Precision 数组
    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(
        p)


def compute_ap(recall, precision):
    """
    计算平均精度 (Average Precision, AP)，给定召回率 (Recall) 和精确率 (Precision) 曲线。
    代码最初来自 https://github.com/rbgirshick/py-faster-rcnn。

    Args:
        recall (list or numpy.ndarray): 召回率曲线的 y 轴值列表或数组。
        precision (list or numpy.ndarray): 精确率曲线的 y 轴值列表或数组。
                                        recall 和 precision 应该是对应点的值。

    Returns:
        float: 按照 py-faster-rcnn 方式计算的平均精度。
    """
    # 正确的 AP 计算方法 (VOC07-style 或 11-point interpolation 之外的标准方法)
    # 首先在曲线的两端添加哨兵值 (sentinel values)
    # 在 recall 数组前添加 0，在末尾添加 1
    mrec = np.concatenate(([0.], recall, [1.]))
    # 在 precision 数组前添加 0，在末尾添加 0
    # (注意：在 recall 为 0 处，precision 通常定义为 1，但这里取 0 是为了后续 envelope 计算)
    mpre = np.concatenate(([0.], precision, [0.]))

    # 计算精确率包络线 (precision envelope)
    # 从后往前遍历，确保包络线是非递增的 (因为 recall 是递增的)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算 PR 曲线下的面积 (即 AP)
    # 寻找 X 轴 (recall) 值发生变化的点，这些点是面积计算的分割点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算面积：sum (Δ recall * prec)
    # 对于每个分割区间 [mrec[j], mrec[j+1]] (其中 j in i)，宽度是 mrec[j+1] - mrec[j]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class DetectionMAP(object):
    """
    计算目标检测的平均精度均值 (mean Average Precision, mAP)。

    目前支持两种计算类型: 11point (VOC07-style) 和 integral (VOC12及以后标准)。

    Args:
        class_num (int): 类别总数。
        overlap_thresh (float): 判断预测框与真实框为真正例 (true positive) 的
            交并比 (IoU) 阈值。默认值为 0.5。
        map_type (str): 计算 mAP 的方法，目前支持 '11point' 和 'integral'。
            默认值为 '11point'。
        is_bbox_normalized (bool): 边界框坐标是否已归一化到 [0, 1] 范围。
            默认值为 False。
        evaluate_difficult (bool): 是否评估困难样本 (difficult bounding boxes)。
            默认值为 False。
        catid2name (dict): 类别ID到类别名称的映射字典。
        classwise (bool): 是否计算每个类别的 AP 并绘制 P-R 曲线。
    """

    def __init__(self,
                 class_num,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 catid2name=None,
                 classwise=False):
        # 保存初始化参数
        self.class_num = class_num
        self.overlap_thresh = overlap_thresh

        # 验证 map_type 参数
        assert map_type in ['11point', 'integral'], \
            "map_type currently only support '11point' and 'integral'"
        self.map_type = map_type
        self.is_bbox_normalized = is_bbox_normalized
        self.evaluate_difficult = evaluate_difficult
        self.classwise = classwise

        # 根据 catid2name 字典构建类别名称列表，用于后续输出
        self.classes = []
        if catid2name is not None:
            for cname in catid2name.values():
                self.classes.append(cname)

        # 重置或初始化内部状态
        self.reset()

    def update(self, bbox, score, label, gt_box, gt_label, difficult=None):
        """
        根据给定的预测结果和真实标签更新内部统计信息。

        Args:
            bbox (list or numpy.ndarray): 预测的边界框列表，每个元素为 [x_min, y_min, x_max, y_max]
                                          或 [x_ctr, y_ctr, width, height, angle] 或 [x1,y1,...,x8,y8]。
            score (list or numpy.ndarray): 预测框的置信度分数列表。
            label (list or numpy.ndarray): 预测框的类别标签列表。
            gt_box (list or numpy.ndarray): 真实的边界框列表。
            gt_label (list or numpy.ndarray): 真实框的类别标签列表。
            difficult (list or numpy.ndarray, optional): 真实框的困难样本标记列表。
        """
        # 如果没有提供 difficult 信息，则默认所有真实框都不是困难样本
        if difficult is None:
            difficult = np.zeros_like(gt_label)

        # 遍历真实标签和困难标记，统计每个类别的真实框数量
        # 只有在评估困难样本或当前框不困难时才计数
        for gtl, diff in zip(gt_label, difficult):
            if self.evaluate_difficult or int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1

        # 遍历所有预测结果
        # visited 数组用于标记真实框是否已被匹配（避免一个真实框被多个预测框匹配）
        visited = [False] * len(gt_label)
        for b, s, l in zip(bbox, score, label):
            # 将预测框转换为列表格式
            pred = b.tolist() if isinstance(b, np.ndarray) else b

            # 寻找与当前预测框类别相同、IoU最大的真实框
            max_idx = -1
            max_overlap = -1.0
            for i, gl in enumerate(gt_label):
                # 只考虑类别匹配的真实框
                if int(gl) == int(l):
                    # 根据真实框的坐标格式选择不同的IoU计算函数
                    if len(gt_box[i]) == 8:  # 假设是8点坐标表示的旋转框
                        overlap = calc_rbox_iou(pred, gt_box[i])
                    else:  # 假设是4点坐标表示的轴对齐框
                        overlap = jaccard_overlap(pred, gt_box[i],
                                                  self.is_bbox_normalized)
                    # 更新最大IoU和对应索引
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = i

            # 根据最大IoU和阈值判断预测结果的正负性
            if max_overlap > self.overlap_thresh:
                # 如果匹配到的真实框可以被评估（非困难或评估困难）
                if self.evaluate_difficult or \
                        int(np.array(difficult[max_idx])) == 0:
                    # 如果该真实框尚未被匹配，则为真正例 (TP)
                    if not visited[max_idx]:
                        # 记录 [置信度, 1.0 (TP)]，并标记该真实框已被匹配
                        self.class_score_poss[int(l)].append([s, 1.0])
                        visited[max_idx] = True
                    else:
                        # 如果该真实框已被匹配，则为假正例 (FP)
                        self.class_score_poss[int(l)].append([s, 0.0])
            else:
                # 如果没有匹配到满足IoU阈值的真实框，则为假正例 (FP)
                self.class_score_poss[int(l)].append([s, 0.0])

    def reset(self):
        """
        重置内部统计信息
        """
        # class_score_poss: 存储每个类别的 [置信度, 是否为TP (1.0 or 0.0)] 列表
        self.class_score_poss = [[] for _ in range(self.class_num)]
        # class_gt_counts: 存储每个类别的真实框总数
        self.class_gt_counts = [0] * self.class_num
        # mAP: 最终计算得到的平均精度均值
        self.mAP = 0.0

    def accumulate(self):
        """
        汇总统计结果并计算 mAP
        """
        mAP = 0.
        valid_cnt = 0  # 有有效预测的类别数量计数器
        eval_results = []  # 存储每个类别的详细评估结果

        # 遍历每个类别
        for score_pos, count in zip(self.class_score_poss, self.class_gt_counts):
            # 如果该类别没有真实框，则跳过
            if count == 0:
                continue
            # 如果该类别没有预测框，则 AP 为 0，计数器加1
            if len(score_pos) == 0:
                valid_cnt += 1
                continue

            # 计算该类别的累积 TP 和 FP 列表
            accum_tp_list, accum_fp_list = self._get_tp_fp_accum(score_pos)

            # 计算精确率 (Precision) 和召回率 (Recall) 列表
            precision = []
            recall = []
            for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                # Precision = TP / (TP + FP)
                precision.append(float(ac_tp) / (ac_tp + ac_fp))
                # Recall = TP / Total GT (for this class)
                recall.append(float(ac_tp) / count)

            # 初始化该类别的 AP
            one_class_ap = 0.0

            # 根据指定的 map_type 计算 AP
            if self.map_type == '11point':
                # VOC07-style: 在11个召回率点 (0, 0.1, ..., 1.0) 上取最大精确率
                max_precisions = [0.] * 11
                start_idx = len(precision) - 1
                for j in range(10, -1, -1):  # 从 10 到 0
                    for i in range(start_idx, -1, -1):  # 从后往前找
                        # 找到第一个 recall < j/10 的点
                        if recall[i] < float(j) / 10.:
                            start_idx = i
                            if j > 0:
                                # 将 j 点的最大精确率传递给 j-1 点
                                max_precisions[j - 1] = max_precisions[j]
                            break
                        else:
                            # 更新 j 点的最大精确率
                            if max_precisions[j] < precision[i]:
                                max_precisions[j] = precision[i]
                # AP 是这11个最大精确率的平均值
                one_class_ap = sum(max_precisions) / 11.
                mAP += one_class_ap
                valid_cnt += 1
            elif self.map_type == 'integral':
                # VOC12及以后标准: 计算 PR 曲线下的面积 (积分)
                import math
                prev_recall = 0.
                for i in range(len(precision)):
                    recall_gap = math.fabs(recall[i] - prev_recall)
                    # 只有当召回率有变化时才计算面积
                    if recall_gap > 1e-6:
                        # 面积近似为 precision[i] * recall_gap
                        one_class_ap += precision[i] * recall_gap
                        prev_recall = recall[i]
                mAP += one_class_ap
                valid_cnt += 1
            else:
                # 不支持的 map_type
                logger.error("Unspported mAP type {}".format(self.map_type))
                sys.exit(1)

            # 保存该类别的详细结果
            eval_results.append({
                'class': self.classes[valid_cnt - 1],  # 使用当前有效类别的名称
                'ap': one_class_ap,
                'precision': precision,
                'recall': recall,
            })

        # 保存所有类别的评估结果
        self.eval_results = eval_results
        # 计算最终的 mAP (所有有效类别的 AP 平均值)
        self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP

    def get_map(self):
        """
        获取 mAP 结果
        """
        if self.mAP is None:
            logger.error("mAP is not calculated.")
        if self.classwise:
            # 如果需要计算每个类别的 AP 和绘制 P-R 曲线
            try:
                from terminaltables import AsciiTable
            except ImportError as e:
                logger.error(
                    'terminaltables not found, plaese install terminaltables. '
                    'for example: `pip install terminaltables`.')
                raise e

            results_per_category = []
            for eval_result in self.eval_results:
                # 添加类别名和 AP 到结果列表
                results_per_category.append(
                    (str(eval_result['class']),
                     '{:0.3f}'.format(float(eval_result['ap']))))
                # 绘制并保存 P-R 曲线
                draw_pr_curve(
                    eval_result['precision'],
                    eval_result['recall'],
                    out_dir='voc_pr_curve',
                    file_name='{}_precision_recall_curve.jpg'.format(
                        eval_result['class']))  # 使用类别名作为文件名

            # 格式化并打印每个类别的 AP 结果表
            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns] for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            logger.info('Per-category of VOC AP: \n{}'.format(table.table))
            logger.info(
                "per-category PR curve has output to voc_pr_curve folder.")
        return self.mAP

    @staticmethod
    def _get_tp_fp_accum(score_pos_list):
        """
        根据 [置信度, 是否为TP (pos)] 记录列表，计算累积的 TP 和 FP 结果。

        Args:
            score_pos_list (list): 格式为 [[score1, pos1], [score2, pos2], ...]
                                   的列表，其中 pos 为 1.0 (TP) 或 0.0 (FP)。

        Returns:
            tuple: 包含累积 TP 列表和累积 FP 列表的元组。
        """
        # 按置信度降序排序
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0  # 累积 TP 计数
        accum_fp = 0  # 累积 FP 计数
        accum_tp_list = []  # 存储每个排序后预测的累积 TP 数
        accum_fp_list = []  # 存储每个排序后预测的累积 FP 数

        # 遍历排序后的列表，计算累积值
        for (score, pos) in sorted_list:
            accum_tp += int(pos)  # 如果是 TP (pos=1)，计数加1
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)  # 如果是 FP (pos=0)，计数加1
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list
