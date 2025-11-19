#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :widerface_utils.py
@Author :CodeCat
@Date   :2025/11/19 17:08
"""
import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from loguru import logger


def face_eval_run(model,
                  image_dir,
                  gt_file,
                  pred_dir='output/pred',
                  eval_mode='widerface',
                  multi_scale=False):
    """
    人脸检测模型评估运行函数

    Args:
        model: 人脸检测模型
        image_dir: 图像目录
        gt_file: 真值文件路径
        pred_dir: 预测结果保存目录
        eval_mode: 评估模式，'widerface' 或 'fddb'
        multi_scale: 是否使用多尺度测试
    """
    # 加载真值文件
    with open(gt_file, 'r') as f:
        gt_lines = f.readlines()
    imid2path = []
    pos_gt = 0
    while pos_gt < len(gt_lines):
        name_gt = gt_lines[pos_gt].strip('\n\t').split()[0]
        imid2path.append(name_gt)
        pos_gt += 1
        n_gt = int(gt_lines[pos_gt].strip('\n\t').split()[0])
        pos_gt += 1 + n_gt
    print('The ground truth file load {} images'.format(len(imid2path)))

    dets_dist = OrderedDict()
    for iter_id, im_path in enumerate(imid2path):
        image_path = os.path.join(image_dir, im_path)
        if eval_mode == 'fddb':
            image_path += '.jpg'
        assert os.path.exists(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if multi_scale:
            # 获取缩放因子
            shrink, max_shrink = get_shrink(image.shape[0], image.shape[1])
            # 多尺度检测
            det0 = detect_face(model, image, shrink)
            det1 = flip_test(model, image, shrink)
            [det2, det3] = multi_scale_test(model, image, max_shrink)
            det4 = multi_scale_test_pyramid(model, image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            # 单尺度检测
            dets = detect_face(model, image, 1)

        if eval_mode == 'widerface':
            # 保存WIDER FACE格式结果
            save_widerface_bboxes(image_path, dets, pred_dir)
        else:
            # 保存FDDB格式结果
            dets_dist[im_path] = dets

        if iter_id % 100 == 0:
            print('Test iter {}'.format(iter_id))

    if eval_mode == 'fddb':
        save_fddb_bboxes(dets_dist, pred_dir)

    print("Finish evaluation.")


def face_img_process(image,
                     mean=[104., 117., 123.],
                     std=[127.502231, 127.502231, 127.502231]):
    """
    人脸图像预处理函数

    Args:
        image: 输入图像
        mean: 图像均值，用于减均值操作
        std: 图像标准差，用于除以标准差操作

    Returns:
        预处理后的图像张量
    """
    img = np.array(image)
    img = to_chw(img)
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img /= np.array(std)[:, np.newaxis, np.newaxis].astype('float32')
    img = [img]
    img = np.array(img)
    return img


def get_shrink(height, width):
    """
    计算图像缩放因子，避免内存溢出

    Args:
        height (int): 图像高度
        width (int): 图像宽度

    Returns:
        tuple: (实际缩放因子, 最大缩放因子)
    """
    # 避免内存溢出
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width)) ** 0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width)) ** 0.5

    def get_round(x, loc):
        """
        对数值进行截断处理

        Args:
            x: 输入数值
            loc: 小数点后保留位数

        Returns:
            截断后的数值
        """
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3

    # 根据缩放因子范围进行调整
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5
    elif max_shrink <= 0.1:
        max_shrink = 0.1

    # 最终缩放因子不能超过1
    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


def detect_face(model, image, shrink):
    """
    人脸检测函数

    Args:
        model: 人脸检测模型
        image: 输入图像
        shrink: 图像缩放因子

    Returns:
        检测结果，包含边界框坐标和置信度
    """
    image_shape = [image.shape[0], image.shape[1]]

    # 如果缩放因子不为1，则调整图像大小
    if shrink != 1:
        h, w = int(image_shape[0] * shrink), int(image_shape[1] * shrink)
        image = cv2.resize(image, (w, h))
        image_shape = [h, w]

    # 对图像进行预处理
    img = face_img_process(image)
    image_shape = np.asarray([image_shape])
    scale_factor = np.asarray([[shrink, shrink]])

    # 准备模型输入数据
    data = {
        "image": torch.tensor(
            img, dtype=torch.float32),
        "im_shape": torch.tensor(
            image_shape, dtype=torch.float32),
        "scale_factor": torch.tensor(
            scale_factor, dtype=torch.float32)
    }

    # 设置模型为评估模式并进行推理
    model.eval()
    with torch.no_grad():  # 不计算梯度
        detection = model(data)

    detection = detection['bbox'].cpu().numpy()

    # layout: xmin, ymin, xmax, ymax, score
    if np.prod(detection.shape) == 1:
        print("No face detected")
        return np.array([[0, 0, 0, 0, 0]])

    # 提取检测框坐标和置信度
    det_conf = detection[:, 1]
    det_xmin = detection[:, 2]
    det_ymin = detection[:, 3]
    det_xmax = detection[:, 4]
    det_ymax = detection[:, 5]

    # 组合为最终的检测结果
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
    return det


def flip_test(model, image, shrink):
    """
    翻转测试函数，对图像进行水平翻转后检测人脸，然后将结果转换回原始坐标系

    Args:
        model: 人脸检测模型
        image: 输入图像
        shrink: 图像缩放因子

    Returns:
        翻转测试后的检测结果
    """
    # 对图像进行水平翻转
    img = cv2.flip(image, 1)
    # 在翻转后的图像上进行人脸检测
    det_f = detect_face(model, img, shrink)
    # 创建与翻转检测结果相同形状的数组
    det_t = np.zeros(det_f.shape)
    # 获取图像宽度
    img_width = image.shape[1]
    # 将翻转后的检测框坐标转换回原始坐标系
    det_t[:, 0] = img_width - det_f[:, 2]  # xmin = img_width - xmax
    det_t[:, 1] = det_f[:, 1]  # ymin = ymin
    det_t[:, 2] = img_width - det_f[:, 0]  # xmax = img_width - xmin
    det_t[:, 3] = det_f[:, 3]  # ymax = ymax
    det_t[:, 4] = det_f[:, 4]  # score = score
    return det_t


def multi_scale_test(model, image, max_shrink):
    """
    多尺度测试函数，使用不同尺度的图像进行人脸检测以提高检测精度

    Args:
        model: 人脸检测模型
        image: 输入图像
        max_shrink: 最大缩放因子

    Returns:
        tuple: (小尺度检测结果, 大尺度检测结果)
    """
    # 缩小检测仅用于检测大脸
    st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
    det_s = detect_face(model, image, st)
    # 过滤出较大的检测框
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
        > 30)[0]
    det_s = det_s[index, :]

    # 放大一倍
    bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
    det_b = detect_face(model, image, bt)

    # 对小脸放大更多倍数
    if max_shrink > 2:
        bt *= 2
        while bt < max_shrink:
            det_b = np.row_stack((det_b, detect_face(model, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(model, image, max_shrink)))

    # 放大的图像仅用于检测小脸
    if bt > 1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    # 缩小的图像仅用于检测大脸
    else:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def multi_scale_test_pyramid(model, image, max_shrink):
    """
    使用图像金字塔进行多尺度人脸检测

    Args:
        model: 人脸检测模型
        image: 输入图像
        max_shrink: 最大缩放因子

    Returns:
        综合多个尺度的检测结果
    """
    # 使用0.25倍缩放检测人脸
    det_b = detect_face(model, image, 0.25)
    # 过滤出较大的检测框
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    # 定义多个缩放比例进行检测
    st = [0.75, 1.25, 1.5, 1.75]
    for i in range(len(st)):
        if st[i] <= max_shrink:
            det_temp = detect_face(model, image, st[i])
            # 放大的图像仅用于检测小脸
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            # 缩小的图像仅用于检测大脸
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            # 将当前尺度的检测结果与之前的结果合并
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def bbox_vote(det):
    """
    边界框投票函数，用于合并重叠的检测框

    Args:
        det: 检测结果，形状为 [N, 5]，包含 [xmin, ymin, xmax, ymax, score]

    Returns:
        投票合并后的检测结果
    """
    # 按置信度从高到低排序
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]

    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])

    while det.shape[0] > 0:
        # 计算IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # 非极大值抑制
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue

        # 按置信度加权坐标
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                      axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score

        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    # 限制检测框数量并过滤低置信度结果
    dets = dets[0:750, :]
    keep_index = np.where(dets[:, 4] >= 0.01)[0]
    dets = dets[keep_index, :]
    return dets


def save_widerface_bboxes(image_path, bboxes_scores, output_dir):
    """
    保存WIDER FACE数据集格式的人脸检测结果

    Args:
        image_path: 图像路径
        bboxes_scores: 检测框和置信度结果，形状为 [N, 5]，包含 [xmin, ymin, xmax, ymax, score]
        output_dir: 输出目录
    """
    # 提取图像名称和类别
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]
    odir = os.path.join(output_dir, image_class)

    # 创建输出目录
    if not os.path.exists(odir):
        os.makedirs(odir)

    # 构建输出文件路径
    ofname = os.path.join(odir, '%s.txt' % (image_name[:-4]))
    f = open(ofname, 'w')
    # 写入图像路径和检测框数量
    f.write('{:s}\n'.format(image_class + '/' + image_name))
    f.write('{:d}\n'.format(bboxes_scores.shape[0]))

    # 写入每个检测框的信息
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        # 写入格式：xmin, ymin, width, height, confidence
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
                xmax - xmin + 1), (ymax - ymin + 1), score))
    f.close()
    print("The predicted result is saved as {}".format(ofname))


def save_fddb_bboxes(bboxes_scores,
                     output_dir,
                     output_fname='pred_fddb_res.txt'):
    """
    保存FDDB数据集格式的人脸检测结果

    Args:
        bboxes_scores: 包含图像路径和检测框的字典，格式为 {image_path: [N, 5] array}
        output_dir: 输出目录
        output_fname: 输出文件名

    Returns:
        predict_file: 预测结果文件路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predict_file = os.path.join(output_dir, output_fname)
    f = open(predict_file, 'w')

    # 遍历每个图像及其检测结果
    for image_path, dets in bboxes_scores.items():
        f.write('{:s}\n'.format(image_path))
        f.write('{:d}\n'.format(dets.shape[0]))

        # 写入每个检测框的信息
        for box_score in dets:
            xmin, ymin, xmax, ymax, score = box_score
            width, height = xmax - xmin, ymax - ymin
            # 写入格式：xmin, ymin, width, height, confidence
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'
                    .format(xmin, ymin, width, height, score))

    f.close()
    print("The predicted result is saved as {}".format(predict_file))
    return predict_file


def to_chw(image):
    """
    将图像从 HWC 格式转换为 CHW 格式

    Args:
        image (np.array): HWC 布局的图像数组

    Returns:
        转换为 CHW 格式的图像数组
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)  # 交换宽度和通道维度
        image = np.swapaxes(image, 1, 0)  # 交换高度和宽度维度
    return image


def lmk2out(results, is_bbox_normalized=False):
    """
    将模型输出的关键点结果转换为最终格式

    Args:
        results: 包含关键点检测结果的字典，应包括: `landmark`, `im_id`
                 如果 is_bbox_normalized=True，还需要 `im_shape`
        is_bbox_normalized: 关键点是否被归一化

    Returns:
        转换后的关键点结果列表
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        face_index = t['face_index'][0]
        prior_box = t['prior_boxes'][0]
        predict_lmk = t['landmark'][0]
        prior = np.reshape(prior_box, (-1, 4))
        predictlmk = np.reshape(predict_lmk, (-1, 10))

        k = 0
        for a in range(len(lengths)):
            num = lengths[a]
            im_id = int(im_ids[a])
            for i in range(num):
                score = bboxes[k][1]
                theindex = face_index[i][0]
                me_prior = prior[theindex, :]
                lmk_pred = predictlmk[theindex, :]
                prior_w = me_prior[2] - me_prior[0]
                prior_h = me_prior[3] - me_prior[1]
                prior_w_center = (me_prior[2] + me_prior[0]) / 2
                prior_h_center = (me_prior[3] + me_prior[1]) / 2
                lmk_decode = np.zeros((10))
                # 解码关键点坐标
                for j in [0, 2, 4, 6, 8]:  # x坐标
                    lmk_decode[j] = lmk_pred[j] * 0.1 * prior_w + prior_w_center
                for j in [1, 3, 5, 7, 9]:  # y坐标
                    lmk_decode[j] = lmk_pred[j] * 0.1 * prior_h + prior_h_center
                im_shape = t['im_shape'][0][a].tolist()
                image_h, image_w = int(im_shape[0]), int(im_shape[1])
                # 如果关键点被归一化，则转换回原始图像坐标系
                if is_bbox_normalized:
                    lmk_decode = lmk_decode * np.array([
                        image_w, image_h, image_w, image_h, image_w, image_h,
                        image_w, image_h, image_w, image_h
                    ])
                lmk_res = {
                    'image_id': im_id,
                    'landmark': lmk_decode,
                    'score': score,
                }
                xywh_res.append(lmk_res)
                k += 1
    return xywh_res


def bbox_overlaps(boxes1, boxes2):
    """
    计算两组边界框之间的交并比(IOU)

    Args:
        boxes1: (N, 4) 形状的浮点数组，表示N个边界框
        boxes2: (K, 4) 形状的浮点数组，表示K个边界框

    Returns:
        overlaps: (N, K) 形状的交并比数组，表示boxes1中每个框与boxes2中每个框的IOU
    """
    # 计算每个框的面积
    box_areas1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (
            boxes1[:, 3] - boxes1[:, 1] + 1)
    box_areas2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (
            boxes2[:, 3] - boxes2[:, 1] + 1)

    # 计算交集区域的宽度和高度
    iw = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2]) - np.maximum(
        boxes1[:, None, 0], boxes2[None, :, 0]) + 1
    ih = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3]) - np.maximum(
        boxes1[:, None, 1], boxes2[None, :, 1]) + 1

    # 确保交集宽度和高度为非负值
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    # 计算交集面积
    intersection = iw * ih

    # 计算并集面积
    union = box_areas1[:, None] + box_areas2[None, :] - intersection

    # 防止除零错误
    union = np.maximum(union, 1e-8)

    # 计算交并比(IOU)
    overlaps = intersection / union
    return overlaps


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    """
    计算图像的精确率-召回率信息

    Args:
        thresh_num: 阈值数量
        pred_info: 预测信息，包含边界框和置信度
        proposal_list: 提案列表，标记是否为正样本
        pred_recall: 预测召回率

    Returns:
        pr_info: 精确率-召回率信息数组
    """
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):
        # 计算当前阈值
        thresh = 1 - (t + 1) / thresh_num
        # 找到置信度大于等于当前阈值的预测框
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0  # 精确率
            pr_info[t, 1] = 0  # 召回率
        else:
            r_index = r_index[-1]  # 取最后一个索引
            # 找到提案列表中为正样本的索引
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)  # 正样本数量
            pr_info[t, 1] = pred_recall[r_index]  # 对应召回率
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    """
    计算整个数据集的精确率-召回率曲线信息

    Args:
        thresh_num: 阈值数量
        pr_curve: 精确率-召回率曲线数据，形状为 (thresh_num, 2)
                 pr_curve[i, 0]: 在第i个阈值下的检测总数（TP + FP）
                 pr_curve[i, 1]: 在第i个阈值下的真正例数（TP）
        count_face: 数据集中真实人脸的总数

    Returns:
        _pr_curve: 处理后的精确率-召回率曲线，形状为 (thresh_num, 2)
                  _pr_curve[i, 0]: 精确率 (Precision = TP / (TP + FP))
                  _pr_curve[i, 1]: 召回率 (Recall = TP / total_ground_truth)
    """
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        # 计算精确率: 真正例 / (真正例 + 假正例)
        if pr_curve[i, 0] > 0:
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        else:
            _pr_curve[i, 0] = 0.0

        # 计算召回率: 真正例 / 真实人脸总数
        if count_face > 0:
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        else:
            _pr_curve[i, 1] = 0.0

    return _pr_curve


def voc_ap(rec, prec):
    """
    计算PASCAL VOC格式的平均精度(Average Precision)

    Args:
        rec: 召回率数组
        prec: 精确率数组

    Returns:
        ap: 平均精度值
    """
    # 首先在数组末尾添加哨兵值
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # 计算精确率包络线
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算PR曲线下的面积，查找X轴(召回率)变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算面积：(\Delta 召回率) * 精确率
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def image_eval(pred, gt, ignore, iou_thresh):
    """
    单张图像评估函数

    Args:
        pred: 预测结果，Nx5形状，包含[xmin, ymin, xmax, ymax, score]
        gt: 真实标签，Nx4形状，包含[xmin, ymin, width, height]
        ignore: 忽略区域标记
        iou_thresh: IOU阈值

    Returns:
        pred_recall: 预测召回率数组
        proposal_list: 提案列表
    """
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    # 将gt的[xmin, ymin, width, height]格式转换为[xmin, ymin, xmax, ymax]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # 计算预测框与真实框的IOU
    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:  # 如果该gt框被标记为忽略
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:  # 如果该gt框未被匹配过
                recall_list[max_idx] = 1

        # 统计已匹配的gt框数量
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list
