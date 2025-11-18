#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :coco_utils.py
@Author :CodeCat
@Date   :2025/11/17 15:39
"""
import os
import sys
import numpy as np
import itertools
from loguru import logger

from det.metrics.json_results import get_det_res, get_det_poly_res, get_seg_res, get_solov2_segm_res, get_keypoint_res, get_pose3d_res
from det.metrics.map_utils import draw_pr_curve


def get_infer_results(outs, catid, bias=0, save_threshold=0):
    """
    在推理阶段获取结果。
    输出格式是包含边界框 (bbox) 或掩码 (mask) 结果的字典。

    例如，边界框结果是一个列表，每个元素包含
    图像ID (image_id)、类别ID (category_id)、边界框坐标 (bbox) 和置信度分数 (score)。

    Args:
        outs (dict): 模型推理的原始输出字典。
                     通常包含 'im_id', 'bbox', 'mask', 'segm', 'keypoint', 'pose3d' 等键。
        catid (dict): 模型输出类别ID到目标检测/分割类别ID的映射字典。
        bias (float, optional): 添加到边界框尺寸的偏置值，默认为0。
                                用于 `get_det_res` 和 `get_det_poly_res` 函数。
        save_threshold (float, optional): 保存检测结果的置信度阈值，默认为0。
                                          低于此阈值的结果将被过滤。
                                          用于 `get_det_res` 函数。

    Returns:
        dict: 包含处理后的推理结果的字典。
              键可能包括 'bbox', 'mask', 'segm', 'keypoint', 'pose3d'。
              值是对应类型结果的列表，列表中的每个元素是一个包含结果信息的字典。
              如果输入 `outs` 中不包含某种类型的结果（如 'mask'），则输出字典中也不会包含该键。
    """
    # 检查输入的推理结果是否有效
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    # 提取图像ID和图像文件路径（如果存在）
    im_id = outs['im_id']
    im_file = outs['im_file'] if 'im_file' in outs else None

    # 初始化推理结果字典
    infer_res = {}

    # --- 处理边界框 (bbox) 结果 ---
    if 'bbox' in outs:
        # 检查 bbox 数据的格式
        # 如果 bbox 的第一个元素长度大于 6，则认为是多边形格式 (例如 [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4])
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            # 使用处理多边形边界框的函数
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            # 否则认为是标准矩形格式 (例如 [class_id, score, x_min, y_min, x_max, y_max])
            # 使用处理标准边界框的函数
            infer_res['bbox'] = get_det_res(
                outs['bbox'],
                outs['bbox_num'],
                im_id,
                catid,
                bias=bias,
                im_file=im_file,
                save_threshold=save_threshold)

    # --- 处理分割掩码 (mask) 结果  ---
    if 'mask' in outs:
        infer_res['mask'] = get_seg_res(outs['mask'], outs['bbox'],
                                        outs['bbox_num'], im_id, catid)

    # --- 处理 SOLOv2 实例分割结果 ---
    if 'segm' in outs:
        # 调用函数处理 SOLOv2 的分割结果
        infer_res['segm'] = get_solov2_segm_res(outs, im_id, catid)

    # --- 处理关键点 (keypoint) 结果 ---
    if 'keypoint' in outs:
        # 调用函数处理关键点结果
        infer_res['keypoint'] = get_keypoint_res(outs, im_id)
        # 更新 bbox_num 以匹配关键点的数量（如果 outs 中有 bbox_num）
        outs['bbox_num'] = [len(infer_res['keypoint'])]

    # --- 处理 3D 姿态 (pose3d) 结果 ---
    if 'pose3d' in outs:
        # 调用函数处理 3D 姿态结果
        infer_res['pose3d'] = get_pose3d_res(outs, im_id)
        # 更新 bbox_num 以匹配 3D 姿态的数量（如果 outs 中有 bbox_num）
        outs['bbox_num'] = [len(infer_res['pose3d'])]

    # 返回处理后的推理结果字典
    return infer_res


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    使用 COCO API 对模型的预测结果进行评估。

    该函数加载 COCO 格式的预测结果（jsonfile）和真实标注（coco_gt 或 anno_file），
    使用 COCOeval 工具计算指定类型（style）的评估指标（如 mAP）。

    Args:
        jsonfile (str): 包含模型预测结果的 JSON 文件路径。
                       例如，包含边界框预测的 'bbox.json' 或包含分割掩码预测的 'mask.json'。
        style (str): COCOeval 的评估类型。
                     可选值包括 'bbox' (边界框), 'segm' (分割), 'proposal' (候选框),
                     'keypoints' (关键点), 'keypoints_crowd' (CrowdPose关键点)。
        coco_gt (COCO object, optional): COCO API 的真实标注对象。
                                        如果提供了此对象，则无需提供 anno_file。
        anno_file (str, optional): COCO 格式的真实标注 JSON 文件路径。
                                   如果提供了此文件，则会在此函数内部创建 COCO 对象。
                                   coco_gt 和 anno_file 必须至少提供一个。
        max_dets (tuple, optional): COCO 评估中每个图像的最大检测数量。
                                    默认为 (100, 300, 1000)。
        classwise (bool, optional): 是否计算每个类别的 AP 并绘制 P-R 曲线。
                                    默认为 False。
        sigmas (numpy.ndarray, optional): 用于关键点评估的 sigma 值数组。
                                          通常与身体部位的可见性相关。
        use_area (bool, optional): 在评估时是否使用 'area' 字段过滤。
                                   如果真实标注中没有 'area' 字段（如某些数据集），请设置为 False。
                                   默认为 True。

    Returns:
        numpy.ndarray: COCOeval.summarize() 输出的统计信息数组，包含各种 AP 和 AR 指标。
                       例如 [AP, AP50, AP75, APs, APm, APl, ARmax1, ARmax10, ARmax100, ...]。
    """
    # 确保至少提供了 coco_gt 或 anno_file
    assert coco_gt is not None or anno_file is not None

    # 根据评估类型选择相应的 COCO API 库
    if style == 'keypoints_crowd':
        # 关键点评估（如 CrowdPose）可能需要 xtcocotools
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    else:
        # 一般评估使用标准的 pycocotools
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    # 如果没有提供 COCO 对象，则根据 anno_file 创建
    if coco_gt is None:
        coco_gt = COCO(anno_file)

    logger.info("Start evaluate...")

    # 加载预测结果 JSON 文件到 COCO API 的结果对象
    coco_dt = coco_gt.loadRes(jsonfile)

    # 根据评估类型 ('style') 创建 COCOeval 对象
    if style == 'proposal':
        # 'proposal' 评估通常用于评估候选框生成算法，不区分类别
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')  # 即使是 proposal，内部也可能用 bbox 相关逻辑
        coco_eval.params.useCats = 0  # 设置为 0 表示不区分类别
        coco_eval.params.maxDets = list(max_dets)  # 设置 maxDets 参数
    elif style == 'keypoints_crowd':
        # 'keypoints_crowd' 类型需要额外的 sigmas 和 use_area 参数
        coco_eval = COCOeval(coco_gt, coco_dt, style, sigmas, use_area)
    else:
        # 其他标准评估类型 ('bbox', 'segm', 'keypoints')
        coco_eval = COCOeval(coco_gt, coco_dt, style)

    # 执行评估计算
    coco_eval.evaluate()
    # 汇总评估结果
    coco_eval.accumulate()
    # 打印汇总的统计信息
    coco_eval.summarize()

    # 如果需要计算每个类别的 AP
    if classwise:
        # 计算每个类别的 AP 和 P-R 曲线
        try:
            from terminaltables import AsciiTable
        except ImportError as e:
            logger.error(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e

        # 从评估结果中获取精度 (precision) 数组
        # 形状: (IoU thresholds, recall thresholds, categories, area ranges, max det thresholds)
        precisions = coco_eval.eval['precision']
        # 获取真实标注中的类别 ID 列表
        cat_ids = coco_gt.getCatIds()
        # 确保类别数量与精度数组的类别维度匹配
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        # 遍历每个类别
        for idx, catId in enumerate(cat_ids):
            # 加载类别信息
            nm = coco_gt.loadCats(catId)[0]
            # 提取该类别的精度数据
            # [:, :, idx, 0, -1] -> (IoU, recall, this_class, all_areas, max_dets[-1])
            precision = precisions[:, :, idx, 0, -1]
            # 过滤掉无效值 (-1)
            precision = precision[precision > -1]
            # 计算 AP (平均精度)：对有效精度值求平均
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan') # 如果没有有效值，则 AP 为 nan
            # 将类别名和 AP 添加到结果列表
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))

            # 为该类别绘制 P-R 曲线
            # 提取 IoU=0.5 (或其他默认值), area=all, max_dets=1000 时的精度-召回率数据
            # [0, :, idx, 0, 2] -> (IoU[0], recall, this_class, area[0], max_dets[2])
            pr_array = precisions[0, :, idx, 0, 2]
            # COCO 标准召回率数组 [0.0, 0.01, ..., 1.0]
            recall_array = np.arange(0.0, 1.01, 0.01)
            # 绘制并保存 P-R 曲线
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=style + '_pr_curve',
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        # 格式化并打印每个类别的 AP 结果表
        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)]) # 分列
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logger.info('Per-category of {} AP: \n{}'.format(style, table.table))
        logger.info("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))

    # 刷新标准输出缓冲区
    sys.stdout.flush()
    # 返回 COCO 评估的主要统计结果 (stats)
    return coco_eval.stats


def json_eval_results(metric, json_directory, dataset):
    """
    使用已存在的 proposal.json, bbox.json 或 mask.json 文件，通过 COCO API 进行评估。

    该函数会查找指定目录下的 JSON 结果文件，并根据文件类型（proposal, bbox, segm）
    调用 COCO API 对模型的预测结果进行评估。

    Args:
        metric (str): 评估指标类型。当前函数硬编码要求为 'COCO'。
        json_directory (str): 包含评估结果 JSON 文件的目录路径。
                              例如，该目录下应包含 'proposal.json', 'bbox.json', 'mask.json'。
        dataset (Dataset object): 数据集对象，必须包含 `get_anno()` 方法
                                  以获取 COCO 格式的真实标注文件路径。
    """
    # 当前函数只支持 COCO 评估指标
    assert metric == 'COCO'

    # 获取真实标注文件的路径
    anno_file = dataset.get_anno()

    # 定义需要查找的 JSON 文件名列表
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']

    # 如果提供了 json_directory，则将文件名列表更新为完整的文件路径
    if json_directory:
        # 检查目录是否存在
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(json_directory)
        # 构建每个 JSON 文件的完整路径
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    # 定义与 JSON 文件对应的 COCO 评估类型
    # 'proposal' 用于评估候选框（RPN等）
    # 'bbox' 用于评估目标检测边界框
    # 'segm' 用于评估实例分割掩码
    coco_eval_style = ['proposal', 'bbox', 'segm']

    # 遍历每个 JSON 文件及其对应的评估类型
    for i, v_json in enumerate(json_file_list):
        # 检查 JSON 文件是否存在
        if os.path.exists(v_json):
            # 如果存在，则调用 cocoapi_eval 函数进行评估
            # 参数: 预测结果 JSON 文件路径, 评估类型, 真实标注文件路径
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            # 如果文件不存在，则记录信息日志
            logger.info("{} not exists!".format(v_json))