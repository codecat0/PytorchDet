#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :json_results.py
@Author :CodeCat
@Date   :2025/11/17 14:48
"""
import six
import numpy as np


def get_det_res(bboxes,
                bbox_nums,
                image_id,
                label_to_cat_id_map,
                bias=0,
                im_file=None,
                save_threshold=0):
    """
    将模型输出的检测框数据转换为标准的检测结果列表格式。

    该函数通常用于将模型预测的原始边界框张量转换为COCO格式的字典列表，
    以便后续评估或可视化。它会根据类别ID映射、置信度阈值等进行过滤。

    Args:
        bboxes (list or numpy.ndarray): 模型输出的边界框数据。
            格式为 [N, 6] 或 [N, 8]，其中 N 是总检测框数量。
            每行的格式通常为 [class_id, score, x_min, y_min, x_max, y_max, ...]。
        bbox_nums (list or numpy.ndarray): 每张图像的检测框数量。
            长度为 batch_size，表示每个批次图像中检测到的框的数量。
        image_id (list or numpy.ndarray): 每张图像的ID列表。
            通常是一个二维列表或数组，例如 [[id1], [id2], ...]，需要取第一个元素。
        label_to_cat_id_map (dict): 模型标签ID到COCO类别ID的映射字典。
            例如，模型输出类别ID为0,1,2，但COCO数据集可能要求80,81,82。
        bias (float, optional): 添加到边界框宽度和高度的偏置值，默认为0。
            可用于调整边界框大小。在COCO格式中，bbox通常表示为 [x, y, width, height]。
        im_file (str, optional): 图像文件路径，如果提供则会添加到结果字典中，默认为None。
        save_threshold (float, optional): 保存检测结果的置信度阈值，默认为0。
            置信度低于此值的检测结果将被忽略。

    Returns:
        list[dict]: 检测结果列表，每个元素是一个字典，包含以下键值对：
            - 'image_id' (int): 图像ID。
            - 'category_id' (int): 映射后的COCO类别ID。
            - 'bbox' (list[float]): 边界框坐标 [x_min, y_min, width, height]。
            - 'score' (float): 检测置信度分数。
            - 'im_file' (str, optional): 如果提供了 `im_file` 参数，则包含该键。
    """
    # 初始化检测结果列表
    det_res = []
    # 用于遍历 `bboxes` 列表的索引
    k = 0

    # 遍历批次中的每张图像
    for i in range(len(bbox_nums)):
        # 获取当前图像的ID，通常 image_id[i] 是一个包含单个ID的列表/数组
        cur_image_id = int(image_id[i][0])
        # 获取当前图像的检测框数量
        det_nums = int(bbox_nums[i])

        # 遍历当前图像的所有检测框
        for j in range(det_nums):
            # 获取当前检测框的数据
            dt = bboxes[k]
            # 索引递增，指向下一个检测框
            k = k + 1

            # 将检测框数据解包为变量
            # dt 通常格式为 [class_id, score, x_min, y_min, x_max, y_max]
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()

            # 过滤条件：如果类别ID无效或置信度低于阈值，则跳过此检测框
            if int(num_id) < 0 or score < save_threshold:
                continue

            # 根据映射字典将模型输出的类别ID转换为COCO格式的类别ID
            category_id = label_to_cat_id_map[int(num_id)]

            # 计算边界框的宽度和高度，并加上偏置
            w = xmax - xmin + bias
            h = ymax - ymin + bias

            # 构建COCO格式的边界框坐标 [x_min, y_min, width, height]
            bbox = [xmin, ymin, w, h]

            # 创建当前检测结果的字典
            dt_res = {
                'image_id': cur_image_id,      # 图像ID
                'category_id': category_id,    # 映射后的类别ID
                'bbox': bbox,                  # 边界框坐标
                'score': score                 # 置信度分数
            }

            # 如果提供了图像文件路径，则将其添加到结果字典中
            if im_file is not None:
                dt_res['im_file'] = im_file

            # 将当前检测结果字典添加到结果列表中
            det_res.append(dt_res)

    # 返回所有处理后的检测结果
    return det_res


def get_det_poly_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    """
    将模型输出的多边形检测框数据转换为标准的检测结果列表格式。

    该函数专门用于处理输出为四边形（8个坐标值）的检测模型结果，
    例如某些场景文字检测或旋转目标检测模型。它会根据类别ID映射进行过滤，
    并将结果格式化为包含图像ID、类别ID、8点坐标和置信度的字典列表。

    Args:
        bboxes (list or numpy.ndarray): 模型输出的边界框数据。
            格式为 [N, 10]，其中 N 是总检测框数量。
            每行的格式为 [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4]，
            表示一个四边形的四个顶点坐标。
        bbox_nums (list or numpy.ndarray): 每张图像的检测框数量。
            长度为 batch_size，表示每个批次图像中检测到的框的数量。
        image_id (list or numpy.ndarray): 每张图像的ID列表。
            通常是一个二维列表或数组，例如 [[id1], [id2], ...]，需要取第一个元素。
        label_to_cat_id_map (dict): 模型标签ID到目标类别ID的映射字典。
            例如，模型输出类别ID为0,1,2，但最终需要映射为COCO格式的80,81,82或其他ID。
        bias (float, optional): 添加到坐标的偏置值，默认为0。
            此参数在此函数中未被使用，但为了与 `get_det_res` 保持接口一致而保留。

    Returns:
        list[dict]: 检测结果列表，每个元素是一个字典，包含以下键值对：
            - 'image_id' (int): 图像ID。
            - 'category_id' (int): 映射后的目标类别ID。
            - 'bbox' (list[float]): 多边形边界框坐标 [x1, y1, x2, y2, x3, y3, x4, y4]。
            - 'score' (float): 检测置信度分数。
    """
    # 初始化检测结果列表
    det_res = []
    # 用于遍历 `bboxes` 列表的索引
    k = 0

    # 遍历批次中的每张图像
    for i in range(len(bbox_nums)):
        # 获取当前图像的ID，通常 image_id[i] 是一个包含单个ID的列表/数组
        cur_image_id = int(image_id[i][0])
        # 获取当前图像的检测框数量
        det_nums = int(bbox_nums[i])

        # 遍历当前图像的所有检测框
        for j in range(det_nums):
            # 获取当前检测框的数据
            dt = bboxes[k]
            # 索引递增，指向下一个检测框
            k = k + 1

            # 将检测框数据解包为变量
            # dt 格式为 [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4]
            num_id, score, x1, y1, x2, y2, x3, y3, x4, y4 = dt.tolist()

            # 过滤条件：如果类别ID无效，则跳过此检测框
            # 注意：此函数没有使用 save_threshold 参数，直接过滤掉无效ID
            if int(num_id) < 0:
                continue

            # 根据映射字典将模型输出的类别ID转换为目标类别ID
            category_id = label_to_cat_id_map[int(num_id)]

            # 构建多边形边界框坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
            rbox = [x1, y1, x2, y2, x3, y3, x4, y4]

            # 创建当前检测结果的字典
            dt_res = {
                'image_id': cur_image_id,      # 图像ID
                'category_id': category_id,    # 映射后的类别ID
                'bbox': rbox,                  # 8点坐标表示的多边形边界框
                'score': score                 # 置信度分数
            }

            # 将当前检测结果字典添加到结果列表中
            det_res.append(dt_res)

    # 返回所有处理后的检测结果
    return det_res


def strip_mask(mask):
    """
    从填充的掩码张量中移除填充部分（通常为-1），提取出原始尺寸的掩码。

    该函数假设掩码张量在批次维度（axis=0）之外，其余维度的边缘部分
    （特别是高度和宽度维度）可能包含填充值（如-1），用于将不同大小的
    掩码统一到一个批次中。函数通过检查第一个样本的首行和首列来推断
    原始的高度和宽度，然后切片提取出不包含填充的部分。

    Args:
        mask (numpy.ndarray or torch.Tensor): 输入的掩码张量。
            通常形状为 [batch_size, padded_height, padded_width] 或
            [batch_size, channels, padded_height, padded_width] 等。
            其中 padded_height 和 padded_width 可能大于实际需要的高度和宽度。
            填充区域通常使用特定值（如-1）标记。

    Returns:
        numpy.ndarray or torch.Tensor: 提取出的原始尺寸掩码，
            形状为 [batch_size, im_h, im_w] 或
            [batch_size, channels, im_h, im_w] 等，
            其中 im_h 和 im_w 是推断出的原始高度和宽度。
    """
    # 获取第一个样本（batch[0]）的首行 [0, 0, :]，通常代表宽度方向的填充情况
    row = mask[0, 0, :]
    # 获取第一个样本的首列 [0, :, 0]，通常代表高度方向的填充情况
    col = mask[0, :, 0]

    # 计算原始高度：总长度减去首列中填充值（-1）的数量
    im_h = len(col) - np.count_nonzero(col == -1)
    # 计算原始宽度：总长度减去首行中填充值（-1）的数量
    im_w = len(row) - np.count_nonzero(row == -1)

    # 使用推断出的原始高度和宽度对掩码进行切片，移除填充部分
    # mask[:, :im_h, :im_w] 表示保留所有批次，高度取前 im_h 个，宽度取前 im_w 个
    return mask[:, :im_h, :im_w]


def get_seg_res(masks, bboxes, mask_nums, image_id, label_to_cat_id_map):
    """
    将模型输出的分割掩码和边界框数据转换为标准的COCO格式分割结果列表。

    该函数通常用于将实例分割模型（如Mask R-CNN）的原始输出转换为COCO格式，
    以便进行评估或可视化。它会将二值掩码编码为RLE（Run-Length Encoding）格式，
    并与对应的类别、置信度等信息一起打包。

    Args:
        masks (list or numpy.ndarray): 模型输出的掩码数据。
            形状通常为 [N, H, W] 或 [N, 1, H, W]，其中 N 是总掩码数量。
            值为0或1，表示像素是否属于实例。
        bboxes (list or numpy.ndarray): 与掩码对应的边界框和类别信息。
            格式为 [N, 6]，每行通常为 [class_id, score, x_min, y_min, x_max, y_max]。
        mask_nums (list or numpy.ndarray): 每张图像的掩码数量。
            长度为 batch_size，表示每个批次图像中生成的掩码数量。
        image_id (list or numpy.ndarray): 每张图像的ID列表。
            通常是一个二维列表或数组，例如 [[id1], [id2], ...]，需要取第一个元素。
        label_to_cat_id_map (dict): 模型标签ID到COCO类别ID的映射字典。

    Returns:
        list[dict]: 分割结果列表，每个元素是一个字典，包含以下键值对：
            - 'image_id' (int): 图像ID。
            - 'category_id' (int): 映射后的COCO类别ID。
            - 'segmentation' (dict): RLE格式的掩码编码。
            - 'score' (float): 检测置信度分数。
    """
    import pycocotools.mask as mask_util

    # 初始化分割结果列表
    seg_res = []
    # 用于遍历 `masks` 和 `bboxes` 列表的索引
    k = 0

    # 遍历批次中的每张图像
    for i in range(len(mask_nums)):
        # 获取当前图像的ID
        cur_image_id = int(image_id[i][0])
        # 获取当前图像的掩码数量
        det_nums = int(mask_nums[i])

        # 提取当前图像对应的所有掩码
        # mask_i 形状为 [det_nums, H, W] 或 [det_nums, 1, H, W]
        mask_i = masks[k:k + det_nums]
        # 移除可能的填充部分（例如由 -1 填充的区域）
        mask_i = strip_mask(mask_i)
        # 确保掩码数据类型为 uint8 (0 或 1)
        mask_i = mask_i.astype(np.uint8)

        # 遍历当前图像的所有掩码和对应的边界框信息
        for j in range(det_nums):
            # 获取当前掩码 (形状为 [H, W])
            mask = mask_i[j]
            # 获取对应的边界框信息，提取置信度和标签
            score = float(bboxes[k][1])
            label = int(bboxes[k][0])
            # 索引递增，指向下一个掩码和边界框
            k = k + 1

            # 过滤条件：如果标签为 -1 (无效标签)，则跳过此结果
            if label == -1:
                continue

            # 根据映射字典将模型输出的标签ID转换为COCO类别ID
            cat_id = label_to_cat_id_map[label]

            # 将二维掩码 [H, W] 转换为 [H, W, 1] 并使用Fortran顺序进行RLE编码
            # COCO API 期望输入为 Fortran 顺序的 uint8 数组
            rle = mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8")
            )[0]

            # 创建当前分割结果的字典
            sg_res = {
                'image_id': cur_image_id,      # 图像ID
                'category_id': cat_id,         # 映射后的类别ID
                'segmentation': rle,           # RLE格式的掩码
                'score': score                 # 置信度分数
            }

            # 将当前分割结果字典添加到结果列表中
            seg_res.append(sg_res)

    # 返回所有处理后的分割结果
    return seg_res


def get_solov2_segm_res(results, image_id, num_id_to_cat_id_map):
    """
    将 SOLOv2 模型的输出结果转换为 COCO 格式的分割结果列表。

    SOLOv2 是一种实例分割模型，它直接预测实例掩码（segm）和对应的类别标签（cate_label）、置信度分数（cate_score）。
    该函数负责将这些原始输出处理成标准的 COCO JSON 格式，以便于评估或可视化。

    Args:
        results (dict): SOLOv2 模型的输出字典，应包含以下键：
            - 'segm' (numpy.ndarray): 形状为 [N, H, W] 的二值掩码数组，N 是实例数量。
            - 'cate_label' (list or numpy.ndarray): 长度为 N 的类别标签列表。
            - 'cate_score' (list or numpy.ndarray): 长度为 N 的置信度分数列表。
        image_id (list or numpy.ndarray): 图像ID列表，通常为 [[id]] 的形式。
        num_id_to_cat_id_map (dict): 模型内部类别ID（num_id）到COCO标准类别ID（cat_id）的映射字典。

    Returns:
        list[dict] or None: COCO格式的分割结果列表。列表中的每个元素是一个字典，包含 'image_id',
                           'category_id', 'segmentation', 'score'。如果输入的掩码数量为0或无效，则返回 None。
    """
    import pycocotools.mask as mask_util

    # 初始化结果列表
    segm_res = []

    # 从模型输出中提取掩码、类别标签和置信度分数
    # 确保掩码数据类型为 uint8，这是 pycocotools.encode 所需的
    segms = results['segm'].astype(np.uint8)
    clsid_labels = results['cate_label']
    clsid_scores = results['cate_score']

    # 获取实例（掩码）的数量
    lengths = segms.shape[0]

    # 提取当前图像的ID，通常 image_id 是一个二维列表 [[id]]
    im_id = int(image_id[0][0])

    # 检查是否有有效的掩码输出
    if lengths == 0 or segms is None:
        return None

    # 遍历每个检测到的实例（除了最后一个，因为原代码循环到 lengths-1）
    # 注意：原代码是 range(lengths - 1)，这会跳过最后一个掩码。这可能是有意为之或是一个潜在的bug。
    # 如果需要处理所有掩码，应改为 range(lengths)。
    for i in range(lengths - 1):
        # 获取当前实例的类别ID和置信度分数
        clsid = int(clsid_labels[i])
        # 使用映射字典将模型内部类别ID转换为COCO标准类别ID
        catid = num_id_to_cat_id_map[clsid]
        score = float(clsid_scores[i])

        # 获取当前实例的二值掩码 [H, W]
        mask = segms[i]

        # 将二维掩码 [H, W] 转换为 [H, W, 1] 并使用Fortran顺序进行RLE编码
        # COCO API 期望输入为 Fortran 顺序的 uint8 数组
        segm = mask_util.encode(
            np.array(mask[:, :, np.newaxis], order='F')
        )[0]

        segm['counts'] = segm['counts'].decode('utf8')

        # 构建当前实例的 COCO 格式结果字典
        coco_res = {
            'image_id': im_id,          # 图像ID
            'category_id': catid,       # 映射后的COCO类别ID
            'segmentation': segm,       # RLE格式的掩码编码
            'score': score              # 检测置信度分数
        }

        # 将当前结果添加到结果列表中
        segm_res.append(coco_res)

    # 返回所有处理后的分割结果
    return segm_res


def get_keypoint_res(results, im_id):
    """
    将模型输出的关键点结果转换为COCO格式的注释列表。

    该函数通常用于将姿态估计模型（如SimpleBaseline、HRNet等）的原始输出
    转换为COCO格式，以便进行评估或可视化。它会为每个检测到的人体实例
    生成一个包含关键点坐标、置信度、边界框和面积的字典。

    Args:
        results (dict): 模型的输出字典，应包含 'keypoint' 键。
            results['keypoint'] 应为一个列表或数组，其长度等于批次大小。
            其中每个元素是一个元组 (kpts, scores)，其中：
                - kpts (numpy.ndarray): 形状为 [N, K, 3] 的数组，N是实例数，K是关键点数。
                  每个关键点包含 [x, y, confidence]。
                - scores (list or numpy.ndarray): 长度为 N 的置信度分数列表。
        im_id (numpy.ndarray): 形状为 [B, ...] 的数组，包含批次中每张图像的ID。
            通常形状为 [B, 1] 或 [B]，其中 B 是批次大小。

    Returns:
        list[dict]: COCO格式的关键点结果列表。列表中的每个元素是一个字典，
            包含以下键值对：
            - 'image_id' (int): 图像ID。
            - 'category_id' (int): 类别ID（对于人体姿态，通常硬编码为1）。
            - 'keypoints' (list[float]): 长度为 3*K 的一维列表，格式为 [x1,y1,v1,x2,y2,v2,...]。
              其中 (xi, yi) 是第i个关键点的坐标，vi是其可见性标志或置信度。
            - 'score' (float): 整个实例的检测置信度分数。
            - 'area' (float): 包含所有关键点的边界框的面积。
            - 'bbox' (list[float]): 包含所有关键点的边界框 [x_min, y_min, width, height]。
    """
    # 初始化结果列表
    anns = []

    # 获取关键点预测结果
    preds = results['keypoint']

    # 遍历批次中的每张图像
    for idx in range(im_id.shape[0]):
        # 获取当前图像的ID
        image_id = im_id[idx].item()

        # 获取当前图像的关键点和分数
        # kpts: 形状 [N, K, 3] (N=实例数, K=关键点数)
        # scores: 长度为 N 的分数列表
        kpts, scores = preds[idx]

        # 遍历当前图像中的每个检测实例
        for kpt, score in zip(kpts, scores):
            # 将关键点数组展平为一维列表 [x1,y1,v1,x2,y2,v2,...]
            kpt_flat = kpt.flatten()

            # 创建当前实例的COCO格式注释字典
            ann = {
                'image_id': image_id,              # 图像ID
                'category_id': 1,                  # 类别ID (硬编码为1，通常代表人)
                'keypoints': kpt_flat.tolist(),    # 展平的关键点坐标列表
                'score': float(score)              # 实例置信度分数
            }

            # --- 计算边界框和面积 ---
            # 提取所有关键点的 x 坐标 (每隔3个元素取第0个: 0, 3, 6, ...)
            x_coords = kpt_flat[0::3]
            # 提取所有关键点的 y 坐标 (每隔3个元素取第1个: 1, 4, 7, ...)
            y_coords = kpt_flat[1::3]

            # 计算边界框的左上角和右下角坐标
            x_min = np.min(x_coords).item()
            x_max = np.max(x_coords).item()
            y_min = np.min(y_coords).item()
            y_max = np.max(y_coords).item()

            # 计算边界框面积
            area = (x_max - x_min) * (y_max - y_min)
            # 计算边界框坐标 [x_min, y_min, width, height]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # 将计算得到的面积和边界框添加到注释字典中
            ann['area'] = area
            ann['bbox'] = bbox

            # 将当前实例的注释字典添加到结果列表中
            anns.append(ann)

    # 返回所有处理后的关键点结果
    return anns


def get_pose3d_res(results, im_id):
    """
    将模型输出的3D姿态结果转换为标准的注释列表格式。

    该函数通常用于将3D姿态估计模型（如VIBE、ProHMR等）的原始输出
    转换为一个结构化的列表，以便于后续处理、评估或可视化。
    每个结果条目包含图像ID、类别ID、3D姿态坐标和置信度分数。

    Args:
        results (dict): 模型的输出字典，应包含 'pose3d' 键。
            results['pose3d'] 应为一个列表或数组，其长度等于批次大小。
            其中每个元素是一个 numpy.ndarray，形状为 [N, K, 3] 或 [N, K*3]，
            其中 N 是实例数，K 是关键点数（例如 COCO 17点）。
            数组值表示3D坐标 (x, y, z)。
        im_id (numpy.ndarray): 形状为 [B, ...] 的数组，包含批次中每张图像的ID。
            通常形状为 [B, 1] 或 [B]，其中 B 是批次大小。

    Returns:
        list[dict]: 3D姿态结果列表。列表中的每个元素是一个字典，
            包含以下键值对：
            - 'image_id' (int): 图像ID，从 im_id 数组中获取。
            - 'category_id' (int): 类别ID（对于人体姿态，通常硬编码为1）。
            - 'pose3d' (list[float]): 3D姿态坐标列表。如果输入是 [K, 3] 形状，
              则会被展平为 [x1,y1,z1,x2,y2,z2,...] 的一维列表。
            - 'score' (float): 实例置信度分数，此处硬编码为 1.0。
    """
    # 初始化结果列表
    anns = []

    # 获取3D姿态预测结果
    preds = results['pose3d']

    # 遍历批次中的每张图像
    for idx in range(im_id.shape[0]):
        # 获取当前图像的ID
        image_id = im_id[idx].item()

        # 获取当前图像的3D姿态数据
        # pose3d 形状通常为 [N, K, 3] (N=实例数, K=关键点数, 3=xyz坐标)
        # 或者 [N, K*3] 的展平形式
        pose3d = preds[idx]

        # 创建当前实例的注释字典
        ann = {
            'image_id': image_id,              # 图像ID
            'category_id': 1,                  # 类别ID (硬编码为1，通常代表人)
            'pose3d': pose3d.tolist(),         # 将numpy数组转换为Python列表
            'score': float(1.)                 # 置信度分数 (硬编码为1.0)
        }

        # 将当前实例的注释字典添加到结果列表中
        anns.append(ann)

    # 返回所有处理后的3D姿态结果
    return anns