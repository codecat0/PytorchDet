#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :visualizer.py
@Author :CodeCat
@Date   :2025/11/17 11:39
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
from loguru import logger
import platform

from det.utils.colormap import colormap
from det.utils.compact import imagedraw_textsize_c
import pycocotools.mask as mask_util
from scipy import ndimage


def load_font(font_size=18):
    """
    根据操作系统加载合适的字体文件
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Times.ttf"
        ]
    elif system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体 (支持中文)
            "C:/Windows/Fonts/simsun.ttc",  # 宋体 (支持中文)
            "C:/Windows/Fonts/arial.ttf",  # Arial
            "C:/Windows/Fonts/times.ttf"  # Times New Roman
        ]
    elif system == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Liberation Sans
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Droid Sans Fallback (支持中文)
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"  # Noto Sans CJK (支持中文)
        ]
    else:
        # 未知系统，尝试一些常见的路径
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf"
        ]

    # 尝试加载找到的第一个存在的字体文件
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                return font
            except OSError:
                continue  # 如果加载失败，尝试下一个

    # 如果所有指定路径都失败，回退到默认字体
    print(f"Warning: Could not load any specified font for {system}. Using default font.")
    return ImageFont.load_default()


def draw_mask(image, im_id, segms, threshold, alpha=0.7):
    """
    在图像上绘制分割掩码（mask）

    Args:
        image (PIL.Image.Image): 输入图像
        im_id (int): 当前图像的 ID（用于匹配 segms 中的 image_id）
        segms (list[dict]): COCO 格式的检测结果列表，每个 dict 包含：
            - 'image_id': 图像ID
            - 'segmentation': RLE 或多边形格式的mask
            - 'score': 置信度分数
        threshold (float): 掩码显示的置信度阈值
        alpha (float): 掩码叠加透明度，范围 [0,1]

    Returns:
        PIL.Image.Image: 叠加掩码后的图像
    """
    mask_color_id = 0
    w_ratio = 0.4  # 控制颜色亮度的混合比例
    color_list = colormap(rgb=True)  # 获取预定义颜色表

    # 将PIL图像转换为可修改的float32数组
    img_array = np.array(image).astype(np.float32)

    # 遍历所有检测结果
    for dt in segms:
        # 跳过不属于当前图像的检测
        if im_id != dt['image_id']:
            continue

        segm, score = dt['segmentation'], dt['score']
        # 跳过低置信度掩码
        if score < threshold:
            continue

        # 解码RLE格式的mask为二值numpy数组 (H, W)
        mask = mask_util.decode(segm)  # 输出为uint8，0/1
        if mask.ndim == 3:  # 多个通道（罕见）
            mask = mask[:, :, 0]

        # 选择颜色（循环使用调色板）
        color_mask = color_list[mask_color_id % len(color_list)].astype(np.float32)
        mask_color_id += 1

        # 调整颜色亮度：让掩码更亮一些（混合白色）
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255

        # 获取mask中非零（前景）像素的位置
        idx = np.nonzero(mask)

        # 混合原始图像与掩码颜色：
        # 公式：out = img * (1 - alpha) + color * alpha
        img_array[idx[0], idx[1], :] *= (1.0 - alpha)
        img_array[idx[0], idx[1], :] += alpha * color_mask

    # 转回uint8并封装为PIL图像
    return Image.fromarray(img_array.astype(np.uint8))


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    在图像上绘制边界框（bounding box）

    Args:
        image (PIL.Image.Image): 输入图像
        im_id (int): 当前图像的 ID（用于匹配 bboxes 中的 image_id）
        catid2name (dict): 类别ID到类别名称的映射字典
        bboxes (list[dict]): COCO 格式的检测结果列表，每个 dict 包含：
            - 'image_id': 图像ID
            - 'category_id': 类别ID
            - 'bbox': 边界框坐标，可以是 [x,y,w,h] (4个值) 或 [x1,y1,x2,y2,x3,y3,x4,y4] (8个值)
            - 'score': 置信度分数
        threshold (float): 边界框显示的置信度阈值

    Returns:
        PIL.Image.Image: 绘制了边界框的图像
    """
    # 尝试加载字体文件，如果失败则使用默认字体
    try:
        font = load_font(font_size=18)
    except:
        logger.warning("无法加载指定字体，使用默认字体。中文标签可能显示异常。")
        font = ImageFont.load_default(size=18)

    draw = ImageDraw.Draw(image)

    # 为每个类别分配颜色
    catid2color = {}
    color_list = colormap(rgb=True)[:40]

    for dt in bboxes:
        # 跳过不属于当前图像的检测结果
        if im_id != dt['image_id']:
            continue

        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        # 跳过低置信度的检测结果
        if score < threshold:
            continue

        # 为新类别分配颜色
        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # 绘制边界框
        if len(bbox) == 4:
            # 标准矩形框 [x_min, y_min, width, height]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            # 绘制矩形框的四条边
            draw.line(
                [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)],
                width=2,
                fill=color
            )
        elif len(bbox) == 8:
            # 任意四边形框 [x1, y1, x2, y2, x3, y3, x4, y4]
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            # 绘制四边形的四条边
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color
            )
            # 计算包围盒的左上角坐标，用于放置标签
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
        else:
            logger.error(f'边界框的形状必须是 [4] 或 [8]，当前是 [{len(bbox)}]！')
            continue  # 跳过错误的边界框

        # 绘制标签文本
        # 格式：类别名 置信度(保留两位小数)
        text = f"{catid2name[catid]} {score:.2f}"
        # 获取文本框尺寸
        text_width, text_height = imagedraw_textsize_c(draw, text, font=font)
        # 在左上角绘制一个填充矩形作为标签背景
        draw.rectangle(
            [(x_min + 1, y_min - text_height), (x_min + text_width + 1, y_min)],
            fill=color
        )
        # 在矩形框上方绘制白色文字
        draw.text((x_min + 1, y_min - text_height), text, fill=(255, 255, 255), font=font)

    return image


def save_result(save_path, results, catid2name, threshold):
    """
    保存检测结果为txt文件

    Args:
        save_path (str): 保存文件的路径
        results (dict): 检测结果字典，应包含以下字段之一：
            - "bbox_res": 边界框检测结果列表，每个元素为包含 'category_id', 'bbox', 'score' 的字典
            - "keypoint_res": 关键点检测结果列表，每个元素为包含 'keypoints', 'score' 的字典
            - "im_id": 图像ID
        catid2name (dict): 类别ID到类别名称的映射字典
        threshold (float): 保存结果的置信度阈值，低于此值的结果将被过滤掉
    """
    # 提取图像ID，转换为整数类型
    img_id = int(results["im_id"])

    # 以写入模式打开文件
    with open(save_path, 'w') as f:
        # 检查结果中是否包含边界框检测结果
        if "bbox_res" in results:
            # 遍历所有边界框检测结果
            for dt in results["bbox_res"]:
                # 提取类别ID、边界框坐标和置信度分数
                catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']

                # 如果置信度分数低于设定的阈值，则跳过该结果
                if score < threshold:
                    continue

                # 格式化边界框预测结果为一行文本
                # 格式：类别名 置信度 坐标1 坐标2 ... (坐标可以是 [x1,y1,w,h] 或 [x1,y1,x2,y2,x3,y3,x4,y4])
                bbox_pred = '{} {} '.format(catid2name[catid], score) + ' '.join(
                    [str(e) for e in bbox]
                )
                # 将格式化后的字符串写入文件并换行
                f.write(bbox_pred + '\n')

        # 检查结果中是否包含关键点检测结果
        elif "keypoint_res" in results:
            # 遍历所有关键点检测结果
            for dt in results["keypoint_res"]:
                # 提取关键点坐标和置信度分数
                kpts = dt['keypoints']
                scores = dt['score']

                # 构建关键点预测结果列表：[图像ID, 分数, 关键点坐标]
                keypoint_pred = [img_id, scores, kpts]

                # 将列表内容打印（写入）到文件中
                print(keypoint_pred, file=f)

        # 如果结果中既没有边界框也没有关键点，则提示无有效结果
        else:
            print("No valid results found, skip txt save")


def draw_segm(image,
              im_id,
              catid2name,
              segms,
              threshold,
              alpha=0.7,
              draw_box=True):
    """
    在图像上绘制分割掩码（segmentation mask）

    Args:
        image (PIL.Image.Image): 输入图像
        im_id (int): 当前图像的 ID（用于匹配 segms 中的 image_id）
        catid2name (dict): 类别ID到类别名称的映射字典
        segms (list[dict]): COCO 格式的检测结果列表，每个 dict 包含：
            - 'image_id': 图像ID
            - 'category_id': 类别ID
            - 'segmentation': RLE 或多边形格式的mask
            - 'score': 置信度分数
        threshold (float): 掩码显示的置信度阈值
        alpha (float): 掩码叠加透明度，范围 [0,1]
        draw_box (bool): 是否绘制边界框，如果为 False 则绘制类别标签

    Returns:
        PIL.Image.Image: 绘制了分割掩码的图像
    """
    mask_color_id = 0
    w_ratio = 0.4  # 控制颜色亮度的混合比例
    color_list = colormap(rgb=True)  # 获取预定义颜色表

    # 将PIL图像转换为可修改的float32数组
    img_array = np.array(image).astype(np.float32)

    # 遍历所有检测结果
    for dt in segms:
        # 跳过不属于当前图像的检测
        if im_id != dt['image_id']:
            continue

        segm, score, catid = dt['segmentation'], dt['score'], dt['category_id']
        # 跳过低置信度掩码
        if score < threshold:
            continue

        # 解码RLE格式的mask为二值numpy数组 (H, W)
        mask = mask_util.decode(segm)  # 输出为uint8，0/1
        if mask.ndim == 3:  # 多个通道（罕见）
            mask = mask[:, :, 0]

        # 选择颜色（循环使用调色板）
        color_mask = color_list[mask_color_id % len(color_list)].astype(np.float32)
        mask_color_id += 1

        # 调整颜色亮度：让掩码更亮一些（混合白色）
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255

        # 获取mask中非零（前景）像素的位置
        idx = np.nonzero(mask)

        # 混合原始图像与掩码颜色：
        # 公式：out = img * (1 - alpha) + color * alpha
        img_array[idx[0], idx[1], :] *= (1.0 - alpha)
        img_array[idx[0], idx[1], :] += alpha * color_mask

        if not draw_box:
            # 不绘制边界框，绘制类别标签
            # 计算掩码的质心坐标
            center_y, center_x = ndimage.measurements.center_of_mass(mask)
            # 格式化标签文本
            label_text = "{}".format(catid2name[catid])
            # 计算标签显示位置，稍微偏移避免在边界上
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            # 在图像上绘制标签文本（注意：cv2使用BGR格式）
            # 将位置和颜色转换为整数
            vis_pos = tuple(map(int, vis_pos))
            cv2.putText(img_array, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # 白色文字
        else:
            # 绘制边界框
            mask_for_bbox = mask_util.decode(segm) * 255

            # 计算边界框坐标
            # sum_x: 沿y轴（行）求和，得到每列的像素值总和
            sum_x = np.sum(mask_for_bbox, axis=0)
            # 找出像素值大于0.5的列索引（即包含前景像素的列）
            x = np.where(sum_x > 0.5)[0]
            # sum_y: 沿x轴（列）求和，得到每行的像素值总和
            sum_y = np.sum(mask_for_bbox, axis=1)
            # 找出像素值大于0.5的行索引（即包含前景像素的行）
            y = np.where(sum_y > 0.5)[0]

            # 获取边界框的左上角和右下角坐标
            if len(x) > 0 and len(y) > 0:
                x0, x1, y0, y1 = int(x[0]), int(x[-1]), int(y[0]), int(y[-1])

                # 绘制边界框矩形
                color_bgr = tuple(map(int, color_mask.astype('int32').tolist()[::-1]))  # 转换为BGR并转为tuple
                cv2.rectangle(img_array, (x0, y0), (x1, y1), color_bgr, 1)

                # 绘制标签文本和背景
                bbox_text = '%s %.2f' % (catid2name[catid], score)
                # 获取文本尺寸
                t_size = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, thickness=1)[0]
                # 绘制标签背景矩形
                bg_color_bgr = color_bgr
                cv2.rectangle(img_array, (x0, y0), (x0 + t_size[0], y0 - t_size[1] - 3), bg_color_bgr, -1)  # -1表示填充
                # 在背景矩形上绘制黑色文字
                text_color_bgr = (0, 0, 0)  # 黑色文字
                cv2.putText(
                    img_array,
                    bbox_text, (x0, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, text_color_bgr,
                    1,
                    lineType=cv2.LINE_AA)  # 抗锯齿

    # 转回uint8并封装为PIL图像
    return Image.fromarray(img_array.astype(np.uint8))


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def draw_pose(image,
              results,
              visual_thread=0.6,
              save_name='pose.jpg',
              save_dir='output',
              returnimg=False,
              ids=None):
    """
    在图像上绘制姿态关键点和连接线

    Args:
        image (PIL.Image.Image): 输入图像
        results (dict or list): 检测结果，包含关键点信息。
            如果是dict，应包含 'keypoints' 键；如果是list，则直接包含关键点数据
        visual_thread (float): 关键点可见性的置信度阈值，低于此值的点将被忽略
        save_name (str): 保存图像的文件名
        save_dir (str): 保存图像的目录
        returnimg (bool): 是否返回绘制后的图像
        ids (list or None): 每个姿态的ID列表，用于区分不同实例的颜色

    Returns:
        PIL.Image.Image or None: 绘制后的图像（如果 returnimg=True）
    """
    # 处理 results 输入格式
    # 如果 results 是一个字典（包含多个字段），提取 'keypoints'
    if isinstance(results, dict) and 'keypoints' in results:
        keypoints_data = results['keypoints']
    # 如果 results 是一个列表（直接包含关键点结果）
    elif isinstance(results, list):
        keypoints_data = results
    else:
        logger.error(
            f"Invalid results format. Expected dict with 'keypoints' or list of pose results, got {type(results)}")
        return image if returnimg else None

    # 从关键点数据中提取坐标
    # 假设每个关键点包含 [x, y, score]，所以总长度是 kpt_nums * 3
    if len(keypoints_data) > 0:
        # 如果是单个骨架数据（列表形式），需要包装成二维数组
        if isinstance(keypoints_data[0], (int, float)):
            # 单个骨架，格式为 [x1,y1,s1, x2,y2,s2, ...]
            total_vals = len(keypoints_data)
            if total_vals % 3 != 0:
                logger.error(f"Keypoints data length {total_vals} is not divisible by 3. Expected [x,y,score] format.")
                return image if returnimg else None
            kpt_nums = total_vals // 3
            skeletons = np.array(keypoints_data).reshape(1, kpt_nums, 3)
        else:
            # 多个骨架，格式为 [[x1,y1,s1, x2,y2,s2, ...], ...] 或 [[x1,y1,s1], [x2,y2,s2], ...]
            if len(keypoints_data[0]) == 3:  # [[x,y,score], [x,y,score], ...] 形式
                skeletons = np.array(keypoints_data).reshape(-1, 1, 3)  # 每个骨架只有一个关键点
                kpt_nums = 1
            elif len(keypoints_data[0]) % 3 == 0:  # [[x1,y1,s1, x2,y2,s2, ...], ...] 形式
                total_vals_per_item = len(keypoints_data[0])
                kpt_nums = total_vals_per_item // 3
                skeletons = np.array(keypoints_data).reshape(-1, kpt_nums, 3)
            else:
                logger.error(f"Invalid keypoints data format: {keypoints_data[0]}")
                return image if returnimg else None
    else:
        # 没有检测到任何姿态，直接返回原图
        return image if returnimg else None

    # 根据关键点数量选择连接边
    if kpt_nums == 17:  # COCO 17点格式
        EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                 (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]
    elif kpt_nums == 16:  # MPII 16点格式
        EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                 (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                 (8, 13)]
    else:  # 自定义关键点数量，不绘制连接线
        EDGES = []
        logger.info(f"Keypoint number {kpt_nums} not recognized (17 for COCO, 16 for MPII). Drawing points only.")

    NUM_EDGES = len(EDGES)

    # 定义颜色列表，用于不同关键点或实例
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # 将PIL图像转换为numpy数组进行绘制
    img = np.array(image).astype(np.float32)

    # 如果提供了颜色集，使用它来选择颜色
    color_set = results.get('colors') if isinstance(results, dict) else None

    # 如果结果中包含边界框，绘制边界框
    if isinstance(results, dict) and 'bbox' in results and ids is None:
        bboxs = results['bbox']
        for j, rect in enumerate(bboxs):
            if j >= len(skeletons):  # 防止bbox和skeleton数量不匹配
                break
            xmin, ymin, xmax, ymax = map(int, rect)  # 确保坐标为整数
            color_idx = 0 if color_set is None or j >= len(color_set) else color_set[j]
            color = colors[0] if color_set is None else colors[color_idx % len(colors)]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

    # 创建画布副本用于绘制关键点
    canvas = img.copy()

    # 绘制关键点
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            # 检查关键点的置信度
            if skeletons[j][i, 2] < visual_thread:
                continue

            # 选择颜色
            if ids is None:
                if color_set is None or j >= len(color_set):
                    color = colors[i % len(colors)]
                else:
                    color = colors[color_set[j] % len(colors)]
            else:
                if j < len(ids):
                    color = get_color(ids[j])
                else:
                    color = colors[i % len(colors)]  # fallback

            # 绘制关键点（实心圆）
            center = tuple(skeletons[j][i, 0:2].astype(np.int32).tolist())
            cv2.circle(
                canvas,
                center,
                2,  # 半径
                color,
                thickness=-1  # 填充
            )

    # 将关键点图层与原图混合
    # 绘制连接线
    stickwidth = 2

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            # 检查两个关键点的置信度
            if skeletons[j][edge[0], 2] < visual_thread or skeletons[j][edge[1], 2] < visual_thread:
                continue

            # 创建临时画布用于绘制单条线
            cur_canvas = canvas.copy()
            # 获取两个关键点的坐标 (y, x) -> (row, col) -> (x, y) for cv2
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]  # Y坐标
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]  # X坐标
            mX = np.mean(X)  # 中点Y
            mY = np.mean(Y)  # 中点X
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5  # 线段长度
            # 计算角度，atan2返回弧度，需要转换为度
            # 注意：atan2(dy, dx)，这里 (X0-X1) 对应 dy, (Y0-Y1) 对应 dx
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # cv2.ellipse2Poly 参数: (中心点), (半轴长), 角度, 起始角, 结束角, 步长
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),  # cv2坐标系 (x, y)
                                       (int(length / 2), stickwidth),  # (半长轴, 半短轴)
                                       int(angle), 0, 360, 1)  # 角度, 起始, 结束, 步长
            # 选择颜色
            if ids is None:
                if color_set is None or j >= len(color_set):
                    color = colors[i % len(colors)]
                else:
                    color_idx = color_set[j]
                    color = colors[color_idx % len(colors)]
            else:
                if j < len(ids):
                    color = get_color(ids[j])
                else:
                    color = colors[i % len(colors)]  # fallback

            # 填充椭圆（作为粗线段）
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            # 将绘制了线段的临时画布与主画布混合
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    # 将绘制完成的numpy数组转换回PIL图像
    image = Image.fromarray(canvas.astype(np.uint8))

    # 如果需要保存或返回图像
    if returnimg:
        return image
    else:
        import os
        save_path = os.path.join(save_dir, save_name)
        image.save(save_path)
        print(f"Pose image saved to {save_path}")
        return None


def draw_pose3d(image,
                pose3d,
                pose2d=None,
                visual_thread=0.6,
                save_name='pose3d.jpg',
                returnimg=True):
    """
    使用 matplotlib 绘制 3D 姿态，并可选择性地叠加 2D 姿态和原图。

    Args:
        image (PIL.Image.Image or numpy.ndarray): 输入图像。如果为 None，则只绘制姿态。
        pose3d (numpy.ndarray): 3D 关键点坐标，形状为 (N, 3) 或 (3, N)，其中 N 是关键点数量。
        pose2d (numpy.ndarray, optional): 2D 关键点坐标，形状为 (N, 3)，其中最后一维包含 [x, y, confidence]。
        visual_thread (float): 2D 关键点可见性的置信度阈值。
        save_name (str): 如果 returnimg 为 False，保存图像的路径。
        returnimg (bool): 是否返回绘制后的 PIL 图像对象。

    Returns:
        PIL.Image.Image or None: 如果 returnimg 为 True，返回绘制后的图像；否则返回 None。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        # 切换后端以避免 GUI 问题
        plt.switch_backend('agg')
    except ImportError as e:
        logger.error('Matplotlib not found, please install matplotlib. '
                     'For example: `pip install matplotlib`.')
        raise e

    # 确保 pose3d 是 (N, 3) 的形状
    if pose3d.shape[0] == 3 and pose3d.shape[1] != 3:
        pose3d = pose3d.T
    if pose3d.shape[1] != 3:
        logger.error(f"Expected pose3d shape (N, 3) or (3, N), got {pose3d.shape}")
        return None if not returnimg else Image.new("RGB", (640, 480), "black")

    # 根据关键点数量选择连接关系
    if pose3d.shape[0] == 24:
        joints_connectivity_dict = [
            [0, 1, 0], [1, 2, 0], [5, 4, 1], [4, 3, 1], [2, 3, 0], [2, 14, 1],
            [3, 14, 1], [14, 16, 1], [15, 16, 1], [15, 12, 1], [6, 7, 0],
            [7, 8, 0], [11, 10, 1], [10, 9, 1], [8, 12, 0], [9, 12, 1],
            [12, 19, 1], [19, 18, 1], [19, 20, 0], [19, 21, 1], [22, 20, 0],
            [23, 21, 1]
        ]
    elif pose3d.shape[0] == 14:
        joints_connectivity_dict = [
            [0, 1, 0], [1, 2, 0], [5, 4, 1], [4, 3, 1], [2, 3, 0], [2, 12, 0],
            [3, 12, 1], [6, 7, 0], [7, 8, 0], [11, 10, 1], [10, 9, 1],
            [8, 12, 0], [9, 12, 1], [12, 13, 1]
        ]
    else:
        logger.error(f"Undefined joints number: {pose3d.shape[0]}. Cannot visualize due to unknown joint connectivity.")
        return None if not returnimg else Image.new("RGB", (640, 480), "black")

    def draw3Dpose(pose3d,
                   ax,
                   lcolor="#3498db",
                   rcolor="#e74c3c",
                   add_labels=False):
        """
        在 3D 轴上绘制姿态连接线。
        """
        for i in joints_connectivity_dict:
            joint_idx_1, joint_idx_2, is_left = i[0], i[1], i[2]
            # 获取两个关键点的 x, y, z 坐标
            x_coords = np.array([pose3d[joint_idx_1, 0], pose3d[joint_idx_2, 0]])
            y_coords = np.array([pose3d[joint_idx_1, 1], pose3d[joint_idx_2, 1]])
            z_coords = np.array([pose3d[joint_idx_1, 2], pose3d[joint_idx_2, 2]])
            # 绘制线段，注意 matplotlib 的坐标系 (x, y, z)
            # 原代码中是 (-x, -z, -y)，这里保持一致或根据需要调整
            ax.plot(-x_coords, -z_coords, -y_coords, lw=2, c=lcolor if is_left else rcolor)

        # 设置坐标轴范围
        RADIUS = 1000
        center_idx = 2 if pose3d.shape[0] == 14 else 14
        x_center, y_center, z_center = pose3d[center_idx, 0], pose3d[center_idx, 1], pose3d[center_idx, 2]
        ax.set_xlim3d([-RADIUS + x_center, RADIUS + x_center])
        ax.set_ylim3d([-RADIUS + z_center, RADIUS + z_center])
        ax.set_zlim3d([-RADIUS + y_center, RADIUS + y_center])

        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")

    def draw2Dpose(pose2d,
                   ax,
                   lcolor="#3498db",
                   rcolor="#e74c3c",
                   add_labels=False):
        """
        在 2D 轴上绘制姿态连接线。
        """
        if pose2d is None:
            return
        for i in joints_connectivity_dict:
            joint_idx_1, joint_idx_2, is_left = i[0], i[1], i[2]
            # 检查两个关键点的置信度是否满足阈值
            if pose2d[joint_idx_1, 2] > visual_thread and pose2d[joint_idx_2, 2] > visual_thread:
                # 获取两个关键点的 x, y 坐标
                x_coords = np.array([pose2d[joint_idx_1, 0], pose2d[joint_idx_2, 0]])
                y_coords = np.array([pose2d[joint_idx_1, 1], pose2d[joint_idx_2, 1]])
                # 在 2D 轴上绘制线段 (x, y)
                ax.plot(x_coords, y_coords, lw=2, c=lcolor if is_left else rcolor)

    def draw_img_pose(pose3d,
                      pose2d=None,
                      frame=None,
                      figsize=(12, 12),
                      savepath=None):
        """
        创建包含多个视图的图像。
        """
        fig = plt.figure(figsize=figsize, dpi=80)
        fig.tight_layout()

        # 子图 1: 原图 + 2D 姿态
        ax1 = fig.add_subplot(221)
        if frame is not None:
            # 如果 frame 是 PIL.Image，需要先转换为 numpy array
            if isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame
            ax1.imshow(frame_np, interpolation='nearest')
        if pose2d is not None:
            draw2Dpose(pose2d, ax1)
        ax1.axis('off')

        # 子图 2: 3D 姿态 (45, 45 视角)
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.view_init(elev=45, azim=45)
        draw3Dpose(pose3d, ax2)

        # 子图 3: 3D 姿态 (0, 0 视角 - 正视图)
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.view_init(elev=0, azim=0)
        draw3Dpose(pose3d, ax3)

        # 子图 4: 3D 姿态 (0, 90 视角 - 侧视图)
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.view_init(elev=0, azim=90)
        draw3Dpose(pose3d, ax4)

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig

    def fig2data(fig):
        """
        将 matplotlib figure 转换为 PIL Image。

        Args:
            fig (matplotlib.figure.Figure): 要转换的 figure 对象。

        Returns:
            PIL.Image.Image: 转换后的 RGB 图像。
        """
        # 绘制 figure 的 canvas
        fig.canvas.draw()

        # 从 figure 获取 RGBA 缓冲区
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf.shape = (h, w, 3)

        # 将 numpy array 转换为 PIL Image
        image = Image.fromarray(buf, mode='RGB')
        return image

    # 执行绘图
    fig = draw_img_pose(pose3d, pose2d, frame=image)
    if fig is None:
        return None if not returnimg else Image.new("RGB", (640, 480), "black")

    data = fig2data(fig)

    if not returnimg:
        data.save(save_name)
        plt.close(fig)
        return None
    else:
        plt.close(fig)
        return data


def visualize_results(image,
                      bbox_res,
                      mask_res,
                      segm_res,
                      keypoint_res,
                      pose3d_res,
                      im_id,
                      catid2name,
                      threshold=0.5):
    """
    Visualize bbox and mask results
    """
    if bbox_res is not None:
        image = draw_bbox(image, im_id, catid2name, bbox_res, threshold)
    if mask_res is not None:
        image = draw_mask(image, im_id, mask_res, threshold)
    if segm_res is not None:
        image = draw_segm(image, im_id, catid2name, segm_res, threshold)
    if keypoint_res is not None:
        image = draw_pose(image, keypoint_res, threshold)
    if pose3d_res is not None:
        pose3d = np.array(pose3d_res[0]['pose3d']) * 1000
        image = draw_pose3d(image, pose3d, visual_thread=threshold)
    return image
