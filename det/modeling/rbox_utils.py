#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :rbox_utils.py
@Author :CodeCat
@Date   :2025/11/17 15:42
"""
import math
import torch
import numpy as np
import cv2


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


def poly2rbox_le135_np(poly):
    """
    将8点四边形坐标转换为旋转边界框 (rbox) 格式 [cx, cy, w, h, angle]。

    该函数假设输入的四边形是近似矩形的，并计算其最小外接矩形（旋转）。它通过计算相邻边的长度
    来确定宽和高（长边为宽，短边为高），并通过向量的反正切计算旋转角度。
    角度会通过 `norm_angle` 函数进行归一化，以符合特定范围要求（根据 `norm_angle` 的实现）。

    Args:
        poly (list or numpy.ndarray): 长度为8的列表或数组，表示四边形的四个顶点坐标。
            格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。
            顶点顺序通常应为连续的（例如顺时针或逆时针）。

    Returns:
        list[float]: 长度为5的列表，表示旋转边界框 [cx, cy, w, h, angle]。
            - cx (float): 旋转框中心的 x 坐标。
            - cy (float): 旋转框中心的 y 坐标。
            - w (float): 旋转框的宽度（较长的边）。
            - h (float): 旋转框的高度（较短的边）。
            - angle (float): 旋转框相对于水平轴的逆时针旋转角度（以弧度为单位），
              并经过 `norm_angle` 归一化。
    """
    # 确保输入是 numpy float32 数组，并且只取前8个元素
    poly = np.array(poly[:8], dtype=np.float32)

    # 将8个坐标值解析为4个顶点
    pt1 = (poly[0], poly[1])  # 第一个顶点 (x1, y1)
    pt2 = (poly[2], poly[3])  # 第二个顶点 (x2, y2)
    pt3 = (poly[4], poly[5])  # 第三个顶点 (x3, y3)
    pt4 = (poly[6], poly[7])  # 第四个顶点 (x4, y4)

    # 计算相邻边的长度
    # edge1: pt1 -> pt2
    edge1 = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    # edge2: pt2 -> pt3
    edge2 = np.sqrt((pt2[0] - pt3[0]) ** 2 + (pt2[1] - pt3[1]) ** 2)

    # 确定宽和高
    # 根据代码逻辑，较长的边为 width，较短的边为 height
    # 注意：这种分配方式假设了特定的顶点顺序（例如，width 对应 pt1->pt2 或 pt2->pt3）
    width = max(edge1, edge2)
    height = min(edge1, edge2)

    # 初始化角度
    rbox_angle = 0

    # 根据哪条边更长来确定角度
    # 如果 edge1 更长，则角度由 pt1 指向 pt2 的向量决定
    if edge1 > edge2:
        rbox_angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    elif edge2 >= edge1:
        rbox_angle = np.arctan2(pt4[1] - pt1[1], pt4[0] - pt1[0])

    # 对计算出的角度进行归一化
    rbox_angle = norm_angle(rbox_angle)

    # 计算旋转框的中心点坐标 (对角线顶点的中点)
    x_ctr = float(pt1[0] + pt3[0]) / 2.0
    y_ctr = float(pt1[1] + pt3[1]) / 2.0

    # 返回中心点、宽、高和归一化后的角度
    return [x_ctr, y_ctr, width, height, rbox_angle]


def poly2rbox_oc_np(poly):
    """
    使用 OpenCV 将 8 点四边形坐标转换为旋转边界框 (rbox) 格式 [cx, cy, w, h, angle]。

    该函数利用 OpenCV 的 `cv2.minAreaRect` 函数来计算包含给定四个点的最小面积外接矩形。
    然后根据 OpenCV 的 Rotated BBox 定义（特别是 4.5.1 版本之后的变化）和特定的范围要求，
    对计算出的中心点、宽高和角度进行调整，最终输出弧度制的角度。

    Args:
        poly (list or numpy.ndarray): 长度为 8 的列表或数组，表示四边形的四个顶点坐标。
            格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。
            顶点顺序对 `minAreaRect` 的结果影响不大，因为它寻找的是最小面积矩形。

    Returns:
        list[float]: 长度为 5 的列表，表示旋转边界框 [cx, cy, w, h, angle]。
            - cx (float): 旋转框中心的 x 坐标。
            - cy (float): 旋转框中心的 y 坐标。
            - w (float): 旋转框的宽度。
            - h (float): 旋转框的高度。
            - angle (float): 旋转框相对于水平轴的逆时针旋转角度（以弧度为单位），
              范围在 (0, π/2] 或 [0, π/2) 之间（取决于 OpenCV 版本和点的顺序）。
    """
    # 将输入坐标转换为 numpy float32 数组，并重塑为 N x 2 的点集格式
    points = np.array(poly, dtype=np.float32).reshape((-1, 2))

    # 使用 OpenCV 计算包含这四个点的最小面积外接矩形
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)

    if angle < 0:
        angle += 90
        w, h = h, w

    # 确保角度在 [0, 90) 范围内
    # 处理 -0.0 的情况（虽然在 float 比较中通常不推荐直接比较 -0.0）
    if angle == -0.0:
        angle = 0.0
    if angle == 90.0:
        angle = 0.0
        w, h = h, w

    # 将角度从度转换为弧度
    angle = angle / 180 * np.pi

    # 返回中心点、宽、高和归一化后的弧度角度
    return [cx, cy, w, h, angle]


def poly2rbox_np(polys, rbox_type='oc'):
    """
    将一批8点四边形坐标转换为旋转边界框 (rbox) 格式 [cx, cy, w, h, angle]。

    该函数支持两种不同的角度定义方式（通过 rbox_type 参数选择）：
    1. 'oc': 使用 OpenCV 的 `minAreaRect` 方法，角度范围经过调整，通常在 [0, π/2) 弧度之间。
    2. 'le135': 使用自定义方法，角度范围经过 `norm_angle` 归一化，根据 `norm_angle` 函数的实现而定。

    Args:
        polys (list of lists or numpy.ndarray): 输入的四边形坐标列表。
            形状应为 [N, 8]，其中 N 是四边形的数量。
            每个四边形的格式为 [x0, y0, x1, y1, x2, y2, x3, y3]。
        rbox_type (str, optional): 指定旋转框角度的定义方式。
            'oc' (默认): 使用 OpenCV 方法 (poly2rbox_oc_np)。
            'le135': 使用自定义方法 (poly2rbox_le135_np)。
            如果不是这两个值之一，将抛出 AssertionError。

    Returns:
        numpy.ndarray: 形状为 [N, 5] 的数组，表示转换后的旋转边界框。
            每行的格式为 [cx, cy, w, h, angle]。
            - cx (float): 旋转框中心的 x 坐标。
            - cy (float): 旋转框中心的 y 坐标。
            - w (float): 旋转框的宽度。
            - h (float): 旋转框的高度。
            - angle (float): 旋转框相对于水平轴的逆时针旋转角度（以弧度为单位），
              根据 `rbox_type` 选择的函数进行计算和归一化。
    """
    # 检查 rbox_type 是否为支持的类型
    assert rbox_type in ['oc', 'le135'], 'only oc or le135 is supported now'

    # 根据 rbox_type 选择相应的转换函数
    poly2rbox_fn = poly2rbox_oc_np if rbox_type == 'oc' else poly2rbox_le135_np

    # 初始化结果列表
    rboxes = []

    # 遍历输入的每个四边形坐标
    for poly in polys:
        # 调用选定的转换函数
        x_ctr, y_ctr, width, height, angle = poly2rbox_fn(poly)
        # 将结果打包为 numpy float32 数组
        rbox = np.array([x_ctr, y_ctr, width, height, angle], dtype=np.float32)
        # 添加到结果列表
        rboxes.append(rbox)

    # 将结果列表转换为 numpy 数组并返回
    # 形状从 [N, (5,)] 变为 [N, 5]
    return np.array(rboxes)


def cal_line_length(point1, point2):
    """
    计算两点之间的欧几里得距离（直线长度）。

    该函数使用标准的欧几里得距离公式计算二维平面上两个点之间的直线距离。
    公式为：sqrt((x2 - x1)^2 + (y2 - y1)^2)。

    Args:
        point1 (tuple or list): 第一个点的坐标，格式为 (x1, y1) 或 [x1, y1]。
        point2 (tuple or list): 第二个点的坐标，格式为 (x2, y2) 或 [x2, y2]。

    Returns:
        float: 两点之间的直线距离。
    """
    # 计算 x 坐标差值的平方
    dx_squared = math.pow(point1[0] - point2[0], 2)
    # 计算 y 坐标差值的平方
    dy_squared = math.pow(point1[1] - point2[1], 2)
    # 计算平方和
    sum_of_squares = dx_squared + dy_squared
    # 计算平方根，得到欧几里得距离
    distance = math.sqrt(sum_of_squares)
    return distance


def get_best_begin_point_single(coordinate):
    """
    寻找四边形坐标点的最佳起始点，使其顶点顺序与标准轴对齐矩形的顶点顺序（左上、右上、右下、左下）最相似。

    该函数通过计算输入四边形的四种可能顶点循环排列与标准轴对齐矩形顶点的距离和，
    选择距离和最小的排列作为“最佳”排列，并返回该排列的坐标。

    Args:
        coordinate (list or numpy.ndarray): 长度为8的列表或数组，表示四边形的四个顶点坐标。
            格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。
            注意：此函数假设输入的四个顶点是连续的（尽管顺序可能不是期望的顺序）。

    Returns:
        numpy.ndarray: 形状为 (8,) 的 numpy float64 数组，表示经过重新排列后的顶点坐标
            [x1', y1', x2', y2', x3', y3', x4', y4']。这个顺序是使得其与标准轴对齐矩形顶点
            顺序距离和最小的那个循环排列。
    """
    # 解包输入坐标
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate

    # 计算包围盒的极值，用于定义“标准”轴对齐矩形的顶点
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)

    # 生成输入四边形的四种可能的循环排列
    # 每种排列代表一个不同的起始点和循环顺序
    combinate = [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 原始顺序
        [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],  # 从最后一个点开始
        [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],  # 从第三个点开始
        [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]  # 从第二个点开始
    ]

    # 定义标准轴对齐矩形的四个顶点（左上、右上、右下、左下）
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    # 初始化最小距离和及其对应的排列索引
    min_distance_sum = float('inf')  # 使用 float('inf') 作为初始最大值更清晰
    best_index = 0

    # 遍历四种排列
    for i in range(4):
        # 计算当前排列的每个顶点到标准矩形对应顶点的距离之和
        distance_sum = (
                cal_line_length(combinate[i][0], dst_coordinate[0]) +  # 当前排列第1点 -> 标准第1点 (左上)
                cal_line_length(combinate[i][1], dst_coordinate[1]) +  # 当前排列第2点 -> 标准第2点 (右上)
                cal_line_length(combinate[i][2], dst_coordinate[2]) +  # 当前排列第3点 -> 标准第3点 (右下)
                cal_line_length(combinate[i][3], dst_coordinate[3])  # 当前排列第4点 -> 标准第4点 (左下)
        )

        # 如果当前距离和更小，则更新最小距离和及索引
        if distance_sum < min_distance_sum:
            min_distance_sum = distance_sum
            best_index = i

    # 根据找到的最佳索引，选择对应的排列
    best_combination = combinate[best_index]

    return np.array(best_combination).reshape(8)


def rbox2poly_np(rboxes):
    """
    将一批旋转边界框 (rbox) 格式 [cx, cy, w, h, angle] 转换为四边形坐标格式 [x0,y0,x1,y1,x2,y2,x3,y3]。

    该函数将每个旋转框的中心坐标、宽高和角度转换为四个顶点的坐标。转换过程是：
    1. 在局部坐标系（以框中心为原点）中定义一个标准矩形的四个顶点。
    2. 使用旋转矩阵将这些顶点旋转指定角度。
    3. 将旋转后的顶点坐标平移到全局坐标系（加上框的中心坐标）。
    4. 使用 `get_best_begin_point_single` 函数标准化顶点顺序，使其从左上角开始顺时针或逆时针排列。

    Args:
        rboxes (list of lists or numpy.ndarray): 输入的旋转框列表。
            形状应为 [N, 5] 或 [N, >5]，其中 N 是旋转框的数量。
            每个旋转框的前5个元素格式为 [x_ctr, y_ctr, w, h, angle]。
            - x_ctr (float): 旋转框中心的 x 坐标。
            - y_ctr (float): 旋转框中心的 y 坐标。
            - w (float): 旋转框的宽度。
            - h (float): 旋转框的高度。
            - angle (float): 旋转框相对于水平轴的逆时针旋转角度（以弧度为单位）。

    Returns:
        numpy.ndarray: 形状为 [N, 8] 的数组，表示转换后的四边形坐标。
            每行的格式为 [x0, y0, x1, y1, x2, y2, x3, y3]。
            顶点顺序经过 `get_best_begin_point_single` 函数标准化，通常为从某个角点开始的循环顺序。
    """
    # 初始化结果列表
    polys = []

    # 遍历输入的每个旋转框
    for i in range(len(rboxes)):
        # 解包当前旋转框的参数
        x_ctr, y_ctr, width, height, angle = rboxes[i][:5]

        # 在局部坐标系（框中心为原点）中定义标准矩形的左上和右下坐标
        # tl: top-left, br: bottom-right
        tl_x, tl_y = -width / 2, -height / 2
        br_x, br_y = width / 2, height / 2

        # 将四个顶点坐标组织成一个 2x4 的矩阵 [[x_coords], [y_coords]]
        # 顺序为：左上 (tl_x, tl_y), 右上 (br_x, tl_y), 右下 (br_x, br_y), 左下 (tl_x, br_y)
        # 注意：这里顺序是 [tl, tr, br, bl]
        rect = np.array([[tl_x, br_x, br_x, tl_x],  # x 坐标
                         [tl_y, tl_y, br_y, br_y]])  # y 坐标

        # 构建 2D 旋转变换矩阵 R
        # R = [[cos(angle), -sin(angle)],
        #      [sin(angle),  cos(angle)]]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a],
                      [sin_a, cos_a]])

        # 应用旋转变换 R @ rect，得到旋转后的顶点坐标（仍在局部坐标系）
        # 结果 poly 是一个 2x4 矩阵，poly[0, :] 是 x 坐标，poly[1, :] 是 y 坐标
        poly_rotated = R.dot(rect)

        # 将旋转后的顶点坐标从局部坐标系平移到全局坐标系
        # 加上框的中心坐标 (x_ctr, y_ctr)
        x_coords = poly_rotated[0, :] + x_ctr  # [x0, x1, x2, x3]
        y_coords = poly_rotated[1, :] + y_ctr  # [y0, y1, y2, y3]

        # 将 x, y 坐标交错排列成 [x0, y0, x1, y1, x2, y2, x3, y3] 的格式
        poly_flat = np.array([x_coords[0], y_coords[0],
                              x_coords[1], y_coords[1],
                              x_coords[2], y_coords[2],
                              x_coords[3], y_coords[3]], dtype=np.float32)

        # 调用函数标准化顶点顺序，使其具有更好的一致性
        poly_standardized = get_best_begin_point_single(poly_flat)

        # 将标准化后的坐标添加到结果列表
        polys.append(poly_standardized)

    polys = np.array(polys)
    return polys


def box2corners(box):
    """
    将旋转框坐标 (中心点, 宽, 高, 角度) 转换为四个角点坐标。

    该函数将一批旋转边界框（格式为 [x, y, w, h, angle]）转换为其四个顶点的笛卡尔坐标。
    转换过程包括：
    1. 从中心坐标系（框中心为原点）计算未旋转的四个角点。
    2. 使用旋转矩阵将这些角点旋转指定角度。
    3. 将旋转后的角点坐标平移到全局坐标系（加上框的中心坐标）。

    Args:
        box (torch.Tensor): 输入的旋转框张量。
            形状为 (B, N, 5)，其中 B 是批次大小，N 是框的数量。
            每个框的格式为 (x, y, w, h, alpha)，其中：
            - x (float): 框中心的 x 坐标。
            - y (float): 框中心的 y 坐标。
            - w (float): 框的宽度。
            - h (float): 框的高度。
            - alpha (float): 框相对于水平轴的逆时针旋转角度（以弧度为单位），
              假设角度范围在 [0, π/2) 内。

    Returns:
        torch.Tensor: 输出的角点坐标张量。
            形状为 (B, N, 4, 2)，其中 4 是四个角点，2 是 (x, y) 坐标。
            角点顺序为：右上、左上、左下、右下（相对于旋转后的框）。
    """
    # 获取批次大小 B
    B = box.shape[0]

    # 将输入框的五个维度分离
    # x, y: 中心坐标 (B, N, 1)
    # w, h: 宽高 (B, N, 1)
    # alpha: 旋转角度 (B, N, 1)
    x, y, w, h, alpha = torch.split(box, 1, dim=-1)

    # 在局部坐标系（框中心为原点）中定义未旋转的四个角点的 x 偏移
    # 顺序：右上(0.5*w, -0.5*h), 左上(-0.5*w, -0.5*h), 左下(-0.5*w, 0.5*h), 右下(0.5*w, 0.5*h)
    # 这里先定义单位偏移，然后乘以宽高
    x_offsets_unit = torch.tensor(
        [0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=box.device
    ).reshape((1, 1, 4))  # (1, 1, 4)
    # 将单位偏移乘以宽度 w，得到实际的 x 偏移 (B, N, 4)
    x_offsets = x_offsets_unit * w

    # 在局部坐标系中定义未旋转的四个角点的 y 偏移
    y_offsets_unit = torch.tensor(
        [-0.5, 0.5, 0.5, -0.5], dtype=torch.float32, device=box.device
    ).reshape((1, 1, 4))  # (1, 1, 4)
    # 将单位偏移乘以高度 h，得到实际的 y 偏移 (B, N, 4)
    y_offsets = y_offsets_unit * h

    # 将 x 和 y 偏移堆叠成一个张量，表示局部坐标系下的角点 (B, N, 4, 2)
    corners_local = torch.stack([x_offsets, y_offsets], dim=-1)  # (B, N, 4, 2)

    # 计算旋转角度的正弦和余弦值 (B, N, 1)
    sin_alpha = torch.sin(alpha)
    cos_alpha = torch.cos(alpha)

    # 构建 2x2 旋转变换矩阵 R 的行向量
    # R = [[cos(alpha), -sin(alpha)],
    #      [sin(alpha),  cos(alpha)]]
    row1 = torch.cat([cos_alpha, -sin_alpha], dim=-1)  # (B, N, 2) -> [cos, -sin] (第一行)
    row2 = torch.cat([sin_alpha, cos_alpha], dim=-1)  # (B, N, 2) -> [sin, cos] (第二行)
    # ---

    # 将两行堆叠成旋转矩阵 (B, N, 2, 2)
    rot_matrix = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)

    # 重塑张量以进行批量矩阵乘法 (B*N, 4, 2) @ (B*N, 2, 2) -> (B*N, 4, 2)
    # corners_local: (B, N, 4, 2) -> (B*N, 4, 2)
    # rot_matrix: (B, N, 2, 2) -> (B*N, 2, 2)
    corners_local_reshaped = corners_local.reshape([-1, 4, 2])
    rot_matrix_reshaped = rot_matrix.reshape([-1, 2, 2])

    # 应用旋转变换
    corners_rotated_reshaped = torch.bmm(corners_local_reshaped, rot_matrix_reshaped)  # (B*N, 4, 2)

    # 将结果重塑回原始批次形状 (B*N, 4, 2) -> (B, N, 4, 2)
    corners_rotated = corners_rotated_reshaped.reshape([B, -1, 4, 2])

    # 将旋转后的角点从局部坐标系平移到全局坐标系
    # 加上框的中心坐标 (x, y)
    corners_final = corners_rotated.clone()
    corners_final[..., 0] += x.squeeze(-1)
    corners_final[..., 1] += y.squeeze(-1)

    return corners_final


def check_points_in_polys(points, polys):
    """
    检查点是否在由四个顶点定义的四边形（多边形）内部。
    该函数使用向量投影的方法来判断点与矩形（或平行四边形）的位置关系。
    输入的四边形是凸四边形，并且顶点按顺序排列（例如，a->b->c->d->a）。

    Args:
        points (torch.Tensor): 形状为 (L, 2) 的张量，包含 L 个待检查的点坐标 (x, y)。
        polys (torch.Tensor): 形状为 (B, N, 4, 2) 的张量，包含 B 批次、N 个四边形，
                              每个四边形由 4 个顶点 (x, y) 定义。

    Returns:
        torch.Tensor: 形状为 (B, N, L) 的布尔张量，指示每个点是否在对应的多边形内。
                      is_in_polys[b, n, l] 为 True 表示第 l 个点在第 b 批次的第 n 个多边形内。
    """
    # [L, 2] -> [1, 1, L, 2] -> [1, 1, L, 2] (为了与 polys 广播)
    points_expanded = points.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, L, 2]

    # [B, N, 4, 2] -> [B, N, 1, 2] (取每个四边形的四个顶点)
    # 这里假设顶点顺序是 a, b, c, d
    a = polys[:, :, 0:1, :]  # shape: [B, N, 1, 2]
    b = polys[:, :, 1:2, :]  # shape: [B, N, 1, 2]
    c = polys[:, :, 2:3, :]  # shape: [B, N, 1, 2]
    d = polys[:, :, 3:4, :]  # shape: [B, N, 1, 2]

    # 计算四边形的两条边向量 (例如，a->b 和 a->d)
    ab = b - a  # shape: [B, N, 1, 2]
    ad = d - a  # shape: [B, N, 1, 2]

    # 计算从四边形顶点 a 到待检查点的向量 ap
    # points_expanded: [1, 1, L, 2], a: [B, N, 1, 2] -> 广播后 ap: [B, N, L, 2]
    ap = points_expanded - a

    # 计算边向量 ab 和 ad 的模长平方 (即向量与自身的点积)
    # sum(..., dim=-1) 沿着最后一个维度 (x, y) 求和
    # norm_ab: [B, N, 1]
    norm_ab = torch.sum(ab * ab, dim=-1, keepdim=False)
    # norm_ad: [B, N, 1]
    norm_ad = torch.sum(ad * ad, dim=-1, keepdim=False)

    # 计算向量 ap 与边向量 ab 和 ad 的点积
    # ap: [B, N, L, 2], ab: [B, N, 1, 2] -> 广播后计算点积 -> [B, N, L]
    # ap_dot_ab = sum(ap * ab, dim=-1) = ap.x * ab.x + ap.y * ab.y
    ap_dot_ab = torch.sum(ap * ab, dim=-1)  # shape: [B, N, L]
    # ap_dot_ad = sum(ap * ad, dim=-1) = ap.x * ad.x + ap.y * ad.y
    ap_dot_ad = torch.sum(ap * ad, dim=-1)  # shape: [B, N, L]

    # 判断点是否在四边形内部
    # 使用向量投影的原理：
    # 1. ap 在 ab 方向上的投影长度: ap_dot_ab / |ab|
    # 2. ap 在 ad 方向上的投影长度: ap_dot_ad / |ad|
    # 点在四边形内当且仅当这两个投影长度都在 [0, |ab|] 和 [0, |ad|] 范围内
    # 即: 0 <= ap_dot_ab <= |ab|^2 且 0 <= ap_dot_ad <= |ad|^2
    # 这里比较的是点积与模长平方，因为 |ab|^2 = ab · ab
    # [B, N, L] 布尔张量
    is_in_polys = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & \
                  (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)

    return is_in_polys


def check_points_in_rotated_boxes(points, boxes):
    """
    检查点是否在旋转框内部。

    该函数通过将旋转框转换为四个角点，然后使用向量投影的方法来判断点与矩形的位置关系。
    输入的旋转框参数为 [x_ctr, y_ctr, width, height, angle]，并将其转换为四边形。

    Args:
        points (torch.Tensor): 形状为 (L, 2) 的张量，包含 L 个待检查的点坐标 (x, y)。
        boxes (torch.Tensor): 形状为 (B, N, 5) 的张量，包含 B 批次、N 个旋转框，
                              每个框的格式为 (x_ctr, y_ctr, w, h, angle)。

    Returns:
        torch.Tensor: 形状为 (B, N, L) 的布尔张量，指示每个点是否在对应的旋转框内。
                      is_in_box[b, n, l] 为 True 表示第 l 个点在第 b 批次的第 n 个框内。
    """
    # 将旋转框 [B, N, 5] 转换为四个角点坐标 [B, N, 4, 2]
    # 需要确保 box2corners 函数是 PyTorch 版本
    corners = box2corners(boxes)  # 假设 box2corners 已经是 torch 版本

    # [L, 2] -> [1, 1, L, 2] (为了与 corners 广播)
    points_expanded = points.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, L, 2]

    # [B, N, 4, 2] -> 分别获取四个顶点 a, b, c, d
    # 这里假设顶点顺序是 a, b, c, d
    a = corners[:, :, 0:1, :]  # shape: [B, N, 1, 2]
    b = corners[:, :, 1:2, :]  # shape: [B, N, 1, 2]
    c = corners[:, :, 2:3, :]  # shape: [B, N, 1, 2]
    d = corners[:, :, 3:4, :]  # shape: [B, N, 1, 2]

    # 计算四边形的两条邻边向量 ab 和 ad
    ab = b - a  # shape: [B, N, 1, 2]
    ad = d - a  # shape: [B, N, 1, 2]

    # 计算从顶点 a 到待检查点的向量 ap
    # points_expanded: [1, 1, L, 2], a: [B, N, 1, 2] -> 广播后 ap: [B, N, L, 2]
    ap = points_expanded - a

    # 计算边向量 ab 和 ad 的模长平方 (即向量与自身的点积)
    # norm_ab: [B, N, 1]
    norm_ab = torch.sum(ab * ab, dim=-1, keepdim=False)  # keepdim=False 以匹配原逻辑
    # norm_ad: [B, N, 1]
    norm_ad = torch.sum(ad * ad, dim=-1, keepdim=False)

    # 计算向量 ap 与边向量 ab 和 ad 的点积
    # ap_dot_ab = sum(ap * ab, dim=-1) = ap.x * ab.x + ap.y * ab.y
    ap_dot_ab = torch.sum(ap * ab, dim=-1)  # shape: [B, N, L]
    # ap_dot_ad = sum(ap * ad, dim=-1) = ap.x * ad.x + ap.y * ad.y
    ap_dot_ad = torch.sum(ap * ad, dim=-1)  # shape: [B, N, L]

    # 判断点是否在四边形内部
    # 使用向量投影的原理：
    # 0 <= ap · ab <= |ab|^2 且 0 <= ap · ad <= |ad|^2
    # [B, N, L] 布尔张量
    is_in_box = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & \
                (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)

    return is_in_box


def rbox_iou(g, p):
    """
    计算两个多边形集合之间的IoU矩阵。

    该函数计算集合g中的每个四边形与集合p中每个四边形的IoU。
    输入是两个包含多边形顶点坐标的数组，输出是IoU矩阵。

    Args:
        g (list or numpy.ndarray): 第一个集合的多边形顶点坐标，形状为 [N, 8]。
                                   每行格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。
        p (list or numpy.ndarray): 第二个集合的多边形顶点坐标，形状为 [M, 8]。
                                   每行格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。

    Returns:
        torch.Tensor: 形状为 [N, M] 的IoU矩阵，其中 iou_matrix[i, j] 表示
                       g[i] 与 p[j] 之间的IoU值。
    """
    from shapely.geometry import Polygon

    # 确保输入是 numpy 数组
    g = np.array(g) # 形状 [N, 8]
    p = np.array(p) # 形状 [M, 8]

    N = g.shape[0]
    M = p.shape[0]

    # 初始化输出矩阵
    iou_matrix = np.zeros((N, M), dtype=np.float32)

    # 遍历所有 g 中的多边形
    for i in range(N):
        # 将 g[i] 转换为 Shapely Polygon
        g_poly_coords = g[i, :8].reshape((4, 2))
        g_poly = Polygon(g_poly_coords)
        g_poly = g_poly.buffer(0) # 尝试修复无效几何体

        # 如果 g_poly 无效，其与所有 p[j] 的 IoU 都是 0，跳过内层循环
        if not g_poly.is_valid:
            continue

        # 预计算 g_poly 的面积
        g_area = g_poly.area
        if g_area == 0:
            # 如果 g_poly 面积为0，其与所有 p[j] 的 IoU 都是 0
            continue # iou_matrix[i, :] 已经是 0

        # 遍历所有 p 中的多边形
        for j in range(M):
            # 将 p[j] 转换为 Shapely Polygon
            p_poly_coords = p[j, :8].reshape((4, 2))
            p_poly = Polygon(p_poly_coords)
            p_poly = p_poly.buffer(0) # 尝试修复无效几何体

            # 检查 p_poly 是否有效
            if not p_poly.is_valid:
                continue # iou_matrix[i, j] 默认为 0

            # 预计算 p_poly 的面积
            p_area = p_poly.area
            if p_area == 0:
                # 如果 p_poly 面积为0，IoU 为 0
                continue # iou_matrix[i, j] 默认为 0

            # 计算交集面积
            try:
                intersection_area = g_poly.intersection(p_poly).area
            except Exception:
                # 如果交集计算失败（例如拓扑错误），IoU 为 0
                continue # iou_matrix[i, j] 默认为 0

            # 计算并集面积
            union_area = g_area + p_area - intersection_area

            # 如果并集面积为0（理论上只有当g和p都面积为0且完全重合才可能，但上面已排除面积为0的情况）
            # 或者由于浮点误差 union_area <= 0
            if union_area <= 0:
                iou_matrix[i, j] = 0.0
            else:
                # 计算并存储 IoU
                iou_matrix[i, j] = intersection_area / union_area

    iou_matrix = torch.from_numpy(iou_matrix)
    return iou_matrix


def rotated_iou_similarity(box1, box2, eps=1e-9, func=''):
    """
    计算两个旋转框集合之间的IoU（交并比）。

    该函数通过调用外部的 `rbox_iou` 操作来计算一批图像中每对旋转框之间的IoU。
    输入 `box1` 和 `box2` 分别代表两个不同集合的旋转框（格式为 [x, y, w, h, angle]），
    输出一个张量，表示 `box1` 中的每个框与 `box2` 中每个框的IoU。

    Args:
        box1 (torch.Tensor): 形状为 [N, M1, 5] 的张量，包含 N 批次、M1 个旋转框。
                             每个框的格式为 [x_ctr, y_ctr, width, height, angle]。
        box2 (torch.Tensor): 形状为 [N, M2, 5] 的张量，包含 N 批次、M2 个旋转框。
                             每个框的格式为 [x_ctr, y_ctr, width, height, angle]。
        eps (float, optional): 用于数值稳定的小量，默认为 1e-9。
        func (str, optional): 可能用于指定IoU计算变体的字符串，当前未使用，默认为空字符串。

    Returns:
        torch.Tensor: 形状为 [N, M1, M2] 的张量，表示 `box1` 和 `box2` 之间每对框的IoU。
                      iou[n, m1, m2] 是 `box1[n, m1]` 和 `box2[n, m2]` 之间的IoU值。
    """
    # 初始化结果列表
    rotated_ious = []

    # 遍历批次中的每一对 box1 和 box2
    # zip(box1, box2) 会将形状 [N, M1, 5] 和 [N, M2, 5] 分别拆分为 N 个 [M1, 5] 和 [M2, 5] 的张量
    for b1, b2 in zip(box1, box2):
        # b1: [M1, 5], b2: [M2, 5]
        # 调用外部函数计算 b1 和 b2 之间的IoU，期望输出形状为 [M1, M2]
        b1 = b1.detach().cpu().numpy()
        b2 = b2.detach().cpu().numpy()
        b1 = rbox2poly_np(b1)
        b2 = rbox2poly_np(b2)
        iou_result = rbox_iou(b1, b2)
        rotated_ious.append(iou_result)

    # 将结果列表堆叠成一个张量，增加一个新的维度（批次维度）在最前面
    # 列表中的每个元素形状为 [M1, M2]，堆叠后形状为 [N, M1, M2]
    return torch.stack(rotated_ious, dim=0)
