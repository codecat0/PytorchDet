#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :keypoint_utils.py
@Author :CodeCat
@Date   :2025/11/10 14:11
"""
"""
this code is based on https://github.com/open-mmlab/mmpose
"""

import cv2
import numpy as np
import torch.nn.functional as F


def rotate_point(pt, angle_rad):
    """
    将一个二维点绕原点旋转指定角度。

    该函数使用标准的二维旋转变换矩阵来计算旋转后的新坐标。
    旋转方向为逆时针（标准数学定义）。

    Args:
        pt (list[float] or tuple[float]): 包含两个元素的列表或元组，表示待旋转的二维点 [x, y]。
        angle_rad (float): 旋转角度，以弧度为单位。正角度表示逆时针旋转。

    Returns:
        list[float]: 包含两个元素的列表，表示旋转后的新点坐标 [new_x, new_y]。
    """
    # 确保输入点是二维的
    assert len(pt) == 2

    # 计算旋转角度的正弦和余弦值
    sn = np.sin(angle_rad)
    cs = np.cos(angle_rad)

    # 应用二维旋转变换矩阵公式:
    # new_x = x * cos(angle) - y * sin(angle)
    # new_y = x * sin(angle) + y * cos(angle)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs

    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """
    计算仿射变换矩阵所需的第三个点。

    在计算仿射变换矩阵时，通常需要三对对应点。给定两个2D点 a 和 b，
    此函数通过将向量 `a - b` 绕点 b 逆时针旋转90度来计算第三个点。
    这样就形成了一个由三个点构成的三角形，可用于确定仿射变换。

    Args:
        a (np.ndarray): 一个二维点，格式为 [x, y]。
        b (np.ndarray): 另一个二维点，格式为 [x, y]。

    Returns:
        np.ndarray: 计算出的第三个点，格式为 [x, y]。
                    该点是通过将向量 (a - b) 逆时针旋转90度后加到点 b 上得到的。
    """
    # 确保输入点是二维的
    assert len(
        a) == 2, 'input of _get_3rd_point should be point with length of 2'
    assert len(
        b) == 2, 'input of _get_3rd_point should be point with length of 2'

    # 计算从点 b 到点 a 的向量
    direction = a - b

    # 将向量 direction = [dx, dy] 逆时针旋转 90 度
    # 逆时针旋转 90 度的变换矩阵是 [[0, -1], [1, 0]]
    # 所以 [dx, dy] 变为 [-dy, dx]
    rotated_direction = np.array([-direction[1], direction[0]], dtype=np.float32)

    # 将旋转后的向量加到点 b 上，得到第三个点
    third_pt = b + rotated_direction

    return third_pt


def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """
    获取仿射变换矩阵，给定中心点、输入尺寸、旋转角度和输出尺寸。

    该函数通过计算源图像上的三个点及其在目标图像上的对应点，
    使用 OpenCV 的 getAffineTransform 函数来生成一个 2x3 的仿射变换矩阵。
    这个矩阵可以用于图像的缩放、旋转和平移操作。

    Args:
        center (np.ndarray[2, ] or list/tuple): 输入图像上感兴趣区域的中心点坐标 (x, y)。
        input_size (np.ndarray[2, ] or scalar or list/tuple): 输入特征图的尺寸 (width, height)。
            如果是标量，则认为宽高相等。
        rot (float): 旋转角度，以度为单位。正角度表示逆时针旋转。
        output_size (np.ndarray[2, ] or list/tuple): 目标热图（或输出图像）的尺寸 (width, height)。
        shift (tuple or list, optional): 相对于宽度/高度的平移偏移比例。
            默认为 (0., 0.)，即无偏移。
        inv (bool, optional): 是否计算逆变换矩阵。
            False (默认): 从源图像 (src) 变换到目标图像 (dst)。
            True: 从目标图像 (dst) 变换到源图像 (src)。

    Returns:
        np.ndarray: 2x3 的仿射变换矩阵 (dtype=np.float32)。
                    该矩阵可用于 cv2.warpAffine 函数。
    """
    # 确保关键点坐标是二维的
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # 处理 input_size 参数，使其成为 [width, height] 的 numpy 数组
    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size  # 重命名以表示这是用于缩放计算的尺寸

    # 将 shift 转换为 numpy 数组
    shift = np.array(shift)
    # 提取源和目标尺寸
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # 将旋转角度从度转换为弧度
    rot_rad = np.pi * rot / 180

    # 计算源图像上相对于中心点的第二个点（考虑旋转）
    # 从中心点向左（-y 方向，因为坐标系 y 向下）移动 src_w * 0.5 的距离，然后旋转
    # rotate_point 函数假设向量是从原点出发的，所以这里是 [0, -src_w * 0.5]
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)

    # 计算目标图像上相对于中心点的第二个点（不考虑旋转，仅缩放）
    # 从中心点向左移动 dst_w * 0.5 的距离
    dst_dir = np.array([0., dst_w * -0.5])

    # 初始化源图像上的三个点 src (3x2 矩阵)
    src = np.zeros((3, 2), dtype=np.float32)
    # 第一个点：根据 shift 偏移后的中心点
    src[0, :] = center + scale_tmp * shift
    # 第二个点：第一个点加上旋转后的方向向量，再加偏移
    src[1, :] = center + src_dir + scale_tmp * shift
    # 第三个点：根据前两个点计算，形成一个三角形，用于确定仿射变换
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # 初始化目标图像上的三个点 dst (3x2 矩阵)
    dst = np.zeros((3, 2), dtype=np.float32)
    # 第一个点：目标图像的中心点
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    # 第二个点：目标图像的中心点加上方向向量
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    # 第三个点：根据前两个点计算
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    # 根据 inv 参数决定变换方向
    if inv:
        # 计算从目标坐标系到源坐标系的变换矩阵 (dst -> src)
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        # 计算从源坐标系到目标坐标系的变换矩阵 (src -> dst)
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    对一个二维点应用仿射变换矩阵。

    该函数将一个二维点 pt 通过给定的 2x3 仿射变换矩阵 t 进行变换。
    仿射变换包括平移、旋转、缩放和剪切等操作。

    Args:
        pt (list or tuple or numpy.ndarray): 包含两个元素的列表、元组或数组，表示待变换的二维点 [x, y]。
        t (numpy.ndarray): 2x3 的仿射变换矩阵。
                           例如: [[a, b, c],
                                  [d, e, f]]
                           表示变换公式:
                           x' = a*x + b*y + c
                           y' = d*x + e*y + f

    Returns:
        numpy.ndarray: 包含两个元素的 numpy 数组，表示变换后的新点坐标 [x', y']。
    """
    # 将输入点 pt 扩展为齐次坐标 (x, y, 1)
    # .T 用于转置，使其成为列向量 (3, 1)，便于矩阵乘法
    new_pt = np.array([pt[0], pt[1], 1.]).T

    # 使用矩阵乘法 np.dot 应用仿射变换矩阵 t
    # t (2x3) @ new_pt (3,) -> (2,) 结果向量 [x', y']
    new_pt = np.dot(t, new_pt)

    # 返回变换后的二维坐标 [x', y']，丢弃齐次坐标的第三维 (始终为1)
    # new_pt[:2] 提取前两个元素
    return new_pt[:2]


def get_affine_mat_kernel(h, w, s, inv=False):
    """
    计算用于图像缩放和填充的仿射变换矩阵和目标尺寸。

    该函数根据给定的原始图像尺寸 (h, w) 和目标短边长度 s，
    计算出一个仿射变换矩阵，用于将图像缩放到一个新尺寸，
    同时保持原始宽高比，并确保新尺寸的最小边为 s，
    最大边按比例缩放并向上取整到最接近的 64 的倍数。

    Args:
        h (int): 原始图像的高度。
        w (int): 原始图像的宽度。
        s (int): 目标图像的短边长度（像素）。
        inv (bool, optional): 是否计算变换矩阵的逆矩阵。默认为 False。

    Returns:
        tuple: 包含两个元素的元组：
            - trans (numpy.ndarray): 2x3 的仿射变换矩阵 (或其逆矩阵)。
            - size_resized (tuple): 缩放后图像的尺寸 (width, height)。
    """
    # 根据原始图像的宽高比决定缩放方式
    if w < h:
        w_ = s
        # 目标高度按比例计算，并向上取整到最接近的 64 的倍数
        # h / w 是原始宽高比，s / w * h 是按比例缩放后的高度
        h_ = int(np.ceil((s / w * h) / 64.) * 64)
        # 用于仿射变换函数的缩放参数
        # scale_w: 用于计算缩放的原始宽度（保持不变）
        scale_w = w
        # scale_h: 用于计算缩放的原始高度（按比例调整以匹配目标尺寸）
        scale_h = h_ / s * h

    else:
        h_ = s
        # 目标宽度按比例计算，并向上取整到最接近的 64 的倍数
        w_ = int(np.ceil((s / h * w) / 64.) * 64)
        # 用于仿射变换函数的缩放参数
        scale_h = h
        scale_w = w_ / s * h

    # 计算原始图像的中心点坐标 (x, y)
    center = np.array([np.round(w / 2.), np.round(h / 2.)])

    # 目标尺寸元组 (width, height)
    size_resized = (w_, h_)

    trans = get_affine_transform(
        center, np.array([scale_w, scale_h]), 0, size_resized, inv=inv)

    # 返回变换矩阵和目标尺寸
    return trans, size_resized


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """
    基于 MMPose 实现，计算满足无偏约束的仿射变换矩阵。

    该函数根据论文 "The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation (CVPR 2020)"
    中提出的无偏数据处理方法，计算一个用于图像变换的 2x3 仿射变换矩阵。该方法旨在解决传统变换中因像素中心对齐方式不同
    而引入的系统性偏差。

    Args:
        theta (float): 旋转角度，以度为单位。
        size_input (np.ndarray or list/tuple): 输入图像的尺寸 [width, height]。
        size_dst (np.ndarray or list/tuple): 目标（输出）图像的尺寸 [width, height]。
        size_target (np.ndarray or list/tuple): 输入平面上感兴趣区域（ROI）的尺寸 [width, height]。

    Returns:
        matrix (np.ndarray): 一个 2x3 的仿射变换矩阵 (dtype=np.float32)。
                             该矩阵可用于 cv2.warpAffine 函数，实现无偏的旋转、缩放和平移。
    """
    # 将旋转角度从度转换为弧度
    theta = np.deg2rad(theta)

    # 初始化 2x3 的变换矩阵
    matrix = np.zeros((2, 3), dtype=np.float32)

    # 计算在 x 和 y 方向上的缩放因子
    # 缩放因子 = 目标图像尺寸 / 输入ROI尺寸
    scale_x = size_dst[0] / size_target[0]  # x 方向缩放
    scale_y = size_dst[1] / size_target[1]  # y 方向缩放

    # --- 填充变换矩阵 ---

    # matrix[0, 0] 和 matrix[0, 1]: x' = matrix[0,0] * x + matrix[0,1] * y + matrix[0,2]
    # 对应 x' = (cos*sc_x) * x + (-sin*sc_x) * y + tx
    matrix[0, 0] = np.cos(theta) * scale_x  # x 对 x' 的旋转和缩放贡献
    matrix[0, 1] = -np.sin(theta) * scale_x  # y 对 x' 的旋转和缩放贡献

    # matrix[0, 2]: x' 的平移分量 tx
    # 这个复杂的表达式确保了变换的无偏性，即 ROI 的中心经过变换后能正确映射到目标图像的中心。
    # -0.5 * size_input[0] * np.cos(theta) + 0.5 * size_input[1] * np.sin(theta):
    #   这部分计算了输入图像中心点经过旋转后，在 x 方向上的坐标变化。
    # + 0.5 * size_target[0]: 将 ROI 的中心（相对于输入图像中心的偏移）考虑进来。
    # 最后乘以 scale_x 应用 x 方向的缩放。
    matrix[0, 2] = scale_x * (
            -0.5 * size_input[0] * np.cos(theta) + 0.5 * size_input[1] * np.sin(theta) + 0.5 * size_target[0]
    )

    # matrix[1, 0] 和 matrix[1, 1]: y' = matrix[1,0] * x + matrix[1,1] * y + matrix[1,2]
    # 对应 y' = (sin*sc_y) * x + (cos*sc_y) * y + ty
    matrix[1, 0] = np.sin(theta) * scale_y  # x 对 y' 的旋转和缩放贡献
    matrix[1, 1] = np.cos(theta) * scale_y  # y 对 y' 的旋转和缩放贡献

    # matrix[1, 2]: y' 的平移分量 ty
    # 逻辑与 tx 类似，但计算的是 y 方向的平移。
    # -0.5 * size_input[0] * np.sin(theta) - 0.5 * size_input[1] * np.cos(theta):
    #   计算输入图像中心点经过旋转后，在 y 方向上的坐标变化。
    # + 0.5 * size_target[1]: 将 ROI 的中心考虑进来。
    # 最后乘以 scale_y 应用 y 方向的缩放。
    matrix[1, 2] = scale_y * (
            -0.5 * size_input[0] * np.sin(theta) - 0.5 * size_input[1] * np.cos(theta) + 0.5 * size_target[1]
    )

    return matrix


def warp_affine_joints(joints, mat):
    """
    对关节点坐标应用由变换矩阵定义的仿射变换。

    该函数将一个或多个二维关节点坐标通过给定的 3x2 仿射变换矩阵 `mat` 进行变换。
    它利用齐次坐标的原理，将 2D 点扩展为 3D 齐次坐标 (x, y, 1)，然后与 3x2 矩阵相乘，
    从而高效地完成平移、旋转、缩放等操作。

    Args:
        joints (np.ndarray[..., 2]): 关节点的原始坐标，最后一维包含 [x, y] 坐标。
                                     可以是任意形状，但最后维度必须为 2。
                                     例如 (N, 2) 或 (H, W, 2)。
        mat (np.ndarray[2, 3] or [3, 2]): 仿射变换矩阵。
                                          **注意**：根据 `affine_transform` 函数的返回值和 `cv2.warpAffine` 的要求，
                                          通常的仿射变换矩阵是 2x3 的。如果输入是 3x2 的矩阵，
                                          其含义可能与标准定义不同。此函数代码逻辑暗示 `mat` 应为 2x3 (尽管注释写 3x2)，
                                          因为它用 `mat.T` (转置) 来与齐次坐标相乘。
                                          为了与代码逻辑一致，假设 `mat` 是 2x3 矩阵。
                                          标准 2x3 仿射矩阵:
                                          [[a, b, c],
                                           [d, e, f]]
                                          表示变换: x' = a*x + b*y + c, y' = d*x + e*y + f

    Returns:
        np.ndarray[..., 2]: 变换后的关节点坐标，形状与输入 `joints` 相同。
    """
    # 确保输入是 numpy 数组
    joints = np.array(joints)
    # 保存原始形状，用于最后恢复
    shape = joints.shape

    # 将 joints 重塑为 (-1, 2)，即将所有关节点压平成二维点列表
    # 例如，如果原 shape 是 (H, W, 2)，则 reshape 后为 (H*W, 2)
    joints = joints.reshape(-1, 2)

    # --- 执行仿射变换 ---
    # 1. 构建齐次坐标矩阵: 在 joints 的第二维 (列) 后添加一列 1s
    #    joints: (N, 2) -> homogeneous_coords: (N, 3)，格式为 [[x1, y1, 1], [x2, y2, 1], ...]
    #    joints[:, 0:1] * 0 + 1 创建一个形状为 (N, 1) 的全1列向量
    ones_col = joints[:, 0:1] * 0 + 1
    homogeneous_coords = np.concatenate((joints, ones_col), axis=1)

    # 2. 应用变换矩阵
    #    如果 mat 是标准的 2x3 仿射矩阵 [[a, b, c], [d, e, f]]
    #    为了使用 np.dot(homogeneous_coords, mat.T)，我们需要 mat.T (3x2)
    #    homogeneous_coords (N, 3) @ mat.T (3, 2) -> result (N, 2)
    #    这正好计算出所有点变换后的 (x', y') 坐标
    #    result[i, 0] = homogeneous_coords[i, 0] * mat[0, 0] + homogeneous_coords[i, 1] * mat[0, 1] + homogeneous_coords[i, 2] * mat[0, 2]
    #                  = joints[i, 0] * mat[0, 0] + joints[i, 1] * mat[0, 1] + mat[0, 2]
    #    result[i, 1] = joints[i, 0] * mat[1, 0] + joints[i, 1] * mat[1, 1] + joints[i, 2] * mat[1, 2]
    #                  = joints[i, 0] * mat[1, 0] + joints[i, 1] * mat[1, 1] + mat[1, 2]
    result = np.dot(homogeneous_coords, mat.T)

    # 3. 将结果 reshape 回原始的形状 (除了最后一维是 2)
    transformed_joints = result.reshape(shape)

    return transformed_joints


def transpred(kpts, h, w, s):
    """
    将关键点坐标从原始图像尺寸转换到经过缩放和填充的图像尺寸。

    该函数首先计算一个从原始图像到缩放后图像的逆仿射变换矩阵，
    然后将这个变换矩阵应用到输入的关键点坐标上，从而得到在新图像坐标系下的关键点位置。

    Args:
        kpts (np.ndarray): 关键点坐标数组，形状通常为 (..., 2) 或 (..., 3)，
                           其中最后一维包含 [x, y] 或 [x, y, confidence]。
                           函数只使用前两个坐标 (x, y)。
        h (int): 原始图像的高度。
        w (int): 原始图像的宽度。
        s (int): 目标图像的短边长度（像素）。

    Returns:
        np.ndarray: 变换后的关键点坐标数组，形状与输入 `kpts` 相同。
                    坐标值对应于经过 `get_affine_mat_kernel` 定义的缩放和填充后的新图像。
    """
    # 计算从缩放后图像到原始图像的逆仿射变换矩阵
    trans, _ = get_affine_mat_kernel(h, w, s, inv=True)

    # 提取关键点的 x, y 坐标，并创建副本以避免修改原始数组
    # 然后使用 warp_affine_joints 函数应用仿射变换矩阵 trans
    # 原始坐标系下的关键点变换到新坐标系下
    transformed_kpts = warp_affine_joints(kpts[..., :2].copy(), trans)

    return transformed_kpts


def transform_preds(coords, center, scale, output_size):
    """
    将坐标从一个尺度和中心转换到另一个尺度和中心（通常是模型输出特征图到原始图像）。

    该函数常用于姿态估计任务中，将模型预测的关键点坐标从输出特征图的坐标系
    转换回原始输入图像的坐标系。它通过计算一个仿射变换矩阵来实现这一转换。

    Args:
        coords (np.ndarray): 输入的坐标数组，形状通常为 (N, 2) 或 (N, 3)，
                             其中 N 是点的数量，最后一维包含 [x, y] 或 [x, y, confidence]。
                             函数只处理前两个坐标 (x, y)。
        center (np.ndarray or list/tuple): 原始图像（或ROI）的中心点坐标 [x, y]。
        scale (np.ndarray or list/tuple or float): 缩放因子。
            通常在姿态估计中，这个值与图像的尺寸相关（例如，高度/200）。
            在仿射变换矩阵计算中，它会被乘以 200。
        output_size (np.ndarray or list/tuple): 输出特征图的尺寸 [width, height]。

    Returns:
        np.ndarray: 转换后的坐标数组，形状与输入 `coords` 相同。
                    坐标值对应于原始图像（或ROI）的坐标系。
    """
    # 创建一个与输入 coords 形状相同的零数组，用于存储转换后的坐标
    target_coords = np.zeros(coords.shape)

    # 计算仿射变换矩阵
    # center: 原始图像的中心点
    # scale * 200: 用于缩放计算的尺寸参数（具体含义取决于 get_affine_transform 的实现）
    # 0: 旋转角度 (这里是0度，即无旋转)
    # output_size: 目标（输出特征图）的尺寸
    # inv=1: 计算逆变换矩阵。这通常意味着矩阵将坐标从 output_size 系统变换回原始图像系统。
    #        具体效果取决于 get_affine_transform 的内部逻辑，但 inv=1 通常用于 "反向" 变换。
    trans = get_affine_transform(center, scale * 200, 0, output_size, inv=1)

    # 遍历每个坐标点
    for p in range(coords.shape[0]):
        # 对每个点的 [x, y] 坐标应用仿射变换矩阵 `trans`
        # affine_transform 函数将 coords[p, 0:2] 这个点按照矩阵 trans 进行变换
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)

    # 返回转换后的坐标数组
    return target_coords


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    """
    计算 Object Keypoint Similarity (OKS) IoU，用于评估姿态估计中关键点的匹配度。

    OKS 是一种类似于 IoU 的度量，但专门用于评估人体姿态估计中的关键点定位精度。
    它考虑了关键点之间的空间距离，并根据关键点类型和目标尺寸进行了归一化。
    OKS 值越高，表示姿态估计越准确。

    Args:
        g (list or numpy.ndarray): 真实（ground truth）关键点坐标，格式为 [x1, y1, v1, x2, y2, v2, ...]。
                                   其中 (xi, yi) 是第 i 个关键点的坐标，vi 是其可见性标志或置信度。
        d (numpy.ndarray): 预测的关键点坐标数组，形状为 (N, K*3)，其中 N 是预测实例数，K 是关键点数。
                           每行格式为 [x1, y1, v1, x2, y2, v2, ...]。
        a_g (float): 真实目标（人体）的面积。
        a_d (numpy.ndarray): 预测目标（人体）的面积数组，形状为 (N,)。
        sigmas (list or numpy.ndarray, optional): 用于归一化距离的关键点类型特定的标准差数组。
                                                  长度应等于关键点数量 K。
                                                  默认值为 COCO 数据集常用的 17 个关键点的 sigmas。
        in_vis_thre (float, optional): 可见性阈值。只有真实和预测的关键点可见性都高于此阈值时，
                                       才会参与 OKS 计算。如果为 None，则所有关键点都参与计算。

    Returns:
        numpy.ndarray: 一个长度为 N 的数组，包含真实关键点 `g` 与 `d` 中每个预测实例的 OKS 值。
    """
    # 如果未提供 sigmas，则使用默认的 COCO sigmas（已除以10）
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0

    # 计算 sigmas 的平方，用于距离计算的分母
    # vars = (sigmas * 2)**2 通常表示关键点的方差（或影响范围）
    vars = (sigmas * 2) ** 2

    # 从真实关键点坐标 g 中分离 x, y 坐标和可见性 v
    # g[0::3] 取出所有 x 坐标 (x1, x2, ...)
    xg = g[0::3]
    # g[1::3] 取出所有 y 坐标 (y1, y2, ...)
    yg = g[1::3]
    # g[2::3] 取出所有可见性标志或置信度 (v1, v2, ...)
    vg = g[2::3]

    # 初始化存储 OKS 值的数组，长度为预测实例数 N
    ious = np.zeros((d.shape[0]))

    # 遍历每个预测实例
    for n_d in range(0, d.shape[0]):
        # 从当前预测实例 d[n_d] 中分离 x, y 坐标和可见性 v
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]

        # 计算真实关键点与预测关键点之间的坐标差
        dx = xd - xg
        dy = yd - yg

        # 计算归一化平方距离 e
        # (dx**2 + dy**2) 是平方欧几里得距离
        # vars 是关键点类型特定的归一化因子
        # ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2 是基于目标面积的归一化因子
        # np.spacing(1) 是一个极小值，防止除以零
        # 整个分母用于将距离归一化到目标尺寸
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2

        # 如果设置了可见性阈值
        if in_vis_thre is not None:
            # 找出真实和预测关键点都可见的索引
            # vg > in_vis_thre 和 vd > in_vis_thre 产生布尔数组
            ind = (vg > in_vis_thre) & (vd > in_vis_thre)
            # 只保留满足可见性条件的 e 值
            e = e[ind]

        # 计算 OKS
        # np.exp(-e) 计算每个关键点的匹配分数（距离越近，分数越高）
        # np.sum(np.exp(-e)) / e.shape[0] 对所有（或可见的）关键点求平均
        # 如果 e.shape[0] 为 0（例如，所有关键点都不可见），则 OKS 为 0.0
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0

    # 返回与每个预测实例对应的 OKS 值数组
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    使用 Object Keypoint Similarity (OKS) 作为相似度度量进行非极大值抑制 (NMS)。

    该函数用于姿态估计任务中，去除重复检测到的人体姿态。它首先根据置信度分数对所有预测的姿态进行排序，
    然后贪婪地选择置信度最高的姿态，并移除与之 OKS 重叠度高于阈值的其他姿态。

    Args:
        kpts_db (list): 包含图像中预测关键点的列表。列表中的每个元素是一个字典，通常包含：
                        - 'keypoints': 关键点坐标数组，格式为 [x1, y1, v1, x2, y2, v2, ...]。
                        - 'score': 该姿态的置信度分数。
                        - 'area': 该姿态包围框的面积。
        thresh (float): OKS 重叠度阈值。如果两个姿态的 OKS 大于等于此阈值，则认为它们是重复的，
                        较低置信度的那个将被抑制（移除）。
        sigmas (np.ndarray, optional): 用于计算 OKS 的关键点类型特定的标准差数组。
                                       如果为 None，则使用 `oks_iou` 函数中的默认值。
        in_vis_thre (float, optional): 用于计算 OKS 的可见性阈值。
                                       只有真实和预测的关键点可见性都高于此阈值时，才会参与 OKS 计算。
                                       如果为 None，则所有关键点都参与计算。

    Returns:
        keep (list): 一个包含应保留的姿态索引的列表。这些索引对应于输入 `kpts_db` 的索引。
    """
    # 如果输入列表为空，则直接返回空列表
    if len(kpts_db) == 0:
        return []

    # 提取所有预测姿态的置信度分数
    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    # 提取所有预测姿态的关键点坐标，并将其展平为一维数组
    kpts = np.array(
        [kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    # 提取所有预测姿态的面积
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    # 对置信度分数进行降序排序，获取排序后的索引
    # order[0] 是置信度最高的姿态的原始索引
    order = scores.argsort()[::-1]

    # 初始化保留的索引列表
    keep = []
    # 循环处理排序后的索引列表
    while order.size > 0:
        # 选择当前置信度最高的姿态索引
        i = order[0]
        # 将该索引添加到保留列表中
        keep.append(i)

        # 计算当前姿态 (kpts[i]) 与剩余所有姿态 (kpts[order[1:]]) 的 OKS IoU
        # areas[i] 是当前姿态的面积，areas[order[1:]] 是剩余姿态的面积
        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, in_vis_thre)

        # 找出 OKS 小于等于阈值的索引
        # 这些索引对应于 `order[1:]` 中的姿态，它们与当前姿态 `i` 不够相似，可以保留
        # np.where 返回的是在 `oks_ovr` (即 `order[1:]`) 中的索引
        inds = np.where(oks_ovr <= thresh)[0]
        # 更新 order 列表，只保留未被抑制的姿态索引
        # inds + 1 是因为 `order[1:]` 相对于原 `order` 的偏移
        order = order[inds + 1]

    # 返回经过 NMS 后保留的姿态索引列表
    return keep


def rescore(overlap, scores, thresh, type='gaussian'):
    """
    根据重叠度 (overlap) 重新调整预测框的置信度分数 (scores)。

    该函数通常用于非极大值抑制 (NMS) 的后续步骤中，目的是降低与高分预测框重叠度较高的低分预测框的分数，
    从而在最终选择时更倾向于保留重叠度低的框，或更大幅度地抑制重叠度高的框。

    Args:
        overlap (numpy.ndarray): 一个一维数组，表示待处理的预测框与某个高分框的重叠度 (如 IoU 或 OKS)。
                                 其长度应与 scores 相同。
        scores (numpy.ndarray): 一个一维数组，包含待处理的预测框的原始置信度分数。
                                该数组将被就地修改（如果使用 'linear' 类型）或创建新数组（如果使用 'gaussian' 类型）。
        thresh (float): 用于调整分数的阈值。其具体作用取决于 `type` 参数。
        type (str, optional): 重新评分的类型。目前支持 'linear' 和 'gaussian'。
                              默认为 'gaussian'。

    Returns:
        numpy.ndarray: 重新调整后的分数数组，形状与输入 `scores` 相同。
    """
    # 确保 overlap 和 scores 的长度相同
    assert overlap.shape[0] == scores.shape[0]

    # 选择线性重评方式
    if type == 'linear':
        # 找出重叠度大于等于阈值的索引
        inds = np.where(overlap >= thresh)[0]
        # 对于这些索引处的分数，应用线性衰减公式: new_score = old_score * (1 - overlap)
        # overlap 越大，(1 - overlap) 越小，分数降低越多。当 overlap >= 1 时，分数变为 0。
        scores[inds] = scores[inds] * (1 - overlap[inds])
        # 返回修改后的 scores 数组
        return scores
    else:
        # 应用高斯衰减公式: new_score = old_score * exp(-(overlap^2) / thresh)
        # 这是一种更平滑的衰减方式。overlap 越大，指数部分越小（负数），exp() 结果越接近 0，分数降低越多。
        # `thresh` 在这里控制衰减的陡峭程度，值越小，衰减越快。
        # 注意：这里的 thresh 用法与 'linear' 中不同，它不是重叠度的阈值，而是高斯函数的参数。
        scores = scores * np.exp(-overlap ** 2 / thresh)
        # 返回计算后的新分数数组
        return scores


def soft_oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    使用软非极大值抑制 (Soft-NMS) 和 Object Keypoint Similarity (OKS) 进行姿态去重。

    与传统的 NMS（硬性抑制，直接移除重叠度高的框）不同，Soft-NMS 会根据重叠度降低预测框的置信度分数，
    而不是直接删除它们。这有助于在密集场景中保留更多潜在的正确检测。
    该函数使用 OKS 作为衡量两个姿态之间相似度的指标。

    Args:
        kpts_db (list): 包含图像中预测关键点的列表。列表中的每个元素是一个字典，通常包含：
                        - 'keypoints': 关键点坐标数组，格式为 [x1, y1, v1, x2, y2, v2, ...]。
                        - 'score': 该姿态的原始置信度分数。
                        - 'area': 该姿态包围框的面积。
        thresh (float): 用于调整分数的阈值。在 `rescore` 函数中使用，具体作用取决于其内部实现。
                        在 'gaussian' 类型中，它控制衰减的陡峭程度。
        sigmas (np.ndarray, optional): 用于计算 OKS 的关键点类型特定的标准差数组。
                                       如果为 None，则使用 `oks_iou` 函数中的默认值。
        in_vis_thre (float, optional): 用于计算 OKS 的可见性阈值。
                                       只有真实和预测的关键点可见性都高于此阈值时，才会参与 OKS 计算。
                                       如果为 None，则所有关键点都参与计算。

    Returns:
        keep (list): 一个包含应保留的姿态索引的列表。这些索引对应于输入 `kpts_db` 的原始索引。
                     保留的姿态是经过 Soft-NMS 处理后置信度仍然较高的那些。
    """
    # 如果输入列表为空，则直接返回空列表
    if len(kpts_db) == 0:
        return []

    # 提取所有预测姿态的原始置信度分数
    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    # 提取所有预测姿态的关键点坐标，并将其展平为一维数组
    kpts = np.array(
        [kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    # 提取所有预测姿态的面积
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    # 对原始分数对应的索引进行降序排序，得到初始处理顺序
    order = scores.argsort()[::-1]
    # 根据排序后的索引重新排列分数数组，使其与 order 顺序一致
    scores = scores[order]

    # 设置最大保留姿态数量（防止处理时间过长）
    # max_dets = order.size # 如果想保留所有可能的姿态，可以取消注释此行并注释下一行
    max_dets = 20
    # 初始化一个数组来存储最终保留的姿态索引
    keep = np.zeros(max_dets, dtype=np.intp)
    # 计数器，记录已保留的姿态数量
    keep_cnt = 0

    # 当还有待处理的姿态且保留数量未达到上限时，继续循环
    while order.size > 0 and keep_cnt < max_dets:
        # 选择当前分数最高的姿态索引
        i = order[0]

        # 计算当前姿态 (kpts[i]) 与剩余所有姿态 (kpts[order[1:]]) 的 OKS IoU
        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, in_vis_thre)

        # 将当前姿态从待处理列表中移除 (处理过的不再参与后续重叠度计算)
        order = order[1:]
        # 使用 rescore 函数根据计算出的 OKS 重叠度调整剩余姿态的分数
        # 这里传入的是排序后分数数组的剩余部分 scores[1:]
        scores = rescore(oks_ovr, scores[1:], thresh)

        # 对调整后的分数再次进行降序排序，得到新的处理顺序
        tmp = scores.argsort()[::-1]
        # 根据新顺序重新排列 order 和 scores 数组
        order = order[tmp]
        scores = scores[tmp]

        # 将当前处理的、分数最高的原始索引 (i) 添加到保留列表中
        keep[keep_cnt] = i
        # 增加保留计数
        keep_cnt += 1

    # 截取 keep 数组到实际保留的数量
    keep = keep[:keep_cnt]

    # 返回保留的姿态索引列表
    return keep


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """
    对输入张量进行上采样或下采样。

    这个函数是 PyTorch 的 `torch.nn.functional.interpolate` 的一个封装，
    添加了在特定条件下（使用 align_corners=True 且缩放比例不是整数倍时）的警告提示。

    Args:
        input (torch.Tensor): 输入张量，形状通常为 (N, C, H, W) 或 (N, C, D, H, W)。
        size (int or tuple, optional): 输出张量的空间尺寸 (例如 H, W 或 D, H, W)。
                                       如果提供了 `scale_factor`，则此参数必须为 None。
        scale_factor (float or tuple, optional): 缩放因子。如果提供了 `size`，则此参数必须为 None。
        mode (str, optional): 插值模式。常用的有 'nearest', 'bilinear', 'bicubic', 'trilinear' 等。
                              默认为 'nearest'。
        align_corners (bool, optional): 如果为 True，输入和输出张量的角点像素的角点对齐。
                                        这在处理特定尺寸缩放时很重要，会影响插值结果。
        warning (bool, optional): 是否启用警告。如果为 True，当检测到可能产生非对齐结果的情况时，
                                  会发出警告。默认为 True。

    Returns:
        torch.Tensor: 经过插值后的输出张量。
    """
    # 如果启用了警告，并且提供了 size 参数且 align_corners=True
    if warning and size is not None and align_corners:
        # 获取输入张量的空间尺寸 (H, W 或 D, H, W)
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        # 获取目标输出尺寸
        output_h, output_w = tuple(int(x) for x in size)

        # 检查输出尺寸是否大于输入尺寸（即进行上采样）
        if output_h > input_h or output_w > input_w:
            # 这种情况通常意味着缩放不是整数倍，使用 align_corners=True 时可能导致像素不对齐
            if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and
                    (output_h - 1) % (input_h - 1) != 0 and
                    (output_w - 1) % (input_w - 1) != 0):
                warnings.warn(
                    f'When align_corners={align_corners}, '
                    'the output would more aligned if '
                    f'input size {(input_h, input_w)} is `x+1` and '
                    f'out size {(output_h, output_w)} is `nx+1`')

    return F.interpolate(input, size, scale_factor, mode, align_corners)


def flip_back(output_flipped, flip_pairs, target_type='GaussianHeatmap'):
    """
    将翻转后的热图（heatmaps）翻转回原始图像的对应形式。

    在姿态估计中，有时会对输入图像进行水平翻转以进行数据增强或测试时增强 (Test-Time Augmentation)。
    模型在翻转后的图像上预测出的热图也需要相应地“翻转回来”，才能与原始图像的坐标系对齐。
    此函数执行这个“翻转回来”的操作。

    注意:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): 从翻转图像上获得的输出热图。
        flip_pairs (list[tuple()]): 关键点对的列表，这些关键点在图像翻转时是镜像对称的
                                   （例如，左耳 -- 右耳）。
                                   例如 [(0, 1), (2, 3)] 表示关键点0和1互为镜像，2和3互为镜像。
        target_type (str, optional): 热图的目标类型。默认为 'GaussianHeatmap'。
                                   如果是 'CombinedTarget'，则每个关键点可能有3个通道 (x, y, confidence)。

    Returns:
        np.ndarray: 翻转回原始图像形式的热图。
    """
    # 确保输入的热图是4维的 (N, K, H, W)
    assert len(output_flipped.shape) == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'

    # 保存原始形状
    shape_ori = output_flipped.shape
    # 初始化通道数
    channels = 1

    # 如果目标类型是 CombinedTarget，通常每个关键点有3个通道
    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        # 对于 CombinedTarget，y 坐标的预测值（通常是第2个通道，索引为1）在翻转后需要取反
        # 因为 y 轴方向在图像坐标系中是向下的，翻转后 y 的变化方向可能需要调整
        # [1::3] 表示从索引1开始，每隔3个取一个元素，即所有 y 坐标通道
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]

    # 重塑数组，将关键点数量 K 拆分为 K/channels 个关键点，每个关键点有 channels 个通道
    # 形状变为 (N, K/channels, channels, H, W)
    output_flipped = output_flipped.reshape((shape_ori[0], -1, channels,
                                             shape_ori[2], shape_ori[3]))

    if isinstance(output_flipped, torch.Tensor):
        output_flipped_back = output_flipped.clone()
    else:
        output_flipped_back = output_flipped.copy()

    # 交换左右对称的关键点通道
    # 例如，如果左耳是索引 0，右耳是索引 1，则将翻转图像上预测的左耳热图替换为右耳的，
    # 将右耳热图替换为左耳的
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]

    # 将形状恢复到原始输入的 K 维度
    output_flipped_back = output_flipped_back.reshape(shape_ori)

    # 最后一步，将整个热图数组在宽度维度 (W) 上进行水平翻转
    # [..., ::-1] 表示对最后一个维度（宽度）进行反向切片
    output_flipped_back = output_flipped_back[..., ::-1]

    # 返回翻转回原始形式的热图
    return output_flipped_back


def _calc_distances(preds, targets, mask, normalize):
    """
    计算预测关键点与真实关键点之间的归一化距离。

    该函数是姿态估计评估中的一个核心步骤，用于衡量预测的准确性。
    它会根据可见性掩码忽略不可见的关键点，并使用提供的归一化因子来标准化距离。

    Note:
        batch_size: N
        num_keypoints: K
        keypoint 维度: D (通常, D=2 表示 x,y 坐标, D=3 表示 x,y,v 可见性)

    Args:
        preds (np.ndarray[N, K, D]): 预测的关键点坐标。
        targets (np.ndarray[N, K, D]): 真实（Groundtruth）关键点坐标。
        mask (np.ndarray[N, K]): 关键点可见性掩码。False 表示不可见关键点，
                                 True 表示可见关键点。不可见关键点在计算精度时将被忽略。
        normalize (np.ndarray[N, D]): 归一化因子数组，通常为热图尺寸或其他尺度因子。

    Returns:
        np.ndarray[K, N]: 归一化后的距离数组。
                          如果目标关键点缺失（由 mask 指示）或归一化因子无效，则对应距离为 -1。
    """
    # 获取批次大小 N、关键点数量 K 和坐标维度 D
    N, K, _ = preds.shape

    # 复制 mask 以避免修改原始输入
    _mask = mask.copy()
    # 如果归一化因子的某个批次的所有维度都为0，则将该批次所有关键点的 mask 设为 False
    # np.where((normalize == 0).sum(1))[0] 找出 normalize 中某行全为0的批次索引
    _mask[np.where((normalize == 0).sum(1))[0], :] = False

    # 初始化距离数组，所有值设为 -1（表示无效或忽略的距离）
    distances = np.full((N, K), -1, dtype=np.float32)

    # 处理无效的归一化值（<= 0），将其设为一个很大的数，以避免除以零或负数
    normalize[np.where(normalize <= 0)] = 1e6

    # 计算归一化距离
    # (preds - targets) / normalize[:, None, :] 计算每个点的差值并归一化
    # normalize[:, None, :] 的形状变为 (N, 1, D)，通过广播应用于 (N, K, D)
    # np.linalg.norm(..., axis=-1) 计算归一化差值向量的 L2 范数（欧几里得距离）
    # [..._mask] 只对可见且有效的关键点计算距离
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)

    # 返回转置后的距离矩阵，形状从 (N, K) 变为 (K, N)
    # 这样外层循环可以遍历关键点 K，内层循环遍历批次 N
    return distances.T


def _distance_acc(distances, thr=0.5):
    """
    计算归一化距离低于给定阈值的百分比，同时忽略值为 -1 的无效距离。

    该函数通常用于姿态估计任务的评估中，衡量预测关键点与真实关键点的匹配精度。
    距离值为 -1 表示该关键点缺失或不可见，应被忽略。

    Note:
        batch_size: N

    Args:
        distances (np.ndarray[N, ]): 归一化后的距离数组，长度为 N。
                                    有效距离为非负数，无效距离标记为 -1。
        thr (float, optional): 距离阈值。低于此阈值的距离被认为预测准确。
                               默认值为 0.5。

    Returns:
        float: 有效距离中低于阈值的百分比（范围在 [0, 1] 之间）。
               如果所有目标关键点都缺失（即没有有效距离），则返回 -1。
    """
    # 创建一个布尔掩码，标记出不等于 -1 的有效距离
    distance_valid = distances != -1

    # 计算有效距离的总数
    num_distance_valid = distance_valid.sum()

    # 如果存在有效距离
    if num_distance_valid > 0:
        # distances[distance_valid] 获取所有有效距离
        # (distances[distance_valid] < thr) 返回一个布尔数组，标记低于阈值的距离
        # .sum() 计算低于阈值的有效距离数量
        # 除以有效距离总数，得到准确率百分比
        return (distances[distance_valid] < thr).sum() / num_distance_valid

    # 如果没有有效距离（所有关键点都缺失），返回 -1
    return -1


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """
    计算每个关键点的 PCK (Percentage of Correct Keypoints) 精度以及所有关键点的平均精度。

    注意:
        PCK 指标衡量身体关节定位的准确性。
        预测位置与真实位置之间的距离通常通过边界框大小进行归一化。
        归一化距离的阈值 (thr) 通常设置为 0.05, 0.1 或 0.2 等。

        - 批次大小: N
        - 关键点数量: K

    Args:
        pred (np.ndarray[N, K, 2]): 预测的关键点坐标。
        gt (np.ndarray[N, K, 2]): 真实（Groundtruth）关键点坐标。
        mask (np.ndarray[N, K]): 关键点可见性掩码。False 表示不可见关节，
                                 True 表示可见关节。不可见关节在计算精度时将被忽略。
        thr (float): PCK 计算的阈值。
        normalize (np.ndarray[N, 2]): 用于高度和宽度的归一化因子。
                                     通常为热图尺寸或其他尺度因子 (例如，人体包围框的尺寸)。

    Returns:
        tuple: 包含关键点精度的元组。

        - acc (np.ndarray[K]): 每个关键点的精度。如果某个关键点的所有样本都不可见，
                               则该关键点的精度为 -1。
        - avg_acc (float): 所有有效关键点的平均精度。
        - cnt (int): 有效关键点的数量（即至少有一个样本可见的关键点数量）。
    """
    # 计算预测坐标与真实坐标之间的归一化距离
    # _calc_distances 函数会根据 mask 和 normalize 因子计算距离
    # 返回形状为 [K, N] 的距离矩阵，其中 K 是关键点数，N 是批次大小
    distances = _calc_distances(pred, gt, mask, normalize)

    # 遍历每个关键点的距离数组，计算其 PCK 准确率
    # _distance_acc 计算一个批次中低于阈值的距离百分比，忽略值为 -1 的距离
    acc = np.array([_distance_acc(d, thr) for d in distances]) # 形状 [K,]

    # 提取有效的精度值（即不为 -1 的值）
    valid_acc = acc[acc >= 0]
    # 计算有效关键点的数量
    cnt = len(valid_acc)
    # 计算所有有效关键点的平均精度
    # 如果没有有效关键点，则平均精度为 0
    avg_acc = valid_acc.mean() if cnt > 0 else 0

    # 返回每个关键点的精度、平均精度和有效关键点数量
    return acc, avg_acc, cnt


def keypoint_auc(pred, gt, mask, normalize, num_step=20):
    """
    计算关键点的 AUC (Area Under Curve) 指标。

    该函数通过在一系列阈值上计算 PCK (Percentage of Correct Keypoints) 准确率，
    并计算这些点与阈值轴构成的曲线下的面积，来评估姿态估计模型的整体性能。
    AUC 值越高，表示模型在不同严格程度的评估标准下都表现良好。

    Note:
        - 批次大小: N
        - 关键点数量: K

    Args:
        pred (np.ndarray[N, K, 2]): 预测的关键点坐标。
        gt (np.ndarray[N, K, 2]): 真实（Groundtruth）关键点坐标。
        mask (np.ndarray[N, K]): 关键点可见性掩码。False 表示不可见关节，
                                 True 表示可见关节。不可见关节在计算精度时将被忽略。
        normalize (float): 用于归一化距离的因子（例如，人体包围框的尺寸）。
                           这个值将被扩展为与 pred 形状兼容的数组。
        num_step (int, optional): 计算 AUC 时的步数（阈值数量）。
                                  默认值为 20。

    Returns:
        float: 曲线下的面积 (AUC)。
               计算方式为：将阈值范围 [0, 1) 均匀分成 num_step 份，
               在每个阈值处计算平均 PCK 准确率，然后使用矩形法（或等价的平均值）近似积分面积。
    """
    # 将单个 normalize 值扩展为形状为 (pred.shape[0], 2) 的数组
    # np.array([[normalize, normalize]]) 创建一个 (1, 2) 的数组
    # np.tile(..., (pred.shape[0], 1)) 将其沿行方向重复 pred.shape[0] 次，沿列方向重复 1 次
    # 最终形状为 (N, 2)，每一行都是 [normalize, normalize]
    nor = np.tile(np.array([[normalize, normalize]]), (pred.shape[0], 1))

    # 生成 x 轴坐标（阈值列表）
    # x = [0.0, 0.05, 0.1, ..., 0.95] (如果 num_step=20)
    # 这些阈值将用于计算不同严格程度下的 PCK 准确率
    x = [1.0 * i / num_step for i in range(num_step)]

    # 初始化 y 轴坐标（对应阈值下的平均 PCK 准确率列表）
    y = []

    # 遍历每个阈值
    for thr in x:
        # 调用 keypoint_pck_accuracy 函数计算在当前阈值下的准确率
        # 传入 pred, gt, mask, 当前阈值 thr, 和扩展后的 normalize 数组 nor
        # 函数返回 acc, avg_acc, cnt，这里只关心 avg_acc
        _, avg_acc, _ = keypoint_pck_accuracy(pred, gt, mask, thr, nor)
        # 将计算得到的平均准确率添加到 y 列表中
        y.append(avg_acc)

    # 初始化 AUC 值
    auc = 0
    # 遍历每个阈值对应的准确率
    for i in range(num_step):
        # 使用矩形法计算面积
        # 每个矩形的宽度是 1.0 / num_step (因为 x 轴从 0 到 1，分成 num_step 段)
        # 每个矩形的高度是 y[i] (即在该阈值下的平均准确率)
        # 将所有矩形面积累加起来得到近似的 AUC
        auc += 1.0 / num_step * y[i]

    # 返回计算得到的 AUC 值
    return auc


def keypoint_epe(pred, gt, mask):
    """
    计算关键点的端点误差 (End-Point Error, EPE)。

    EPE 是衡量预测关键点与真实关键点之间欧几里得距离的指标。
    该函数会根据可见性掩码忽略不可见的关键点，并返回所有有效预测的平均误差。

    Note:
        - 批次大小: N
        - 关键点数量: K

    Args:
        pred (np.ndarray[N, K, 2]): 预测的关键点坐标。
        gt (np.ndarray[N, K, 2]): 真实（Groundtruth）关键点坐标。
        mask (np.ndarray[N, K]): 关键点可见性掩码。False 表示不可见关节，
                                 True 表示可见关节。不可见关节在计算精度时将被忽略。

    Returns:
        float: 平均端点误差 (Average end-point error)。
               计算方式为：所有有效 (mask=True 且坐标有效) 预测的欧几里得距离之和 / 有效预测的数量。
               如果没有有效预测，则返回 0。
    """
    # 创建一个归一化因子数组，这里设置为全 1，意味着不对距离进行额外的尺度归一化
    # 形状为 (N, 2)，其中 2 对应 x 和 y 坐标
    normalize = np.ones((pred.shape[0], pred.shape[2]), dtype=np.float32)

    # 调用 _calc_distances 函数计算归一化后的距离矩阵
    # 输入 pred, gt, mask, normalize
    # 返回形状为 [K, N] 的距离矩阵，其中 -1 表示无效或忽略的距离
    distances = _calc_distances(pred, gt, mask, normalize)

    # 提取所有非 -1 的有效距离值，形成一个一维数组
    distance_valid = distances[distances != -1]

    # 计算平均端点误差
    # 将所有有效距离求和，然后除以有效距离的数量
    # 使用 max(1, len(distance_valid)) 作为分母，防止除以零的情况
    # 如果没有有效距离 (len(distance_valid) == 0)，则返回 0
    return distance_valid.sum() / max(1, len(distance_valid))