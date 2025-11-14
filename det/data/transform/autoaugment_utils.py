#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :autoaugment_utils.py
@Author :CodeCat
@Date   :2025/11/6 11:13
"""

# Reference:
#   https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py


import inspect
import math
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from copy import deepcopy

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

# Represents an invalid bounding box that is used for checking for padding
# lists of bounding box coordinates for a few augmentation operations
_INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]


def policy_v0():
    """Autoaugment policy that was used in AutoAugment Detection Paper."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [
        [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
        [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
        [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
    ]
    return policy


def policy_v1():
    """Autoaugment policy that was used in AutoAugment Detection Paper."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [
        [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
        [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
        [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
        [('Color', 0.0, 0), ('ShearX_Only_BBoxes', 0.8, 4)],
        [('ShearY_Only_BBoxes', 0.8, 2), ('Flip_Only_BBoxes', 0.0, 10)],
        [('Equalize', 0.6, 10), ('TranslateX_BBox', 0.2, 2)],
        [('Color', 1.0, 10), ('TranslateY_Only_BBoxes', 0.4, 6)],
        [('Rotate_BBox', 0.8, 10), ('Contrast', 0.0, 10)],  # ,
        [('Cutout', 0.2, 2), ('Brightness', 0.8, 10)],
        [('Color', 1.0, 6), ('Equalize', 1.0, 2)],
        [('Cutout_Only_BBoxes', 0.4, 6), ('TranslateY_Only_BBoxes', 0.8, 2)],
        [('Color', 0.2, 8), ('Rotate_BBox', 0.8, 10)],
        [('Sharpness', 0.4, 4), ('TranslateY_Only_BBoxes', 0.0, 4)],
        [('Sharpness', 1.0, 4), ('SolarizeAdd', 0.4, 4)],
        [('Rotate_BBox', 1.0, 8), ('Sharpness', 0.2, 8)],
        [('ShearY_BBox', 0.6, 10), ('Equalize_Only_BBoxes', 0.6, 8)],
        [('ShearX_BBox', 0.2, 6), ('TranslateY_Only_BBoxes', 0.2, 10)],
        [('SolarizeAdd', 0.6, 8), ('Brightness', 0.8, 10)],
    ]
    return policy


def policy_vtest():
    """Autoaugment test policy for debugging."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [[('TranslateX_BBox', 1.0, 4), ('Equalize', 1.0, 10)], ]
    return policy


def policy_v2():
    """Additional policy that performs well on object detection."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [
        [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)],
        [('Rotate_BBox', 0.4, 8), ('Sharpness', 0.4, 2),
         ('Rotate_BBox', 0.8, 10)],
        [('TranslateY_BBox', 1.0, 8), ('AutoContrast', 0.8, 2)],
        [('AutoContrast', 0.4, 6), ('ShearX_BBox', 0.8, 8),
         ('Brightness', 0.0, 10)],
        [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10),
         ('AutoContrast', 0.6, 0)],
        [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)],
        [('TranslateY_BBox', 0.0, 4), ('Equalize', 0.6, 8),
         ('Solarize', 0.0, 10)],
        [('TranslateY_BBox', 0.2, 2), ('ShearY_BBox', 0.8, 8),
         ('Rotate_BBox', 0.8, 8)],
        [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)],
        [('Color', 0.8, 4), ('TranslateY_BBox', 1.0, 6),
         ('Rotate_BBox', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('BBox_Cutout', 1.0, 4), ('Cutout', 0.2, 8)],
        [('Rotate_BBox', 0.0, 0), ('Equalize', 0.6, 6),
         ('ShearY_BBox', 0.6, 8)],
        [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2),
         ('Brightness', 0.2, 2)],
        [('TranslateY_BBox', 0.4, 8), ('Solarize', 0.4, 6),
         ('SolarizeAdd', 0.2, 10)],
        [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
    ]
    return policy


def policy_v3():
    """"Additional policy that performs well on object detection."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [
        [('Posterize', 0.8, 2), ('TranslateX_BBox', 1.0, 8)],
        [('BBox_Cutout', 0.2, 10), ('Sharpness', 1.0, 8)],
        [('Rotate_BBox', 0.6, 8), ('Rotate_BBox', 0.8, 10)],
        [('Equalize', 0.8, 10), ('AutoContrast', 0.2, 10)],
        [('SolarizeAdd', 0.2, 2), ('TranslateY_BBox', 0.2, 8)],
        [('Sharpness', 0.0, 2), ('Color', 0.4, 8)],
        [('Equalize', 1.0, 8), ('TranslateY_BBox', 1.0, 8)],
        [('Posterize', 0.6, 2), ('Rotate_BBox', 0.0, 10)],
        [('AutoContrast', 0.6, 0), ('Rotate_BBox', 1.0, 6)],
        [('Equalize', 0.0, 4), ('Cutout', 0.8, 10)],
        [('Brightness', 1.0, 2), ('TranslateY_BBox', 1.0, 6)],
        [('Contrast', 0.0, 2), ('ShearY_BBox', 0.8, 0)],
        [('AutoContrast', 0.8, 10), ('Contrast', 0.2, 10)],
        [('Rotate_BBox', 1.0, 10), ('Cutout', 1.0, 10)],
        [('SolarizeAdd', 0.8, 6), ('Equalize', 0.8, 8)],
    ]
    return policy


def _equal(val1, val2, eps=1e-8):
    return abs(val1 - val2) <= eps


def blend(image1, image2, factor):
    """
    将 image1 和 image2 使用 'factor' 进行混合。

    Args:
        image1 (np.ndarray): 类型为 uint8 的图像张量。
        image2 (np.ndarray): 类型为 uint8 的图像张量。
        factor (float): 大于 0.0 的浮点数值。
            - factor 为 0.0 时表示仅使用 image1。
            - factor 为 1.0 时表示仅使用 image2。
            - factor 在 0.0 和 1.0 之间时，表示在两个图像之间线性插值像素值。
            - factor 大于 1.0 时表示“外推”两个像素值之间的差异，并将结果裁剪到 0 和 255 之间。

    Returns:
        np.ndarray: 类型为 uint8 的混合图像张量。

    """
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.astype(np.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return np.clip(temp, a_min=0, a_max=255).astype(np.uint8)


def cutout(image, pad_size, replace=0):
    """
    对图像应用cutout操作（https://arxiv.org/abs/1708.04552）。

    Args:
        image: 图像张量，类型为uint8。
        pad_size: 指定要生成的零掩码的大小，该掩码将应用于图像。掩码的大小将为
            (2*pad_size x 2*pad_size)。
        replace: 在应用cutout掩码的区域中填充的图像像素值。

    Returns:
        类型为uint8的图像张量。

    Example:
        img = cv2.imread("/home/vis/gry/train/img_data/test.jpg", cv2.COLOR_BGR2RGB)
        new_img = cutout(img, pad_size=50, replace=0)
    """
    image_height, image_width = image.shape[0], image.shape[1]

    cutout_center_height = np.random.randint(low=0, high=image_height)
    cutout_center_width = np.random.randint(low=0, high=image_width)

    lower_pad = np.maximum(0, cutout_center_height - pad_size)
    upper_pad = np.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = np.maximum(0, cutout_center_width - pad_size)
    right_pad = np.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = np.pad(np.zeros(
        cutout_shape, dtype=image.dtype),
                  padding_dims,
                  'constant',
                  constant_values=1)
    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, [1, 1, 3])
    image = np.where(
        np.equal(mask, 0),
        np.ones_like(
            image, dtype=image.dtype) * replace,
        image)
    return image.astype(np.uint8)


def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return np.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = image.astype(np.int64) + addition
    added_image = np.clip(added_image, a_min=0, a_max=255).astype(np.uint8)
    return np.where(image < threshold, added_image, image)


def color(image, factor):
    """use cv2 to deal"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    degenerate = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return blend(degenerate, image, factor)


# refer to https://github.com/4uiiurz1/pytorch-auto-augment/blob/024b2eac4140c38df8342f09998e307234cafc80/auto_augment.py#L197
def contrast(img, factor):
    img = ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = np.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return np.left_shift(np.right_shift(image, shift), shift)


def rotate(image, degrees, replace):
    """
    旋转图像。

    Args:
        image (np.ndarray): 一个类型为 uint8 的图像张量。
        degrees (float): 浮点数，一个标量角度（以度为单位），用于旋转所有图像。如果
            degrees 为正数，图像将顺时针旋转，否则将逆时针旋转。
        replace (np.ndarray): 一个一维张量，包含一个或三个值，用于填充旋转操作引起的空白像素。

    Returns:
        np.ndarray: 旋转后的图像版本。
    """
    image = wrap(image)
    image = Image.fromarray(image)
    image = image.rotate(degrees)
    image = np.array(image, dtype=np.uint8)
    return unwrap(image, replace)


def random_shift_bbox(image,
                      bbox,
                      pixel_scaling,
                      replace,
                      new_min_bbox_coords=None):
    """
    将bbox和图像内容移动到稍微新的随机位置。

    Args:
        image: 3D uint8 Tensor。
        bbox: 一个有4个元素(min_y, min_x, max_y, max_x)的1D Tensor，类型为float，
            表示归一化坐标，范围在0到1之间。新bbox的左上角潜在值将在
            [old_min - pixel_scaling * bbox_height/2, old_min - pixel_scaling * bbox_height/2]
            之间。
        pixel_scaling: 一个在0和1之间的浮点数，指定新的bbox位置将从中采样的像素范围。
        replace: 一个一值或三值的1D tensor，用于填充空像素。
        new_min_bbox_coords: 如果不为None，则这是一个指定新bbox的(min_y, min_x)坐标的元组。
            通常这是随机指定的，但这允许手动设置。坐标是绝对坐标，范围在0到图像高度/宽度之间，类型为int32。

    Returns:
        新的图像，其中包含移动后的bbox位置，以及包含新坐标的新bbox。
    """
    # Obtains image height and width and create helper clip functions.
    image_height, image_width = image.shape[0], image.shape[1]
    image_height = float(image_height)
    image_width = float(image_width)

    def clip_y(val):
        return np.clip(val, a_min=0, a_max=image_height - 1).astype(np.int32)

    def clip_x(val):
        return np.clip(val, a_min=0, a_max=image_width - 1).astype(np.int32)

    # Convert bbox to pixel coordinates.
    min_y = int(image_height * bbox[0])
    min_x = int(image_width * bbox[1])
    max_y = clip_y(image_height * bbox[2])
    max_x = clip_x(image_width * bbox[3])

    bbox_height, bbox_width = (max_y - min_y + 1, max_x - min_x + 1)
    image_height = int(image_height)
    image_width = int(image_width)

    # Select the new min/max bbox ranges that are used for sampling the
    # new min x/y coordinates of the shifted bbox.
    minval_y = clip_y(min_y - np.int32(pixel_scaling * float(bbox_height) /
                                       2.0))
    maxval_y = clip_y(min_y + np.int32(pixel_scaling * float(bbox_height) /
                                       2.0))
    minval_x = clip_x(min_x - np.int32(pixel_scaling * float(bbox_width) / 2.0))
    maxval_x = clip_x(min_x + np.int32(pixel_scaling * float(bbox_width) / 2.0))

    # Sample and calculate the new unclipped min/max coordinates of the new bbox.
    if new_min_bbox_coords is None:
        unclipped_new_min_y = np.random.randint(
            low=minval_y, high=maxval_y, dtype=np.int32)
        unclipped_new_min_x = np.random.randint(
            low=minval_x, high=maxval_x, dtype=np.int32)
    else:
        unclipped_new_min_y, unclipped_new_min_x = (
            clip_y(new_min_bbox_coords[0]), clip_x(new_min_bbox_coords[1]))
    unclipped_new_max_y = unclipped_new_min_y + bbox_height - 1
    unclipped_new_max_x = unclipped_new_min_x + bbox_width - 1

    # Determine if any of the new bbox was shifted outside the current image.
    # This is used for determining if any of the original bbox content should be
    # discarded.
    new_min_y, new_min_x, new_max_y, new_max_x = (
        clip_y(unclipped_new_min_y), clip_x(unclipped_new_min_x),
        clip_y(unclipped_new_max_y), clip_x(unclipped_new_max_x))
    shifted_min_y = (new_min_y - unclipped_new_min_y) + min_y
    shifted_max_y = max_y - (unclipped_new_max_y - new_max_y)
    shifted_min_x = (new_min_x - unclipped_new_min_x) + min_x
    shifted_max_x = max_x - (unclipped_new_max_x - new_max_x)

    # Create the new bbox tensor by converting pixel integer values to floats.
    new_bbox = np.stack([
        float(new_min_y) / float(image_height), float(new_min_x) /
        float(image_width), float(new_max_y) / float(image_height),
        float(new_max_x) / float(image_width)
    ])

    # Copy the contents in the bbox and fill the old bbox location
    # with gray (128).
    bbox_content = image[shifted_min_y:shifted_max_y + 1, shifted_min_x:
                         shifted_max_x + 1, :]

    def mask_and_add_image(min_y_, min_x_, max_y_, max_x_, mask, content_tensor,
                           image_):
        """Applies mask to bbox region in image then adds content_tensor to it."""
        mask = np.pad(mask, [[min_y_, (image_height - 1) - max_y_],
                             [min_x_, (image_width - 1) - max_x_], [0, 0]],
                      'constant',
                      constant_values=1)

        content_tensor = np.pad(content_tensor,
                                [[min_y_, (image_height - 1) - max_y_],
                                 [min_x_, (image_width - 1) - max_x_], [0, 0]],
                                'constant',
                                constant_values=0)
        return image_ * mask + content_tensor

    # Zero out original bbox location.
    mask = np.zeros_like(image)[min_y:max_y + 1, min_x:max_x + 1, :]
    grey_tensor = np.zeros_like(mask) + replace[0]
    image = mask_and_add_image(min_y, min_x, max_y, max_x, mask, grey_tensor,
                               image)

    # Fill in bbox content to new bbox location.
    mask = np.zeros_like(bbox_content)
    image = mask_and_add_image(new_min_y, new_min_x, new_max_y, new_max_x, mask,
                               bbox_content, image)

    return image.astype(np.uint8), new_bbox


def _clip_bbox(min_y, min_x, max_y, max_x):
    """
    裁剪边界框坐标在0和1之间。

    Args:
        min_y: 归一化的边界框y轴最小值，类型为浮点数，范围在0到1之间。
        min_x: 归一化的边界框x轴最小值，类型为浮点数，范围在0到1之间。
        max_y: 归一化的边界框y轴最大值，类型为浮点数，范围在0到1之间。
        max_x: 归一化的边界框x轴最大值，类型为浮点数，范围在0到1之间。

    Returns:
        返回裁剪后的坐标值，范围在0和1之间。
    """
    min_y = np.clip(min_y, a_min=0, a_max=1.0)
    min_x = np.clip(min_x, a_min=0, a_max=1.0)
    max_y = np.clip(max_y, a_min=0, a_max=1.0)
    max_x = np.clip(max_x, a_min=0, a_max=1.0)
    return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    """
    调整边界框坐标以确保面积大于0。

    Args:
        min_y: 归一化的边界框坐标，类型为浮点数，范围在0到1之间。
        min_x: 归一化的边界框坐标，类型为浮点数，范围在0到1之间。
        max_y: 归一化的边界框坐标，类型为浮点数，范围在0到1之间。
        max_x: 归一化的边界框坐标，类型为浮点数，范围在0到1之间。
        delta: 浮点数，用于在边界框的最小/最大坐标相同的情况下创建一个大小为2 * delta的间隙，
            这可以防止边界框的面积为零。

    Returns:
        返回一个新的边界框坐标元组，范围在0到1之间，并且保证面积大于0。
    """
    height = max_y - min_y
    width = max_x - min_x

    def _adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = np.maximum(max_coord, 0.0 + delta)
        min_coord = np.minimum(min_coord, 1.0 - delta)
        return min_coord, max_coord

    if _equal(height, 0):
        min_y, max_y = _adjust_bbox_boundaries(min_y, max_y)

    if _equal(width, 0):
        min_x, max_x = _adjust_bbox_boundaries(min_x, max_x)

    return min_y, min_x, max_y, max_x


def _scale_bbox_only_op_probability(prob):
    """Reduce the probability of the bbox-only operation.

    Probability is reduced so that we do not distort the content of too many
    bounding boxes that are close to each other. The value of 3.0 was a chosen
    hyper parameter when designing the autoaugment algorithm that we found
    empirically to work well.

    Args:
        prob: Float that is the probability of applying the bbox-only operation.

    Returns:
        Reduced probability.
    """
    return prob / 3.0


def _apply_bbox_augmentation(image, bbox, augmentation_func, *args):
    """
    对由bbox指定的图像子部分应用augmentation_func。

    Args:
        image (3D uint8 Tensor): 输入图像。
        bbox (1D Tensor): 包含4个元素(min_y, min_x, max_y, max_x)的1D张量，
            这些元素是类型为float的归一化坐标，表示范围在0到1之间的值。
        augmentation_func (function): 将应用于图像子部分的增强函数。
        *args: 传递给augmentation_func的其他参数。

    Returns:
        modified image (3D uint8 Tensor): 返回修改后的图像，
            其中图像中bbox位置将应用augmentation_func。
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    min_y = int(image_height * bbox[0])
    min_x = int(image_width * bbox[1])
    max_y = int(image_height * bbox[2])
    max_x = int(image_width * bbox[3])

    # Clip to be sure the max values do not fall out of range.
    max_y = np.minimum(max_y, image_height - 1)
    max_x = np.minimum(max_x, image_width - 1)

    # Get the sub-tensor that is the image within the bounding box region.
    bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

    # Apply the augmentation function to the bbox portion of the image.
    augmented_bbox_content = augmentation_func(bbox_content, *args)

    # Pad the augmented_bbox_content and the mask to match the shape of original
    # image.
    augmented_bbox_content = np.pad(
        augmented_bbox_content, [[min_y, (image_height - 1) - max_y],
                                 [min_x, (image_width - 1) - max_x], [0, 0]],
        'constant',
        constant_values=1)

    # Create a mask that will be used to zero out a part of the original image.
    mask_tensor = np.zeros_like(bbox_content)

    mask_tensor = np.pad(mask_tensor,
                         [[min_y, (image_height - 1) - max_y],
                          [min_x, (image_width - 1) - max_x], [0, 0]],
                         'constant',
                         constant_values=1)
    # Replace the old bbox content with the new augmented content.
    image = image * mask_tensor + augmented_bbox_content
    return image.astype(np.uint8)


def _concat_bbox(bbox, bboxes):
    """Helper function that concates bbox to bboxes along the first dimension."""

    # Note if all elements in bboxes are -1 (_INVALID_BOX), then this means
    # we discard bboxes and start the bboxes Tensor with the current bbox.
    bboxes_sum_check = np.sum(bboxes)
    bbox = np.expand_dims(bbox, 0)
    # This check will be true when it is an _INVALID_BOX
    if _equal(bboxes_sum_check, -4):
        bboxes = bbox
    else:
        bboxes = np.concatenate([bboxes, bbox], 0)
    return bboxes


def _apply_bbox_augmentation_wrapper(image, bbox, new_bboxes, prob,
                                     augmentation_func, func_changes_bbox,
                                     *args):
    """
    以概率prob应用_apply_bbox_augmentation。

    Args:
        image: 3D uint8 Tensor。
        bbox: 1D Tensor，包含4个元素(min_y, min_x, max_y, max_x)，类型为float，表示归一化坐标，取值范围在0到1之间。
        new_bboxes: 2D Tensor，是图像中bbox的列表，这些bbox在aug_func修改后生成。当func_changes_bbox设置为true时，这些bbox才会被修改。每个bbox包含4个元素(min_y, min_x, max_y, max_x)，类型为float，表示归一化的bbox坐标，取值范围在0到1之间。
        prob: Float类型，表示应用_apply_bbox_augmentation的概率。
        augmentation_func: 将应用于图像子部分的增强函数。
        func_changes_bbox: Boolean类型。augmentation_func是否除了返回图像外还返回bbox。
        *args: 当调用augmentation_func时，将传递的其他参数。

    Returns:
        一个元组。第一个元素是图像的修改版本，其中如果以概率prob选择了调用bbox位置的图像，则会对其应用augmentation_func。
        第二个元素是一个长度为4的Tensor列表，包含应用augmentation_func后修改的bbox。
    """

    should_apply_op = (np.random.rand() + prob >= 1)
    if func_changes_bbox:
        if should_apply_op:
            augmented_image, bbox = augmentation_func(image, bbox, *args)
        else:
            augmented_image, bbox = (image, bbox)
    else:
        if should_apply_op:
            augmented_image = _apply_bbox_augmentation(image, bbox,
                                                       augmentation_func, *args)
        else:
            augmented_image = image
    new_bboxes = _concat_bbox(bbox, new_bboxes)
    return augmented_image.astype(np.uint8), new_bboxes


def _apply_multi_bbox_augmentation(image, bboxes, prob, aug_func,
                                   func_changes_bbox, *args):
    """
    对图像中的每个边界框应用 aug_func。

    Args:
        image: 3D uint8 Tensor。
        bboxes: 包含图像中边界框的 2D Tensor 列表。每个边界框有 4 个元素（min_y, min_x, max_y, max_x），类型为 float。
        prob: 浮点数，表示对图像中特定边界框应用 aug_func 的概率。
        aug_func: 增强函数，将应用于由 bboxes 中的边界框值指示的图像子部分。
        func_changes_bbox: 布尔值。增强函数是否除了图像外还返回边界框。
        *args: 调用增强函数时将传递的其他参数。

    Returns:
        图像的修改版本，其中图像中的每个边界框位置都将被应用增强函数（如果它以概率 prob 被独立调用）。
        还返回最终的边界框，如果 func_changes_bbox 设置为 false，则它们将保持不变；如果为 true，则将返回新的修改后的边界框。
    """
    # Will keep track of the new altered bboxes after aug_func is repeatedly
    # applied. The -1 values are a dummy value and this first Tensor will be
    # removed upon appending the first real bbox.
    new_bboxes = np.array(_INVALID_BOX)

    # If the bboxes are empty, then just give it _INVALID_BOX. The result
    # will be thrown away.
    bboxes = np.array((_INVALID_BOX)) if bboxes.size == 0 else bboxes

    assert bboxes.shape[1] == 4, "bboxes.shape[1] must be 4!!!!"

    # pylint:disable=g-long-lambda
    # pylint:disable=line-too-long
    wrapped_aug_func = lambda _image, bbox, _new_bboxes: _apply_bbox_augmentation_wrapper(_image, bbox, _new_bboxes, prob, aug_func, func_changes_bbox, *args)
    # pylint:enable=g-long-lambda
    # pylint:enable=line-too-long

    # Setup the while_loop.
    num_bboxes = bboxes.shape[0]  # We loop until we go over all bboxes.
    idx = 0  # Counter for the while loop.

    # Conditional function when to end the loop once we go over all bboxes
    # images_and_bboxes contain (_image, _new_bboxes)
    def cond(_idx, _images_and_bboxes):
        return _idx < num_bboxes

    # Shuffle the bboxes so that the augmentation order is not deterministic if
    # we are not changing the bboxes with aug_func.
    # if not func_changes_bbox:
    #     print(bboxes)
    #     loop_bboxes = np.take(bboxes,np.random.permutation(bboxes.shape[0]),axis=0)
    #     print(loop_bboxes)
    # else:
    #     loop_bboxes = bboxes
    # we can not shuffle the bbox because it does not contain class information here
    loop_bboxes = deepcopy(bboxes)

    # Main function of while_loop where we repeatedly apply augmentation on the
    # bboxes in the image.
    # pylint:disable=g-long-lambda
    body = lambda _idx, _images_and_bboxes: [
            _idx + 1, wrapped_aug_func(_images_and_bboxes[0],
                                         loop_bboxes[_idx],
                                         _images_and_bboxes[1])]
    while (cond(idx, (image, new_bboxes))):
        idx, (image, new_bboxes) = body(idx, (image, new_bboxes))

    # Either return the altered bboxes or the original ones depending on if
    # we altered them in anyway.
    if func_changes_bbox:
        final_bboxes = new_bboxes
    else:
        final_bboxes = bboxes
    return image, final_bboxes


def _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, aug_func,
                                           func_changes_bbox, *args):
    """Checks to be sure num bboxes > 0 before calling inner function."""
    num_bboxes = len(bboxes)
    new_image = deepcopy(image)
    new_bboxes = deepcopy(bboxes)
    if num_bboxes != 0:
        new_image, new_bboxes = _apply_multi_bbox_augmentation(
            new_image, new_bboxes, prob, aug_func, func_changes_bbox, *args)
    return new_image, new_bboxes


def rotate_only_bboxes(image, bboxes, prob, degrees, replace):
    """Apply rotate to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, rotate, func_changes_bbox, degrees, replace)


def shear_x_only_bboxes(image, bboxes, prob, level, replace):
    """Apply shear_x to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, shear_x, func_changes_bbox, level, replace)


def shear_y_only_bboxes(image, bboxes, prob, level, replace):
    """Apply shear_y to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, shear_y, func_changes_bbox, level, replace)


def translate_x_only_bboxes(image, bboxes, prob, pixels, replace):
    """Apply translate_x to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, translate_x, func_changes_bbox, pixels, replace)


def translate_y_only_bboxes(image, bboxes, prob, pixels, replace):
    """Apply translate_y to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, translate_y, func_changes_bbox, pixels, replace)


def flip_only_bboxes(image, bboxes, prob):
    """Apply flip_lr to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob,
                                                  np.fliplr, func_changes_bbox)


def solarize_only_bboxes(image, bboxes, prob, threshold):
    """Apply solarize to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, solarize,
                                                  func_changes_bbox, threshold)


def equalize_only_bboxes(image, bboxes, prob):
    """Apply equalize to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, equalize,
                                                  func_changes_bbox)


def cutout_only_bboxes(image, bboxes, prob, pad_size, replace):
    """Apply cutout to each bbox in the image with probability prob."""
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, cutout, func_changes_bbox, pad_size, replace)


def _rotate_bbox(bbox, image_height, image_width, degrees):
    """
    旋转边界框坐标。

    Args:
        bbox: 1D张量，包含4个元素（min_y, min_x, max_y, max_x），类型为浮点数，
            表示归一化坐标，范围在0到1之间。
        image_height: 整数，图像的高度。
        image_width: 整数，图像的宽度。
        degrees: 浮点数，旋转角度，单位为度。如果为正数，则顺时针旋转图像；
            如果为负数，则逆时针旋转图像。

    Returns:
        与bbox形状相同的张量，但包含旋转后的坐标。
    """
    image_height, image_width = (float(image_height), float(image_width))

    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # Translate the bbox to the center of the image and turn the normalized 0-1
    # coordinates to absolute pixel locations.
    # Y coordinates are made negative as the y axis of images goes down with
    # increasing pixel values, so we negate to make sure x axis and y axis points
    # are in the traditionally positive direction.
    min_y = -int(image_height * (bbox[0] - 0.5))
    min_x = int(image_width * (bbox[1] - 0.5))
    max_y = -int(image_height * (bbox[2] - 0.5))
    max_x = int(image_width * (bbox[3] - 0.5))
    coordinates = np.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x],
                            [max_y, max_x]]).astype(np.float32)
    # Rotate the coordinates according to the rotation matrix clockwise if
    # radians is positive, else negative
    rotation_matrix = np.stack([[math.cos(radians), math.sin(radians)],
                                [-math.sin(radians), math.cos(radians)]])
    new_coords = np.matmul(rotation_matrix,
                           np.transpose(coordinates)).astype(np.int32)

    # Find min/max values and convert them back to normalized 0-1 floats.
    min_y = -(float(np.max(new_coords[0, :])) / image_height - 0.5)
    min_x = float(np.min(new_coords[1, :])) / image_width + 0.5
    max_y = -(float(np.min(new_coords[0, :])) / image_height - 0.5)
    max_x = float(np.max(new_coords[1, :])) / image_width + 0.5

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return np.stack([min_y, min_x, max_y, max_x])


def rotate_with_bboxes(image, bboxes, degrees, replace):
    """
    对图像进行旋转，并调整边界框的位置。

    Args:
        image (numpy.ndarray): 待旋转的图像。
        bboxes (numpy.ndarray): 边界框的坐标，形状为 (n, 4)，其中 n 是边界框的数量，每个边界框由四个坐标 (x_min, y_min, x_max, y_max) 表示。
        degrees (float): 旋转的角度，以度为单位。正值表示顺时针旋转，负值表示逆时针旋转。
        replace (bool): 是否用背景色填充旋转后的图像空白区域。

    Returns:
        tuple: 包含旋转后的图像和新的边界框坐标。

            - image (numpy.ndarray): 旋转后的图像。
            - new_bboxes (numpy.ndarray): 旋转后的边界框坐标，形状与输入相同。

    """
    # Rotate the image.
    image = rotate(image, degrees, replace)

    # Convert bbox coordinates to pixel values.
    image_height, image_width = image.shape[:2]
    # pylint:disable=g-long-lambda
    wrapped_rotate_bbox = lambda bbox: _rotate_bbox(bbox, image_height, image_width, degrees)
    # pylint:enable=g-long-lambda
    new_bboxes = np.zeros_like(bboxes)
    for idx in range(len(bboxes)):
        new_bboxes[idx] = wrapped_rotate_bbox(bboxes[idx])
    return image, new_bboxes


def translate_x(image, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
    return unwrap(np.array(image), replace)


def translate_y(image, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))
    return unwrap(np.array(image), replace)


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
    """
    将bbox的坐标移动指定的像素数。

    Args:
        bbox: 一个长度为4的1D Tensor，包含四个元素 (min_y, min_x, max_y, max_x)，
            类型为float，表示归一化后的坐标，范围在0到1之间。
        image_height: Int，图像的高度。
        image_width: Int，图像的宽度。
        pixels: Int，要移动bbox的像素数。
        shift_horizontal: Boolean，如果为True则在X维度上移动，否则在Y维度上移动。

    Returns:
        一个与bbox形状相同的tensor，但现在包含移动后的坐标。
    """
    pixels = int(pixels)
    # Convert bbox to integer pixel locations.
    min_y = int(float(image_height) * bbox[0])
    min_x = int(float(image_width) * bbox[1])
    max_y = int(float(image_height) * bbox[2])
    max_x = int(float(image_width) * bbox[3])

    if shift_horizontal:
        min_x = np.maximum(0, min_x - pixels)
        max_x = np.minimum(image_width, max_x - pixels)
    else:
        min_y = np.maximum(0, min_y - pixels)
        max_y = np.minimum(image_height, max_y - pixels)

    # Convert bbox back to floats.
    min_y = float(min_y) / float(image_height)
    min_x = float(min_x) / float(image_width)
    max_y = float(max_y) / float(image_height)
    max_x = float(max_x) / float(image_width)

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return np.stack([min_y, min_x, max_y, max_x])


def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
    """
    将图像和边界框在X/Y维度上平移的等价函数。

    Args:
        image: 3D uint8 Tensor，表示图像。
        bboxes: 2D Tensor，包含图像中所有边界框的列表。每个边界框有4个元素
            (min_y, min_x, max_y, max_x)，类型为float，值在[0, 1]之间。
        pixels: int，图像和边界框平移的像素数。
        replace: 一个一维张量，值为1，用于填充空像素。
        shift_horizontal: bool，如果为True则在X维度上平移，否则在Y维度上平移。

    Returns:
        返回一个元组，包含平移后的3D uint8 Tensor图像，以及平移后的边界框。
        平移后的图像和边界框坐标将反映图像的平移。
    """
    if shift_horizontal:
        image = translate_x(image, pixels, replace)
    else:
        image = translate_y(image, pixels, replace)

    # Convert bbox coordinates to pixel values.
    image_height, image_width = image.shape[0], image.shape[1]
    # pylint:disable=g-long-lambda
    wrapped_shift_bbox = lambda bbox: _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal)
    # pylint:enable=g-long-lambda
    new_bboxes = deepcopy(bboxes)
    num_bboxes = len(bboxes)
    for idx in range(num_bboxes):
        new_bboxes[idx] = wrapped_shift_bbox(bboxes[idx])
    return image.astype(np.uint8), new_bboxes


def shear_x(image, level, replace):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1    level
    #    0    1].
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0))
    return unwrap(np.array(image), replace)


def shear_y(image, level, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1    0
    #    level    1].
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, 0, 0, level, 1, 0))
    return unwrap(np.array(image), replace)


def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
    """
    根据图像剪切程度调整边界框。

    Args:
        bbox: 长度为4的1D张量，包含四个元素（min_y, min_x, max_y, max_x），类型为float，表示归一化坐标，范围在0到1之间。
        image_height: 图像高度，类型为int。
        image_width: 图像宽度，类型为int。
        level: 浮点数，表示图像的剪切程度。
        shear_horizontal: 如果为True，则在X维进行剪切；否则在Y维进行剪切。

    Returns:
        返回与bbox形状相同的张量，但坐标已调整。
    """
    image_height, image_width = (float(image_height), float(image_width))

    # Change bbox coordinates to be pixels.
    min_y = int(image_height * bbox[0])
    min_x = int(image_width * bbox[1])
    max_y = int(image_height * bbox[2])
    max_x = int(image_width * bbox[3])
    coordinates = np.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
    coordinates = coordinates.astype(np.float32)

    # Shear the coordinates according to the translation matrix.
    if shear_horizontal:
        translation_matrix = np.stack([[1, 0], [-level, 1]])
    else:
        translation_matrix = np.stack([[1, -level], [0, 1]])
    translation_matrix = translation_matrix.astype(np.float32)
    new_coords = np.matmul(translation_matrix,
                           np.transpose(coordinates)).astype(np.int32)

    # Find min/max values and convert them back to floats.
    min_y = float(np.min(new_coords[0, :])) / image_height
    min_x = float(np.min(new_coords[1, :])) / image_width
    max_y = float(np.max(new_coords[0, :])) / image_height
    max_x = float(np.max(new_coords[1, :])) / image_width

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return np.stack([min_y, min_x, max_y, max_x])


def shear_with_bboxes(image, bboxes, level, replace, shear_horizontal):
    """
    对图像应用剪切变换并移动边界框。

    Args:
        image: 3D uint8 Tensor。
        bboxes: 2D Tensor，是图像中边界框的列表。每个边界框
            有4个元素（min_y, min_x, max_y, max_x），类型为float，值在[0, 1]之间。
        level: Float。图像剪切的程度。该值将在-0.3到0.3之间。
        replace: 一个或一个三值的1D tensor，用于填充空白像素。
        shear_horizontal: Boolean。如果为True，则在X维度上进行剪切；否则在Y维度上进行剪切。

    Returns:
        一个包含3D uint8 Tensor的元组，该Tensor是图像按level剪切后的结果。元组的第二个元素是bboxes，
        其中坐标将发生移动以反映剪切后的图像。
    """
    if shear_horizontal:
        image = shear_x(image, level, replace)
    else:
        image = shear_y(image, level, replace)

    # Convert bbox coordinates to pixel values.
    image_height, image_width = image.shape[:2]
    # pylint:disable=g-long-lambda
    wrapped_shear_bbox = lambda bbox: _shear_bbox(bbox, image_height, image_width, level, shear_horizontal)
    # pylint:enable=g-long-lambda
    new_bboxes = deepcopy(bboxes)
    num_bboxes = len(bboxes)
    for idx in range(num_bboxes):
        new_bboxes[idx] = wrapped_shear_bbox(bboxes[idx])
    return image.astype(np.uint8), new_bboxes


def autocontrast(image):
    """
    对图像应用自动对比度调整。

    Args:
        image (numpy.ndarray): 一个3D的uint8张量，表示图像。

    Returns:
        numpy.ndarray: 自动对比度调整后的图像，类型为uint8。

    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = float(np.min(image))
        hi = float(np.max(image))

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            img = np.clip(im, a_min=0, a_max=255.0)
            return im.astype(np.uint8)

        result = scale_values(image) if hi > lo else image
        return result

    # Assumes RGB for now.    Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = np.stack([s1, s2, s3], 2)
    return image


def sharpness(image, factor):
    """Implements Sharpness function from PIL."""
    orig_image = image
    image = image.astype(np.float32)
    # Make image 4D for conv operation.
    # SMOOTH PIL Kernel.
    kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.
    result = cv2.filter2D(image, -1, kernel).astype(np.uint8)

    # Blend the final result.
    return blend(result, orig_image, factor)


def equalize(image):
    """
    对图像进行直方图均衡化处理。

    Args:
        image (np.ndarray): 输入的图像数据，形状为 (height, width, channels)，数据类型应为 np.uint8。

    Returns:
        np.ndarray: 直方图均衡化处理后的图像数据，形状与输入图像相同，数据类型为 np.uint8。

    """

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c].astype(np.int32)
        # Compute the histogram of the image channel.
        histo, _ = np.histogram(im, range=[0, 255], bins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = np.where(np.not_equal(histo, 0))
        nonzero_histo = np.reshape(np.take(histo, nonzero), [-1])
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
            # Clip the counts to be in range.    This is done
            # in the C code for image.point.
            return np.clip(lut, a_min=0, a_max=255).astype(np.uint8)

        # If step is zero, return the original image.    Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            result = np.take(build_lut(histo, step), im)

        return result.astype(np.uint8)

    # Assumes RGB for now.    Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = np.stack([s1, s2, s3], 2)
    return image


def wrap(image):
    """Returns 'image' with an extra channel set to all 1s."""
    shape = image.shape
    extended_channel = 255 * np.ones([shape[0], shape[1], 1], image.dtype)
    extended = np.concatenate([image, extended_channel], 2).astype(image.dtype)
    return extended


def unwrap(image, replace):
    """
    展开由wrap函数生成的图像。

    对于每个空间位置的最后一个通道中的0，该空间维度中的其余三个通道将变灰（设置为128）。
    对包裹的张量执行平移和剪切等操作会在空白位置留下0。
    一些转换会根据值的强度进行预处理，我们希望这些空白像素采用“平均值”，而不是纯黑色。

    Args:
        image: 一个具有4个通道的3D图像张量。
        replace: 一个用于填充空白像素的一维张量，其值为一或三个。

    Returns:
        image: 一个具有3个通道的3D图像张量。
    """
    image_shape = image.shape
    # Flatten the spatial dimensions.
    flattened_image = np.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = flattened_image[:, 3]

    replace = np.concatenate([replace, np.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    alpha_channel = np.reshape(alpha_channel, (-1, 1))
    alpha_channel = np.tile(alpha_channel, reps=(1, flattened_image.shape[1]))

    flattened_image = np.where(
        np.equal(alpha_channel, 0),
        np.ones_like(
            flattened_image, dtype=image.dtype) * replace,
        flattened_image)

    image = np.reshape(flattened_image, image_shape)
    image = image[:, :, :3]
    return image.astype(np.uint8)


def _cutout_inside_bbox(image, bbox, pad_fraction):
    """
    在边界框内生成裁剪掩码和边界框内像素的平均值。

    首先，在图像中随机选择一个位置作为中心，将在该位置应用裁剪掩码。请注意，该位置可能位于图像边界附近，
    因此可能不会应用完整的裁剪掩码。

    Args:
        image: 3D uint8 Tensor。
        bbox: 具有4个元素（min_y, min_x, max_y, max_x）的1D Tensor，类型为float，
            表示在0和1之间的归一化坐标。
        pad_fraction: 指定裁剪掩码相对于原始边界框大小的大小的浮点数。
            如果pad_fraction为0.25，则裁剪掩码的形状将为
            (0.25 * bbox高度, 0.25 * bbox宽度)。

    Returns:
        一个元组。第一个元素是与图像形状相同的tensor，其中每个元素为1或0，
        用于确定在哪里对图像应用裁剪。第二个元素是边界框所在图像区域的像素平均值。
        掩码值: [0,1]
    """

    image_height, image_width = image.shape[0], image.shape[1]
    # Transform from shape [1, 4] to [4].
    bbox = np.squeeze(bbox)

    min_y = int(float(image_height) * bbox[0])
    min_x = int(float(image_width) * bbox[1])
    max_y = int(float(image_height) * bbox[2])
    max_x = int(float(image_width) * bbox[3])

    # Calculate the mean pixel values in the bounding box, which will be used
    # to fill the cutout region.
    mean = np.mean(image[min_y:max_y + 1, min_x:max_x + 1], axis=(0, 1))
    # Cutout mask will be size pad_size_heigh * 2 by pad_size_width * 2 if the
    # region lies entirely within the bbox.
    box_height = max_y - min_y + 1
    box_width = max_x - min_x + 1
    pad_size_height = int(pad_fraction * (box_height / 2))
    pad_size_width = int(pad_fraction * (box_width / 2))

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = np.random.randint(min_y, max_y + 1, dtype=np.int32)
    cutout_center_width = np.random.randint(min_x, max_x + 1, dtype=np.int32)

    lower_pad = np.maximum(0, cutout_center_height - pad_size_height)
    upper_pad = np.maximum(
        0, image_height - cutout_center_height - pad_size_height)
    left_pad = np.maximum(0, cutout_center_width - pad_size_width)
    right_pad = np.maximum(0,
                           image_width - cutout_center_width - pad_size_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

    mask = np.pad(np.zeros(
        cutout_shape, dtype=image.dtype),
                  padding_dims,
                  'constant',
                  constant_values=1)

    mask = np.expand_dims(mask, 2)
    mask = np.tile(mask, [1, 1, 3])
    return mask, mean


def bbox_cutout(image, bboxes, pad_fraction, replace_with_mean):
    """
    根据bbox信息对图像应用cutout。

    这是cutout的一种变体，它使用bbox信息来做出更明智的决策，
    以决定在哪里放置cutout掩码。

    Args:
        image: 3D uint8 Tensor。
        bboxes: 2D Tensor，图像中所有bbox的列表。每个bbox有4个元素
            (min_y, min_x, max_y, max_x)，类型为float，值在[0, 1]之间。
        pad_fraction: Float，指定cutout掩码的大小相对于原始bbox大小的比例。
            如果pad_fraction为0.25，则cutout掩码的形状将为
            (0.25 * bbox高度, 0.25 * bbox宽度)。
        replace_with_mean: Boolean，指定在应用cutout掩码时应填充什么值。
            由于传入的图像将是uint8类型，并且不会应用任何均值归一化，
            因此默认情况下，我们将值设置为128。如果replace_with_mean为True，
            则我们找到跨通道维度的平均像素值，并使用这些值来填充应用cutout掩码的位置。

    Returns:
        一个元组。第一个元素是与图像相同形状的tensor，已应用cutout。
        第二个元素是传入的不变的bboxes。
    """

    def apply_bbox_cutout(image, bboxes, pad_fraction):
        """Applies cutout to a single bounding box within image."""
        # Choose a single bounding box to apply cutout to.
        random_index = np.random.randint(0, bboxes.shape[0], dtype=np.int32)
        # Select the corresponding bbox and apply cutout.
        chosen_bbox = np.take(bboxes, random_index, axis=0)
        mask, mean = _cutout_inside_bbox(image, chosen_bbox, pad_fraction)

        # When applying cutout we either set the pixel value to 128 or to the mean
        # value inside the bbox.
        replace = mean if replace_with_mean else [128] * 3

        # Apply the cutout mask to the image. Where the mask is 0 we fill it with
        # `replace`.
        image = np.where(
            np.equal(mask, 0),
            np.ones_like(
                image, dtype=image.dtype) * replace,
            image).astype(image.dtype)
        return image

    # Check to see if there are boxes, if so then apply boxcutout.
    if len(bboxes) != 0:
        image = apply_bbox_cutout(image, bboxes, pad_fraction)

    return image, bboxes


NAME_TO_FUNC = {
        'AutoContrast': autocontrast,
        'Equalize': equalize,
        'Posterize': posterize,
        'Solarize': solarize,
        'SolarizeAdd': solarize_add,
        'Color': color,
        'Contrast': contrast,
        'Brightness': brightness,
        'Sharpness': sharpness,
        'Cutout': cutout,
        'BBox_Cutout': bbox_cutout,
        'Rotate_BBox': rotate_with_bboxes,
        # pylint:disable=g-long-lambda
        'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
                image, bboxes, pixels, replace, shift_horizontal=True),
        'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
                image, bboxes, pixels, replace, shift_horizontal=False),
        'ShearX_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
                image, bboxes, level, replace, shear_horizontal=True),
        'ShearY_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
                image, bboxes, level, replace, shear_horizontal=False),
        # pylint:enable=g-long-lambda
        'Rotate_Only_BBoxes': rotate_only_bboxes,
        'ShearX_Only_BBoxes': shear_x_only_bboxes,
        'ShearY_Only_BBoxes': shear_y_only_bboxes,
        'TranslateX_Only_BBoxes': translate_x_only_bboxes,
        'TranslateY_Only_BBoxes': translate_y_only_bboxes,
        'Flip_Only_BBoxes': flip_only_bboxes,
        'Solarize_Only_BBoxes': solarize_only_bboxes,
        'Equalize_Only_BBoxes': equalize_only_bboxes,
        'Cutout_Only_BBoxes': cutout_only_bboxes,
}


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = np.floor(np.random.rand() + 0.5) >= 1
    final_tensor = tensor if should_flip else -tensor
    return final_tensor


def _rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return (level, )


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0, )  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return (level, )


def _enhance_level_to_arg(level):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1, )


def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level, )


def _translate_level_to_arg(level, translate_const):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level, )


def _bbox_cutout_level_to_arg(level, hparams):
    cutout_pad_fraction = (level /
                           _MAX_LEVEL) * 0.75  # hparams.cutout_max_pad_fraction
    return (cutout_pad_fraction, False)  # hparams.cutout_bbox_replace_with_mean


def level_to_arg(hparams):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Posterize': lambda level: (int((level / _MAX_LEVEL) * 4), ),
        'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256), ),
        'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110), ),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'Cutout':
        lambda level: (int((level / _MAX_LEVEL) * 100), ),  # hparams.cutout_const=100
        # pylint:disable=g-long-lambda
        'BBox_Cutout': lambda level: _bbox_cutout_level_to_arg(level, hparams),
        'TranslateX_BBox':
        lambda level: _translate_level_to_arg(level, 250),  # hparams.translate_const=250
        'TranslateY_BBox':
        lambda level: _translate_level_to_arg(level, 250),  # hparams.translate_cons
        # pylint:enable=g-long-lambda
        'ShearX_BBox': _shear_level_to_arg,
        'ShearY_BBox': _shear_level_to_arg,
        'Rotate_BBox': _rotate_level_to_arg,
        'Rotate_Only_BBoxes': _rotate_level_to_arg,
        'ShearX_Only_BBoxes': _shear_level_to_arg,
        'ShearY_Only_BBoxes': _shear_level_to_arg,
        # pylint:disable=g-long-lambda
        'TranslateX_Only_BBoxes':
        lambda level: _translate_level_to_arg(level, 120),  # hparams.translate_bbox_const
        'TranslateY_Only_BBoxes':
        lambda level: _translate_level_to_arg(level, 120),  # hparams.translate_bbox_const
        # pylint:enable=g-long-lambda
        'Flip_Only_BBoxes': lambda level: (),
        'Solarize_Only_BBoxes':
        lambda level: (int((level / _MAX_LEVEL) * 256), ),
        'Equalize_Only_BBoxes': lambda level: (),
        # pylint:disable=g-long-lambda
        'Cutout_Only_BBoxes':
        lambda level: (int((level / _MAX_LEVEL) * 50), ),  # hparams.cutout_bbox_const
        # pylint:enable=g-long-lambda
    }


def bbox_wrapper(func):
    """Adds a bboxes function argument to func and returns unchanged bboxes."""

    def wrapper(images, bboxes, *args, **kwargs):
        return (func(images, *args, **kwargs), bboxes)

    return wrapper


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(augmentation_hparams)[name](level)

    # Check to see if prob is passed into function. This is used for operations
    # where we alter bboxes independently.
    # pytype:disable=wrong-arg-types
    if 'prob' in inspect.getfullargspec(func)[0]:
        args = tuple([prob] + list(args))
    # pytype:enable=wrong-arg-types

    # Add in replace arg if it is required for the function that is being called.
    if 'replace' in inspect.getfullargspec(func)[0]:
        # Make sure replace is the final argument
        assert 'replace' == inspect.getfullargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])

    # Add bboxes as the second positional argument for the function if it does
    # not already exist.
    if 'bboxes' not in inspect.getfullargspec(func)[0]:
        func = bbox_wrapper(func)
    return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob, bboxes):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)
    assert 'bboxes' == inspect.getfullargspec(func)[0][1]

    # If prob is a function argument, then this randomness is being handled
    # inside the function, so make sure it is always called.
    if 'prob' in inspect.getfullargspec(func)[0]:
        prob = 1.0

    # Apply the function with probability `prob`.
    should_apply_op = np.floor(np.random.rand() + 0.5) >= 1
    if should_apply_op:
        augmented_image, augmented_bboxes = func(image, bboxes, *args)
    else:
        augmented_image, augmented_bboxes = (image, bboxes)
    return augmented_image, augmented_bboxes


def select_and_apply_random_policy(policies, image, bboxes):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = np.random.randint(0, len(policies), dtype=np.int32)
    # policy_to_select = 6 # for test
    for (i, policy) in enumerate(policies):
        if i == policy_to_select:
            image, bboxes = policy(image, bboxes)
    return (image, bboxes)


def build_and_apply_nas_policy(policies, image, bboxes, augmentation_hparams):
    """Build a policy from the given policies passed in and apply to image.

    Args:
        policies: list of lists of tuples in the form `(func, prob, level)`, `func`
            is a string name of the augmentation function, `prob` is the probability
            of applying the `func` operation, `level` is the input argument for
            `func`.
        image: numpy array that the resulting policy will be applied to.
        bboxes:
        augmentation_hparams: Hparams associated with the NAS learned policy.

    Returns:
        A version of image that now has data augmentation applied to it based on
        the `policies` pass into the function. Additionally, returns bboxes if
        a value for them is passed in that is not None
    """
    replace_value = [128, 128, 128]

    # func is the string name of the augmentation function, prob is the
    # probability of applying the operation and level is the parameter associated

    # tf_policies are functions that take in an image and return an augmented
    # image.
    tf_policies = []
    for policy in policies:
        tf_policy = []
        # Link string name to the correct python function and make sure the correct
        # argument is passed into that function.
        for policy_info in policy:
            policy_info = list(
                policy_info) + [replace_value, augmentation_hparams]

            tf_policy.append(_parse_policy_info(*policy_info))
        # Now build the tf policy that will apply the augmentation procedue
        # on image.
        def make_final_policy(tf_policy_):
            def final_policy(image_, bboxes_):
                for func, prob, args in tf_policy_:
                    image_, bboxes_ = _apply_func_with_prob(func, image_, args,
                                                            prob, bboxes_)
                return image_, bboxes_

            return final_policy

        tf_policies.append(make_final_policy(tf_policy))

    augmented_images, augmented_bboxes = select_and_apply_random_policy(
        tf_policies, image, bboxes)
    # If no bounding boxes were specified, then just return the images.
    return (augmented_images, augmented_bboxes)


# TODO(barretzoph): Add in ArXiv link once paper is out.
def distort_image_with_autoaugment(image, bboxes, augmentation_name):
    """Applies the AutoAugment policy to `image` and `bboxes`.

    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        bboxes: `Tensor` of shape [N, 4] representing ground truth boxes that are
            normalized between [0, 1].
        augmentation_name: The name of the AutoAugment policy to use. The available
            options are `v0`, `v1`, `v2`, `v3` and `test`. `v0` is the policy used for
            all of the results in the paper and was found to achieve the best results
            on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
            found on the COCO dataset that have slight variation in what operations
            were used during the search procedure along with how many operations are
            applied in parallel to a single image (2 vs 3).

    Returns:
        A tuple containing the augmented versions of `image` and `bboxes`.
    """
    available_policies = {
        'v0': policy_v0,
        'v1': policy_v1,
        'v2': policy_v2,
        'v3': policy_v3,
        'test': policy_vtest
    }
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(
            augmentation_name))

    policy = available_policies[augmentation_name]()
    augmentation_hparams = {}
    return build_and_apply_nas_policy(policy, image, bboxes,
                                      augmentation_hparams)