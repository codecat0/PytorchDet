#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :ops.py
@Author :CodeCat
@Date   :2025/11/12 19:12
"""
import torch
import torch.nn.functional as F


def prior_box(input_tensor,
              image_tensor,
              min_sizes,
              max_sizes=None,
              aspect_ratios=[1.],
              variance=[0.1, 0.1, 0.2, 0.2],
              flip=False,
              clip=False,
              steps=[0.0, 0.0],
              offset=0.5,
              min_max_aspect_ratios_order=False,
              name=None):  # name is ignored in PyTorch
    """
    Generates prior boxes for SSD algorithm in PyTorch.

    Args:
        input_tensor (Tensor): 4-D tensor [B, C, H, W], the feature map.
        image_tensor (Tensor): 4-D tensor [B, C, H_orig, W_orig], the original image.
        min_sizes (list[float]): List of min sizes for anchor boxes.
        max_sizes (list[float], optional): List of max sizes for anchor boxes. Defaults to None.
        aspect_ratios (list[float], optional): List of aspect ratios. Defaults to [1.].
        variance (list[float], optional): List of variance values. Defaults to [0.1, 0.1, 0.2, 0.2].
        flip (bool, optional): Whether to flip aspect ratios. Defaults to False.
        clip (bool, optional): Whether to clip boxes to image boundaries. Defaults to False.
        steps (list[float], optional): Steps for anchor boxes in [step_w, step_h].
                                       If [0, 0], it's calculated based on input and image size. Defaults to [0., 0.].
        offset (float, optional): Center offset for anchor boxes. Defaults to 0.5.
        min_max_aspect_ratios_order (bool, optional): Order of anchor generation. Defaults to False.
        name (str, optional): Ignored in PyTorch.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - boxes (Tensor): Shape [H, W, num_priors, 4] with [x_min, y_min, x_max, y_max].
            - variances (Tensor): Shape [H, W, num_priors, 4] with variance values.
    """
    # --- Get dimensions ---
    batch_size, _, feat_h, feat_w = input_tensor.shape
    _, _, img_h, img_w = image_tensor.shape

    # --- Handle steps ---
    if steps[0] == 0.0 or steps[1] == 0.0:
        step_h = img_h / feat_h
        step_w = img_w / feat_w
    else:
        step_h = steps[1]
        step_w = steps[0]

    # --- Prepare aspect ratios ---
    ar_list = []
    for ar in aspect_ratios:
        ar_list.append(ar)
        if flip:
            ar_list.append(1.0 / ar)
    # Remove duplicates (e.g., if ar=1.0) while preserving order
    ar_list = list(dict.fromkeys(ar_list))

    # --- Prepare sizes ---
    min_sizes_list = min_sizes if isinstance(min_sizes, (list, tuple)) else [min_sizes]
    max_sizes_list = max_sizes if max_sizes is not None and isinstance(max_sizes, (list, tuple)) else (
        [max_sizes] if max_sizes is not None else [])
    variance_tensor = torch.tensor(variance, dtype=input_tensor.dtype, device=input_tensor.device).view(1, 1, 1,
                                                                                                        -1)  # [1, 1, 1, 4]

    # --- Calculate total number of priors per location ---
    num_priors_per_location = 0
    if not min_max_aspect_ratios_order:
        # Standard order: aspect ratio boxes first, then min/max size boxes
        num_priors_per_location += len(ar_list) * len(min_sizes_list)
        # Add boxes for min_sizes (usually square)
        num_priors_per_location += len(min_sizes_list)
        # Add boxes for max_sizes (usually square, calculated as sqrt(min * max))
        for min_s, max_s in zip(min_sizes_list, max_sizes_list):
            if max_s is not None:
                num_priors_per_location += 1
    else:
        # Caffe order: min/max size boxes first, then aspect ratio boxes
        # Add boxes for min_sizes (square)
        num_priors_per_location += len(min_sizes_list) * len(ar_list)
        # Add boxes for max_sizes (square, sqrt(min * max))
        for min_s, max_s in zip(min_sizes_list, max_sizes_list):
            if max_s is not None:
                num_priors_per_location += 1
        # Add boxes for aspect ratios
        num_priors_per_location += len(ar_list) * 2  # 2 for each aspect ratio

    # --- Generate grid of anchor centers ---
    # Create coordinate vectors for feature map grid centers in the original image space
    dtype = input_tensor.dtype
    device = input_tensor.device

    # Centers are offset from the top-left corner of each grid cell
    grid_cx = (torch.arange(0, feat_w, dtype=dtype, device=device) + offset) * step_w  # [W]
    grid_cy = (torch.arange(0, feat_h, dtype=dtype, device=device) + offset) * step_h  # [H]
    # Meshgrid to create H x W coordinate matrices
    grid_y, grid_x = torch.meshgrid(grid_cy, grid_cx, indexing='ij')  # [H, W], [H, W]

    # --- Calculate anchor dimensions based on sizes and aspect ratios ---
    # This part calculates all possible w, h combinations for all priors at a *single* location
    # Then we will broadcast these across all H x W locations.
    anchor_dims = []
    if not min_max_aspect_ratios_order:
        # Order: aspect ratio boxes first
        for ar in ar_list:
            for min_size in min_sizes_list:
                w = min_size * torch.sqrt(torch.tensor(ar, dtype=dtype, device=device))
                h = min_size / torch.sqrt(torch.tensor(ar, dtype=dtype, device=device))
                anchor_dims.append((w, h))
        # Min size boxes (square)
        for min_size in min_sizes_list:
            anchor_dims.append((torch.tensor(min_size, dtype=dtype, device=device),
                                torch.tensor(min_size, dtype=dtype, device=device)))
        # Max size boxes (square, sqrt(min*max))
        for min_size, max_size in zip(min_sizes_list, max_sizes_list):
            if max_size is not None:
                sqrt_prod = torch.sqrt(torch.tensor(min_size * max_size, dtype=dtype, device=device))
                anchor_dims.append((sqrt_prod, sqrt_prod))
    else:  # min_max_aspect_ratios_order = True
        # Min size boxes (square)
        for min_size in min_sizes_list:
            anchor_dims.append((torch.tensor(min_size, dtype=dtype, device=device),
                                torch.tensor(min_size, dtype=dtype, device=device)))
        # Max size boxes (square, sqrt(min*max))
        for min_size, max_size in zip(min_sizes_list, max_sizes_list):
            if max_size is not None:
                sqrt_prod = torch.sqrt(torch.tensor(min_size * max_size, dtype=dtype, device=device))
                anchor_dims.append((sqrt_prod, sqrt_prod))
        # Aspect ratio boxes
        for ar in ar_list:
            for min_size in min_sizes_list:
                w = min_size * torch.sqrt(torch.tensor(ar, dtype=dtype, device=device))
                h = min_size / torch.sqrt(torch.tensor(ar, dtype=dtype, device=device))
                anchor_dims.append((w, h))

    # Check if calculated number of dims matches expected num_priors_per_location
    assert len(
        anchor_dims) == num_priors_per_location, f"Calculated {len(anchor_dims)} priors, expected {num_priors_per_location}. Check parameters."

    # Unpack dimensions
    anchor_w = torch.tensor([dim[0] for dim in anchor_dims], dtype=dtype, device=device)  # [num_priors]
    anchor_h = torch.tensor([dim[1] for dim in anchor_dims], dtype=dtype, device=device)  # [num_priors]

    # --- Broadcast centers and dimensions across H, W, and num_priors ---
    # Expand grid coordinates to [H, W, 1]
    grid_x_expanded = grid_x.unsqueeze(-1)  # [H, W, 1]
    grid_y_expanded = grid_y.unsqueeze(-1)  # [H, W, 1]
    # Expand anchor dimensions to [1, 1, num_priors]
    anchor_w_expanded = anchor_w.view(1, 1, -1)  # [1, 1, num_priors]
    anchor_h_expanded = anchor_h.view(1, 1, -1)  # [1, 1, num_priors]

    # Calculate corner coordinates
    x_min = grid_x_expanded - anchor_w_expanded / 2.0  # [H, W, num_priors]
    y_min = grid_y_expanded - anchor_h_expanded / 2.0  # [H, W, num_priors]
    x_max = grid_x_expanded + anchor_w_expanded / 2.0  # [H, W, num_priors]
    y_max = grid_y_expanded + anchor_h_expanded / 2.0  # [H, W, num_priors]

    # Stack to get boxes [H, W, num_priors, 4]
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    # Expand variances to match boxes shape [H, W, num_priors, 4]
    variances = variance_tensor.expand(feat_h, feat_w, num_priors_per_location, 4)

    # --- Clip boxes if requested ---
    if clip:
        # Clip to image boundaries [0, img_w] x [0, img_h]
        boxes = torch.clamp(boxes, min=0.0,
                            max=torch.tensor([img_w, img_h, img_w, img_h], dtype=dtype, device=device).view(1, 1, 1,
                                                                                                            -1))

    return boxes, variances
