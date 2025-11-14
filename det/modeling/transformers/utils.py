#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :utils.py
@Author :CodeCat
@Date   :2025/11/10 17:06
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bbox_utils import bbox_overlaps

__all__ = [
    '_get_clones', 'bbox_overlaps', 'bbox_cxcywh_to_xyxy',
    'bbox_xyxy_to_cxcywh', 'sigmoid_focal_loss', 'inverse_sigmoid',
    'deformable_attention_core_func', 'varifocal_loss_with_logits'
]


def _get_clones(module, N):
    """
    Create N clones of the given module using nn.ModuleList and copy.deepcopy.

    Args:
        module (nn.Module): The module to be cloned.
        N (int): The number of clones to create.

    Returns:
        nn.ModuleList: A list-like container holding N independent copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def bbox_cxcywh_to_xyxy(x):
    """
    Convert bounding box coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        x (torch.Tensor): Shape [..., 4], where the last dimension contains [cx, cy, w, h].

    Returns:
        torch.Tensor: Shape [..., 4], where the last dimension contains [x1, y1, x2, y2].
    """
    cxcy, wh = torch.split(x, 2, dim=-1)
    x1y1 = cxcy - 0.5 * wh
    x2y2 = cxcy + 0.5 * wh
    return torch.cat([x1y1, x2y2], dim=-1)


def bbox_xyxy_to_cxcywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        x (torch.Tensor): Shape [..., 4], where the last dimension contains [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Shape [..., 4], where the last dimension contains [cx, cy, w, h].
    """
    x1y1, x2y2 = torch.split(x, 2, dim=-1)
    cx_cy = (x1y1 + x2y2) / 2.0
    w_h = x2y2 - x1y1
    return torch.cat([cx_cy, w_h], dim=-1)


def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    """
    Calculate the focal loss for binary classification using sigmoid activation.

    Args:
        logit (torch.Tensor): Predicted logits, shape [N, ...].
        label (torch.Tensor): Ground truth labels, same shape as logit.
        normalizer (float): Normalization factor for the loss. Defaults to 1.0.
        alpha (float): Alpha factor for focal loss. Defaults to 0.25.
        gamma (float): Gamma factor for focal loss. Defaults to 2.0.

    Returns:
        torch.Tensor: Scalar focal loss value.
    """
    prob = F.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    focal_weight = (1 - p_t) ** gamma
    loss = ce_loss * focal_weight

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss

    mean_loss = loss.mean(dim=1)
    sum_loss = mean_loss.sum()
    return sum_loss / normalizer

def inverse_sigmoid(x, eps=1e-5):
    """
    Calculate the inverse sigmoid (logit) of x.

    Args:
        x (torch.Tensor): Input tensor, values should be in the range [0, 1].
        eps (float): Small value to ensure numerical stability by preventing log(0).

    Returns:
        torch.Tensor: The inverse sigmoid (logit) of x.
    """
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes,
                                   value_level_start_index, sampling_locations,
                                   attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2] (height, width for each level)
        value_level_start_index (Tensor|List): [n_levels] (start index for each level in value_length dim)
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    # Calculate split sizes for torch.split
    split_shape = [h * w for h, w in value_spatial_shapes]
    # Split value tensor along dim=1 based on spatial shapes
    value_list = torch.split(value, split_shape, dim=1)
    # Scale sampling locations from [0, 1] to [-1, 1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # Process value for current level
        # [N, H_*W_, M, D] -> [N, H_*W_, M*D] -> [N, M*D, H_*W] -> [N*M, D, H, W]
        value_l_ = value_list[level].flatten(2).transpose(0, 2).transpose(0, 1).reshape(
            bs * n_head, c, h, w) # Use transpose or permute as needed

        # Process sampling grids for current level
        # [N, Lq, M, P, 2] -> [N, M, Lq, P, 2] -> [N*M, Lq, P, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3).flatten(0, 1) # Use permute for multi-dim transpose

        # Sample value using grid_sample
        # [N*M, D, Lq, P]
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # Process attention weights
    # [N, Lq, M, L, P] -> [N, M, Lq, L, P] -> [N*M, 1, Lq, L*P]
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    # Stack sampled values, flatten last two dims, multiply with weights, sum over last dim, reshape
    # Stacked shape: [N*M, D, Lq, L, P] -> [N*M, D, Lq, L*P] -> [N*M, 1, Lq, L*P] (after mul) -> [N*M, D, Lq] (after sum)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(dim=-1).reshape(bs, n_head * c, Len_q)

    # Transpose final output: [N, C_out, Lq] -> [N, Lq, C_out]
    return output.permute(0, 2, 1)


def discrete_sample(x, grid):
    """
    Args:
        x (Tensor): [N, C, H, W]
        grid (Tensor): [N, grid_H, grid_W, 2] (grid coordinates, typically in [0, 1] or [0, spatial_size-1] format)
    Returns:
        output (Tensor): [N, C, grid_H, grid_W]
    """
    N, C, H, W = x.shape
    _, grid_H, grid_W, _ = grid.shape
    # Create spatial shape tensor
    spatial_shape = torch.tensor([[W, H]], dtype=torch.float32, device=x.device) # Ensure same device as x
    # Scale grid coordinates, add 0.5, convert to int64, then flatten
    # Assuming grid is in [0, 1] format, multiply by [W, H] to get pixel coordinates
    index = (grid * spatial_shape + 0.5).to(torch.int64).flatten(1, 2) # Shape: [N, grid_H * grid_W, 2]
    # Extract and clamp h and w indices
    h_index = index[:, :, 1].clamp(0, H - 1) # Shape: [N, grid_H * grid_W]
    w_index = index[:, :, 0].clamp(0, W - 1) # Shape: [N, grid_H * grid_W]
    # Create batch indices
    batch_index = torch.arange(N, device=x.device).unsqueeze(-1).repeat([1, grid_H * grid_W]) # Shape: [N, grid_H * grid_W]
    # Use advanced indexing to sample from x
    # x: [N, C, H, W], batch_index: [N, grid_H*grid_W], h_index: [N, grid_H*grid_W], w_index: [N, grid_H*grid_W]
    # Resulting shape: [N, grid_H*grid_W, C]
    output = x[batch_index, :, h_index, w_index]
    # Reshape and transpose to get [N, C, grid_H, grid_W]
    # Current shape: [N, grid_H*grid_W, C] -> permute to [N, C, grid_H*grid_W] -> reshape to [N, C, grid_H, grid_W]
    output = output.permute(0, 2, 1).reshape(N, C, grid_H, grid_W)
    return output


def deformable_attention_core_func_v2(value,
                                      value_spatial_shapes,
                                      sampling_locations,
                                      attention_weights,
                                      num_points_list,
                                      sampling_method='default'):
    """
    Args:
        value (Tensor): [batch_num, value_len, num_heads, head_dim]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        sampling_locations (Tensor): [batch_num, query_len, num_heads, total_num_points, 2]
        attention_weights (Tensor): [batch_num, query_len, num_heads, total_num_points]
        num_points_list (List): The number of sampling point corresponding to each level
        sampling_method (str): default(grid_sample) or discrete(discrete_sample)

    Returns:
        output (Tensor): [batch_num, query_len, num_heads * head_dim]
    """
    assert sampling_method in ['default', 'discrete'], NotImplementedError
    batch_num, _, num_heads, head_dim = value.shape
    query_len = sampling_locations.shape[1]
    num_levels = len(num_points_list)

    # Transpose and flatten value tensor
    # [N, Len_v, M, D] -> [N, M, D, Len_v] -> [N*M, D, Len_v]
    value = value.permute(0, 2, 3, 1).flatten(0, 1)
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = torch.split(value, split_shape, dim=-1)
    value_list = [
        value.reshape(batch_num * num_heads, head_dim, h, w) # Reshape each split part to [N*M, D, H, W]
        for value, (h, w) in zip(value_list, value_spatial_shapes)
    ]

    if sampling_method == 'default':
        sampling_grids = 2 * sampling_locations - 1
    else:
        sampling_grids = sampling_locations

    # Transpose and flatten sampling_grids
    # [N, Lq, M, total_P, 2] -> [N, M, Lq, total_P, 2] -> [N*M, Lq, total_P, 2]
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    # Split sampling_grids based on num_points_list
    # Each part corresponds to one level
    sampling_grids_list = torch.split(sampling_grids, num_points_list, dim=-2)

    sampling_value_list = []
    for idx in range(num_levels):
        # value_list[idx]: [N*M, D, H, W]
        # sampling_grids_list[idx]: [N*M, Lq, P_level, 2]
        # _sampling_value: [N*M, D, Lq, P_level]
        if sampling_method == 'default':
            _sampling_value = F.grid_sample(value_list[idx],
                                            sampling_grids_list[idx],
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=False)
        else:
            # Assuming discrete_sample is also converted to PyTorch
            # Input: value_list[idx] [N*M, D, H, W], sampling_grids_list[idx] [N*M, Lq, P_level, 2]
            # Output: [N*M, D, Lq, P_level] (assuming discrete_sample handles the grid shape correctly)
            _sampling_value = discrete_sample(value_list[idx],
                                              sampling_grids_list[idx])
        sampling_value_list.append(_sampling_value)

    # Process attention weights
    # [N, Lq, M, total_P] -> [N, M, Lq, total_P] -> [N*M, Lq, total_P] -> [N*M, 1, Lq, total_P]
    attn_weights = attention_weights.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
    # Concatenate sampling values along the last dimension (P_level -> total_P)
    # sampling_value_list elements: [N*M, D, Lq, P_level_i] -> concatenated: [N*M, D, Lq, total_P]
    sampling_value = torch.cat(sampling_value_list, dim=-1)
    # Weighted sum: [N*M, D, Lq, total_P] * [N*M, 1, Lq, total_P] -> [N*M, D, Lq, total_P] -> [N*M, D, Lq]
    # Sum along the last dimension (total_P)
    output = (sampling_value * attn_weights).sum(dim=-1)
    # Reshape output: [N*M, D, Lq] -> [N, M*D, Lq]
    output = output.reshape(batch_num, num_heads * head_dim, query_len)
    # Transpose final output: [N, M*D, Lq] -> [N, Lq, M*D]
    return output.permute(0, 2, 1)


def get_valid_ratio(mask):
    """
    Calculate the ratio of valid (True/1) pixels along the height and width dimensions.

    Args:
        mask (torch.Tensor): A boolean or binary tensor of shape [batch_size, H, W],
                             where True/1 indicates valid pixels.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, 2], where the last dimension
                      contains [valid_ratio_width, valid_ratio_height].
    """
    _, H, W = mask.shape
    valid_ratio_h = torch.sum(mask[:, :, 0], dim=1) / H
    valid_ratio_w = torch.sum(mask[:, 0, :], dim=1) / W
    return torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)


def get_denoising_training_group(targets,
                                 num_classes,
                                 num_queries,
                                 class_embed,
                                 num_denoising=100,
                                 label_noise_ratio=0.5,
                                 box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None

    # Calculate number of ground truths per sample
    num_gts = [len(t) for t in targets["gt_class"]]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    # Calculate number of groups
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    # Pad ground truths to max number
    bs = len(targets["gt_class"])
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=targets["gt_class"][0].device) # Ensure same device
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=targets["gt_bbox"][0].device) # Ensure same device
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=targets["gt_class"][0].device) # Ensure same device

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
            pad_gt_mask[i, :num_gt] = True # Use True for valid, False for padded

    # Repeat for num_group times
    input_query_class = input_query_class.repeat([1, num_group]) # [bs, max_gt_num * num_group]
    input_query_bbox = input_query_bbox.repeat([1, num_group, 1]) # [bs, max_gt_num * num_group, 4]
    pad_gt_mask = pad_gt_mask.repeat([1, num_group]) # [bs, max_gt_num * num_group]

    # Get indices for non-padded ground truths (positive indices for denoising)
    # torch.nonzero returns [num_valid, 2] tensor, [:, 1] gets the column indices (within repeated sequence)
    dn_positive_idx = torch.nonzero(pad_gt_mask)[:, 1] # Shape: [total_valid_gts_across_batch_and_groups]
    # Split the indices back based on original num_gts and num_group
    # Create split sizes: [n1*num_group, n2*num_group, ...]
    split_sizes = [n * num_group for n in num_gts]
    dn_positive_idx = torch.split(dn_positive_idx, split_sizes) # List of tensors

    # Total denoising queries
    num_denoising = int(max_gt_num * num_group)

    if label_noise_ratio > 0:
        # Flatten for noise application
        flat_class = input_query_class.flatten() # Shape: [bs * num_denoising]
        flat_mask = pad_gt_mask.flatten() # Shape: [bs * num_denoising]

        # Generate random mask for label noise
        # Note: PaddlePaddle's original logic: rand < (label_noise_ratio * 0.5)
        # This creates a boolean mask. We then cast to float32 in PaddlePaddle.
        # In PyTorch, we can work directly with the boolean mask for nonzero,
        # but the original intent might be to apply noise to half of the selected masked elements.
        # Let's re-interpret: noise_prob = label_noise_ratio * 0.5
        # So, mask elements are True with prob label_noise_ratio * 0.5, AND they must be valid (pad_gt_mask is True).
        noise_mask = torch.rand(flat_class.shape, device=flat_class.device) < (label_noise_ratio * 0.5)
        # Apply the pad mask: only apply noise to valid elements
        final_noise_mask = noise_mask & flat_mask # Element-wise AND

        chosen_idx = torch.nonzero(final_noise_mask).squeeze(-1) # Get linear indices where noise should be applied
        if chosen_idx.numel() > 0: # Check if any indices were selected
            # Generate new random labels for chosen indices
            new_label = torch.randint(0, num_classes, chosen_idx.shape, dtype=flat_class.dtype, device=flat_class.device)
            # Apply new labels using advanced indexing (scatter_ equivalent)
            flat_class.scatter_(0, chosen_idx, new_label) # In-place update on the flattened tensor

        # Reshape back
        input_query_class = flat_class.reshape([bs, num_denoising])
        pad_gt_mask = flat_mask.reshape([bs, num_denoising])

    if box_noise_scale > 0:
        # Calculate noise range based on bbox dimensions
        # diff = [w*0.5*scale, h*scale] repeated to [w*0.5*scale, h*scale, w*scale, h*scale]
        diff = torch.cat(
            [input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]], # [..., 2:] are w, h
            dim=-1) * box_noise_scale # Shape: [bs, num_denoising, 4]
        # Generate random noise in [-1, 1]
        random_noise = torch.rand(input_query_bbox.shape, device=input_query_bbox.device) * 2.0 - 1.0
        diff *= random_noise
        input_query_bbox += diff
        # Apply inverse sigmoid
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # Extend class embedding with a zero vector for the 'no object' class (num_classes)
    # class_embed shape: [num_classes, embed_dim] -> [num_classes + 1, embed_dim]
    extended_class_embed = torch.cat(
        [class_embed, torch.zeros([1, class_embed.shape[-1]], dtype=class_embed.dtype, device=class_embed.device)], dim=0)
    # Gather embeddings based on input_query_class indices
    # input_query_class: [bs, num_denoising] -> flattened -> indices for extended_class_embed
    # torch.gather requires index tensor to have same number of dimensions as source tensor
    # extended_class_embed: [num_classes + 1, embed_dim]
    # input_query_class.flatten(): [bs * num_denoising]
    # Need to expand indices to match the source tensor's last dim
    flat_indices = input_query_class.flatten().unsqueeze(-1).expand(-1, class_embed.shape[-1]) # [bs * num_denoising, embed_dim]
    # Gather along dim 0 (class dimension)
    gathered_embeds = torch.gather(extended_class_embed, 0, flat_indices) # [bs * num_denoising, embed_dim]
    # Reshape back to [bs, num_denoising, embed_dim]
    input_query_class = gathered_embeds.reshape([bs, num_denoising, -1])

    # Create attention mask
    tgt_size = num_denoising + num_queries
    # Create a full mask of False (PyTorch uses True for *masking out*, False for *allowing attention*)
    # So, ones < 0 creates a False tensor.
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=class_embed.device) # Initially all False (allow attention)

    # match query (from num_denoising onwards) cannot see the reconstruction (first num_denoising)
    attn_mask[num_denoising:, :num_denoising] = True # Block attention

    # reconstruct cannot see each other within groups
    for i in range(num_group):
        start_idx = max_gt_num * i
        end_idx = max_gt_num * (i + 1)
        # Mask out attention *from* current group *to* other groups
        # To other groups later in the sequence
        if i < num_group - 1:
            attn_mask[start_idx:end_idx, max_gt_num * (i + 1):num_denoising] = True
        # To other groups earlier in the sequence
        if i > 0:
            attn_mask[start_idx:end_idx, :start_idx] = True

    attn_mask = torch.logical_not(attn_mask)

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None

    # Calculate number of ground truths per sample
    # listcomp is not well-supported in SOT mode for now.
    num_gts = []
    for t in targets["gt_class"]:
        num_gts.append(len(t))
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    # Calculate number of groups
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    # Pad ground truths to max number
    bs = len(targets["gt_class"])
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=targets["gt_class"][0].device) # Ensure same device
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=targets["gt_bbox"][0].device) # Ensure same device
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=targets["gt_class"][0].device) # Ensure same device

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
            pad_gt_mask[i, :num_gt] = True # Use True for valid, False for padded

    # each group has positive and negative queries. Repeat 2 * num_group times.
    input_query_class = input_query_class.repeat([1, 2 * num_group]) # [bs, max_gt_num * 2 * num_group]
    input_query_bbox = input_query_bbox.repeat([1, 2 * num_group, 1]) # [bs, max_gt_num * 2 * num_group, 4]
    pad_gt_mask = pad_gt_mask.repeat([1, 2 * num_group]) # [bs, max_gt_num * 2 * num_group]

    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], dtype=pad_gt_mask.dtype, device=pad_gt_mask.device) # [bs, max_gt_num * 2, 1]
    negative_gt_mask[:, max_gt_num:] = 1 # Second half (negative) is 1
    negative_gt_mask = negative_gt_mask.repeat([1, num_group, 1]) # [bs, max_gt_num * 2 * num_group, 1]
    positive_gt_mask = 1 - negative_gt_mask # [bs, max_gt_num * 2 * num_group, 1]

    # contrastive denoising training positive index
    # Squeeze the last dimension of positive_gt_mask and multiply with pad_gt_mask
    positive_gt_mask_squeezed = positive_gt_mask.squeeze(-1) # [bs, max_gt_num * 2 * num_group]
    positive_gt_mask_final = positive_gt_mask_squeezed * pad_gt_mask # Element-wise multiplication [bs, max_gt_num * 2 * num_group]
    # Get indices where positive_gt_mask_final is True and pad_gt_mask is True
    dn_positive_idx = torch.nonzero(positive_gt_mask_final)[:, 1] # Shape: [total_positive_gts_across_batch_and_groups]
    # Split the indices back based on original num_gts and num_group
    # Create split sizes: [n1*num_group, n2*num_group, ...] (each sample contributes n * num_group positive indices)
    split_sizes = [n * num_group for n in num_gts]
    dn_positive_idx = torch.split(dn_positive_idx, split_sizes) # List of tensors

    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        # Flatten for noise application
        flat_class = input_query_class.flatten() # Shape: [bs * num_denoising]
        flat_mask = pad_gt_mask.flatten() # Shape: [bs * num_denoising]

        # Generate random mask for label noise
        # Note: PaddlePaddle's original logic: rand < (label_noise_ratio * 0.5)
        noise_mask = torch.rand(flat_class.shape, device=flat_class.device) < (label_noise_ratio * 0.5)
        # Apply the pad mask: only apply noise to valid elements
        final_noise_mask = noise_mask & flat_mask # Element-wise AND

        chosen_idx = torch.nonzero(final_noise_mask).squeeze(-1) # Get linear indices where noise should be applied
        if chosen_idx.numel() > 0: # Check if any indices were selected
            # Generate new random labels for chosen indices
            new_label = torch.randint(0, num_classes, chosen_idx.shape, dtype=flat_class.dtype, device=flat_class.device)
            # Apply new labels using advanced indexing (scatter_ equivalent)
            flat_class.scatter_(0, chosen_idx, new_label) # In-place update on the flattened tensor

        # Reshape back
        input_query_class = flat_class.reshape([bs, num_denoising])
        pad_gt_mask = flat_mask.reshape([bs, num_denoising])

    if box_noise_scale > 0:
        # Convert bbox format
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox) # [bs, num_denoising, 4]

        # Calculate noise range based on bbox dimensions (w, h)
        # diff = [w*0.5*scale, h*0.5*scale, w*0.5*scale, h*0.5*scale] (repeated)
        diff = input_query_bbox[..., 2:].repeat([1, 1, 2]) * 0.5 * box_noise_scale # [bs, num_denoising, 4]

        # Generate random sign: 1 or -1
        rand_sign = torch.randint(0, 2, input_query_bbox.shape, device=input_query_bbox.device).float() * 2.0 - 1.0 # [bs, num_denoising, 4]

        # Generate random part [0, 1], then modify based on negative_gt_mask
        rand_part = torch.rand(input_query_bbox.shape, device=input_query_bbox.device) # [bs, num_denoising, 4]
        # Expand negative_gt_mask to match rand_part's last dim (4) for broadcasting
        neg_mask_expanded = negative_gt_mask.expand_as(rand_part) # [bs, num_denoising, 4]
        # If negative query (mask is 1), rand_part becomes rand_part + 1.0, else stays rand_part
        rand_part = (rand_part + 1.0) * neg_mask_expanded + rand_part * (1.0 - neg_mask_expanded)
        # Multiply by sign
        rand_part *= rand_sign

        # Apply noise
        known_bbox += rand_part * diff
        # Clamp bbox coordinates to [0, 1]
        known_bbox = known_bbox.clamp(min=0.0, max=1.0) # Use clamp instead of clamp_
        # Convert bbox format back
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        # Apply inverse sigmoid
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # Extend class embedding with a zero vector for the 'no object' class (num_classes)
    # class_embed shape: [num_classes, embed_dim] -> [num_classes + 1, embed_dim]
    extended_class_embed = torch.cat(
        [class_embed, torch.zeros([1, class_embed.shape[-1]], dtype=class_embed.dtype, device=class_embed.device)], dim=0)
    # Gather embeddings based on input_query_class indices
    # input_query_class: [bs, num_denoising] -> flattened -> indices for extended_class_embed
    flat_indices = input_query_class.flatten().unsqueeze(-1).expand(-1, class_embed.shape[-1]) # [bs * num_denoising, embed_dim]
    # Gather along dim 0 (class dimension)
    gathered_embeds = torch.gather(extended_class_embed, 0, flat_indices) # [bs * num_denoising, embed_dim]
    # Reshape back to [bs, num_denoising, embed_dim]
    input_query_class = gathered_embeds.reshape([bs, num_denoising, -1])

    # Create attention mask
    tgt_size = num_denoising + num_queries
    # Create a full mask of False (PyTorch uses True for *masking out*, False for *allowing attention*)
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=class_embed.device) # Initially all False (allow attention)

    # match query (from num_denoising onwards) cannot see the reconstruction (first num_denoising)
    attn_mask[num_denoising:, :num_denoising] = True # Block attention

    # reconstruct cannot see each other within groups (each group now has 2 * max_gt_num elements)
    for i in range(num_group):
        start_idx = max_gt_num * 2 * i
        end_idx = max_gt_num * 2 * (i + 1)
        # Mask out attention *from* current group *to* other groups
        # To other groups later in the sequence
        if i < num_group - 1:
            attn_mask[start_idx:end_idx, max_gt_num * 2 * (i + 1):num_denoising] = True
        # To other groups earlier in the sequence
        if i > 0:
            attn_mask[start_idx:end_idx, :start_idx] = True

    # PyTorch attention mask uses True to *mask out* (i.e., set to -inf before softmax)
    # Invert the mask so True means "mask out", False means "attend"
    attn_mask = torch.logical_not(attn_mask)

    dn_meta = {
        "dn_positive_idx": dn_positive_idx, # List of tensors containing positive indices per batch sample
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta

def get_sine_pos_embed(pos_tensor,
                       num_pos_feats=128,
                       temperature=10000,
                       exchange_xy=True):
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (Tensor): Shape as `(None, n)`. e.g., [batch_size, num_queries, n_coords] or [num_queries, n_coords]
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        Tensor: Returned position embedding  # noqa
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2. * math.pi
    # Calculate the dimension index tensor for sine/cosine
    # dim_t corresponds to the 2i or 2i+1 in the original transformer paper formula
    dim_t = 2. * (torch.arange(num_pos_feats, device=pos_tensor.device) // 2) # Shape: [num_pos_feats]
    # Calculate the scaling factor for each dimension
    dim_t = scale / (temperature**(dim_t / num_pos_feats)) # Shape: [num_pos_feats]

    def sine_func(x):
        """
        Applies the sine/cosine embedding to a single coordinate tensor x.
        x shape: [batch_size, num_queries, 1] or [num_queries, 1] (after split)
        """
        x_scaled = x * dim_t # Broadcasting: [batch_size, num_queries, 1] * [num_pos_feats] -> [batch_size, num_queries, num_pos_feats]
        stacked = torch.stack((x_scaled[:, :, 0::2].sin(), x_scaled[:, :, 1::2].cos()), dim=3) # Shape: [B, N, F_even, 2] or [N, F_even, 2]
        flattened = stacked.flatten(2) # Shape: [B, N, min_len * 2]
        return flattened

    pos_tensor_split = torch.split(pos_tensor, 1, dim=-1)
    pos_res = [sine_func(x) for x in pos_tensor_split]

    if exchange_xy and len(pos_res) >= 2:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]

    pos_res = torch.cat(pos_res, dim=2)
    return pos_res


def mask_to_box_coordinate(mask,
                           normalize=False,
                           format="xyxy",
                           dtype=torch.float32): # Use torch dtype
    """
    Compute the bounding boxes around the provided mask.
    Args:
        mask (Tensor:bool): [b, c, h, w]

    Returns:
        bbox (Tensor): [b, c, 4]
    """
    assert mask.ndim == 4
    assert format in ["xyxy", "xywh"]

    h, w = mask.shape[-2:]
    # Generate coordinate grids
    y_coords, x_coords = torch.meshgrid(torch.arange(h, dtype=dtype, device=mask.device),
                                        torch.arange(w, dtype=dtype, device=mask.device),
                                        indexing='ij') # Use indexing='ij' to match PaddlePaddle behavior
    # x_coords shape: [h, w], y_coords shape: [h, w]

    # Calculate x_min, x_max
    x_mask = x_coords * mask.to(x_coords.dtype)
    x_max_vals = x_mask.flatten(-2).max(dim=-1)[0] + 1
    x_min_vals = torch.where(mask.to(torch.bool), x_mask, torch.tensor(1e8, dtype=x_coords.dtype, device=mask.device)).flatten(-2).min(dim=-1)[0]

    # Calculate y_min, y_max
    y_mask = y_coords * mask.to(y_coords.dtype)
    y_max_vals = y_mask.flatten(-2).max(dim=-1)[0] + 1
    y_min_vals = torch.where(mask.to(torch.bool), y_mask, torch.tensor(1e8, dtype=y_coords.dtype, device=mask.device)).flatten(-2).min(dim=-1)[0]

    # Stack coordinates into bbox format [x_min, y_min, x_max, y_max]
    out_bbox = torch.stack([x_min_vals, y_min_vals, x_max_vals, y_max_vals], dim=-1)

    # Create a mask indicating which channels have any True values (valid masks)
    valid_mask = mask.any(dim=[2, 3]).unsqueeze(2).to(out_bbox.dtype)
    # Zero out bbox coordinates where the mask was entirely False
    out_bbox = out_bbox * valid_mask

    if normalize:
        # Normalize by width (w) and height (h)
        # out_bbox: [b, c, 4], normalization tensor: [4]
        normalization_tensor = torch.tensor([w, h, w, h], dtype=dtype, device=mask.device)
        out_bbox = out_bbox / normalization_tensor

    if format == "xyxy":
        return out_bbox
    else:
        return bbox_xyxy_to_cxcywh(out_bbox)

def varifocal_loss_with_logits(pred_logits,
                               gt_score,
                               label,
                               normalizer=1.0,
                               alpha=0.75,
                               gamma=2.0):
    """
    Calculate the varifocal loss for binary classification using sigmoid activation.

    Args:
        pred_logits (torch.Tensor): Predicted logits, shape [N, ...].
        gt_score (torch.Tensor): Ground truth scores (e.g., IoU), same shape as pred_logits.
        label (torch.Tensor): Ground truth labels (0 or 1), same shape as pred_logits.
        normalizer (float): Normalization factor for the loss. Defaults to 1.0.
        alpha (float): Alpha factor for varifocal loss. Defaults to 0.75.
        gamma (float): Gamma factor for varifocal loss. Defaults to 2.0.

    Returns:
        torch.Tensor: Scalar varifocal loss value.
    """
    pred_score = F.sigmoid(pred_logits)
    # Calculate the weight for the loss
    # weight = alpha * (pred_score)^gamma * (1 - label) + gt_score * label
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

    loss = F.binary_cross_entropy_with_logits(pred_logits, gt_score, weight=weight, reduction='none')
    mean_loss = loss.mean(1)
    sum_loss = mean_loss.sum()
    return sum_loss / normalizer