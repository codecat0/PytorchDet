#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :post_process.py
@Author :CodeCat
@Date   :2025/11/10 17:02
"""
from collections import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from det.modeling.bbox_utils import nonempty_bbox
from .transformers.utils import bbox_cxcywh_to_xyxy


class BBoxPostProcess(object):
    def __init__(self,
                 num_classes=80,
                 decode=None,
                 nms=None,
                 export_onnx=False,
                 export_eb=False):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.export_onnx = export_onnx
        self.export_eb = export_eb
        self.origin_shape_list = None

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed.

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
            export_onnx (bool): whether export model to onnx
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
        """
        if self.nms is not None:
            bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
            bbox_pred, bbox_num, before_nms_indexes = self.nms(bboxes, score,
                                                               self.num_classes)

        else:
            bbox_pred, bbox_num = self.decode(head_out, rois, im_shape,
                                              scale_factor)

        if self.export_onnx:
            # add fake box after postprocess when exporting onnx
            fake_bboxes = torch.tensor(
                np.array(
                    [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32), device=bbox_pred.device)  # Ensure same device

            bbox_pred = torch.cat([bbox_pred, fake_bboxes])
            bbox_num = bbox_num + 1

        if self.nms is not None:
            return bbox_pred, bbox_num, before_nms_indexes
        else:
            return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.

        Notes:
        Currently only support bs = 1.

        Args:
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """
        if self.export_eb:
            # enable rcnn models for edgeboard hw to skip the following postprocess.
            return bboxes, bboxes, bbox_num

        if not self.export_onnx:
            bboxes_list = []
            bbox_num_list = []
            id_start = 0
            fake_bboxes = torch.tensor(
                np.array(
                    [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32), device=bboxes.device)  # Ensure same device
            fake_bbox_num = torch.tensor(np.array([1], dtype=np.int32), device=bbox_num.device)  # Ensure same device

            # add fake bbox when output is empty for each batch
            for i in range(bbox_num.shape[0]):
                if bbox_num[i] == 0:
                    bboxes_i = fake_bboxes
                    bbox_num_i = fake_bbox_num
                else:
                    bboxes_i = bboxes[id_start:id_start + int(bbox_num[i]), :]
                    bbox_num_i = bbox_num[i:i + 1]
                    # id_start: 0-dim, bbox_num: 1-dim. Use bbox_num[i] instead of bbox_num[i:i+1] in pir.
                    id_start += int(bbox_num[i])  # Convert to int for addition
                bboxes_list.append(bboxes_i)
                bbox_num_list.append(bbox_num_i)
            bboxes = torch.cat(bboxes_list)
            bbox_num = torch.cat(bbox_num_list)

        # Calculate original shape
        # im_shape: [batch, 2] (h, w), scale_factor: [batch, 2] (scale_y, scale_x)
        origin_shape = torch.floor(im_shape / scale_factor + 0.5)

        if not self.export_onnx:
            origin_shape_list = []
            scale_factor_list = []
            # scale_factor: scale_y, scale_x per batch
            for i in range(bbox_num.shape[0]):
                # Number of boxes for current batch sample
                num_boxes_i = int(bbox_num[i:i + 1]) if bbox_num.dim() > 0 else int(bbox_num)
                if num_boxes_i == 0:
                    # If no boxes for this sample, skip or append empty tensors of correct shape
                    # This loop might not execute, but logic is here if needed
                    continue
                # Expand origin_shape for current batch sample to [num_boxes_i, 2]
                # origin_shape[i:i+1, :] has shape [1, 2]
                expand_shape = origin_shape[i:i + 1, :].expand(num_boxes_i, -1)  # [num_boxes_i, 2]

                scale_y = scale_factor[i, 0]
                scale_x = scale_factor[i, 1]
                # Create scale tensor [scale_x, scale_y, scale_x, scale_y] for current batch
                scale = torch.stack([scale_x, scale_y, scale_x, scale_y])
                # Expand scale for current batch sample to [num_boxes_i, 4]
                expand_scale = scale.expand(num_boxes_i, -1)

                origin_shape_list.append(expand_shape)
                scale_factor_list.append(expand_scale)

            # Concatenate lists to form final tensors
            self.origin_shape_list = torch.cat(origin_shape_list) if origin_shape_list else torch.empty((0, 2),
                                                                                                        dtype=origin_shape.dtype,
                                                                                                        device=origin_shape.device)
            scale_factor_list = torch.cat(scale_factor_list) if scale_factor_list else torch.empty((0, 4),
                                                                                                   dtype=scale_factor.dtype,
                                                                                                   device=scale_factor.device)

        else:
            # simplify the computation for bs=1 when exporting onnx
            scale_y = scale_factor[0, 0]
            scale_x = scale_factor[0, 1]
            scale = torch.stack([scale_x, scale_y, scale_x, scale_y]).unsqueeze(0)
            # bbox_num[0:1] is a 1-dim tensor with 1 element, get its value
            num_boxes_bs1 = int(bbox_num[0:1]) if bbox_num.dim() > 0 else int(bbox_num)
            self.origin_shape_list = origin_shape[0:1, :].expand(num_boxes_bs1, -1)
            scale_factor_list = scale.expand(num_boxes_bs1, -1)

        # bboxes: [N, 6], label, score, bbox
        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]

        # rescale bbox to original image
        # pred_bbox: [N, 4], scale_factor_list: [N, 4]
        scaled_bbox = pred_bbox / scale_factor_list

        # Get original dimensions for clipping
        # self.origin_shape_list: [N, 2] (orig_h, orig_w)
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = torch.zeros_like(origin_h, dtype=scaled_bbox.dtype, device=scaled_bbox.device)

        # clip bbox to [0, original_size]
        x1 = torch.maximum(torch.minimum(scaled_bbox[:, 0], origin_w), zeros)
        y1 = torch.maximum(torch.minimum(scaled_bbox[:, 1], origin_h), zeros)
        x2 = torch.maximum(torch.minimum(scaled_bbox[:, 2], origin_w), zeros)
        y2 = torch.maximum(torch.minimum(scaled_bbox[:, 3], origin_h), zeros)
        # Stack clipped coordinates back into [N, 4] format
        pred_bbox = torch.stack([x1, y1, x2, y2], dim=-1)

        # filter empty bbox
        # pred_bbox: [N, 4]
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = torch.unsqueeze(keep_mask, dim=1)
        # pred_label: [N, 1], keep_mask: [N, 1], ones_like: [N, 1]
        pred_label = torch.where(keep_mask, pred_label,
                                 torch.ones_like(pred_label) * -1)
        # Concatenate final results
        pred_result = torch.cat([pred_label, pred_score, pred_bbox], dim=1)
        return bboxes, pred_result, bbox_num

    def get_origin_shape(self):
        return self.origin_shape_list


def paste_mask(masks, boxes, im_h, im_w, assign_on_cpu=False):
    """
    Paste the mask prediction to the original image.

    Args:
        masks (Tensor): Predicted masks, shape [N, 1, mask_h, mask_w] (e.g., [N, 1, 28, 28]).
        boxes (Tensor): Bounding boxes, shape [N, 4] in format [x1, y1, x2, y2].
        im_h (int or Tensor): Height of the original image.
        im_w (int or Tensor): Width of the original image.
        assign_on_cpu (bool): Whether to perform the operation on CPU.

    Returns:
        img_masks (Tensor): Pasted masks on the original image space, shape [N, im_h, im_w].
    """
    x0_int, y0_int = 0, 0
    x1_int, y1_int = im_w, im_h
    # Split boxes along axis 1 (last dim of boxes is 4) -> x0, y0, x1, y1 each shape [N, 1]
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)

    N = masks.shape[0]
    # Create coordinate grids for the original image space
    img_y = torch.arange(y0_int, y1_int, dtype=masks.dtype, device=masks.device) + 0.5
    img_x = torch.arange(x0_int, x1_int, dtype=masks.dtype, device=masks.device) + 0.5

    # Move tensors to CPU if required
    if assign_on_cpu:
        masks_cpu = masks.cpu()
        x0_cpu = x0.cpu()
        y0_cpu = y0.cpu()
        x1_cpu = x1.cpu()
        y1_cpu = y1.cpu()

        img_y_cpu = torch.arange(y0_int, y1_int, dtype=masks_cpu.dtype, device=masks_cpu.device) + 0.5
        img_x_cpu = torch.arange(x0_int, x1_int, dtype=masks_cpu.dtype, device=masks_cpu.device) + 0.5
        img_y_norm_cpu = (img_y_cpu[None, :] - y0_cpu) / (y1_cpu - y0_cpu) * 2 - 1
        img_x_norm_cpu = (img_x_cpu[None, :] - x0_cpu) / (x1_cpu - x0_cpu) * 2 - 1

        # Prepare grid for grid_sample
        # gx shape: [N, im_h, im_w], gy shape: [N, im_h, im_w]
        gx_cpu = img_x_norm_cpu[:, None, :].expand(N, img_y_norm_cpu.shape[1], img_x_norm_cpu.shape[1])
        gy_cpu = img_y_norm_cpu[:, :, None].expand(N, img_y_norm_cpu.shape[1], img_x_norm_cpu.shape[1])
        # Stack gx and gy along the last dimension to create the grid: [N, im_h, im_w, 2]
        grid_cpu = torch.stack([gx_cpu, gy_cpu], dim=3)

        img_masks_cpu = F.grid_sample(masks_cpu, grid_cpu, align_corners=False)
        # Squeeze the channel dimension: [N, im_h, im_w]
        img_masks_result = img_masks_cpu[:, 0, :, :]
        return img_masks_result

    else:
        # Prepare grid for grid_sample on the original device
        # gx shape: [N, im_h, im_w], gy shape: [N, im_h, im_w]
        gx = img_x[:, None, :].expand(N, img_y.shape[0], img_x.shape[0])
        gy = img_y[:, :, None].expand(N, img_y.shape[0], img_x.shape[0])
        # Stack gx and gy along the last dimension to create the grid: [N, im_h, im_w, 2]
        grid = torch.stack([gx, gy], dim=3)

        # Perform grid sampling
        img_masks = F.grid_sample(masks, grid, align_corners=False)
        # Squeeze the channel dimension: [N, im_h, im_w]
        return img_masks[:, 0, :, :]


class MaskPostProcess(object):
    """
    refer to:
    https://github.com/facebookresearch/detectron2/layers/mask_ops.py

    Get Mask output according to the output from model
    """

    def __init__(self,
                 binary_thresh=0.5,
                 export_onnx=False,
                 assign_on_cpu=False):
        super(MaskPostProcess, self).__init__()
        self.binary_thresh = binary_thresh
        self.export_onnx = export_onnx
        self.assign_on_cpu = assign_on_cpu

    def __call__(self, mask_out, bboxes, bbox_num, origin_shape):
        """
        Decode the mask_out and paste the mask to the origin image.

        Args:
            mask_out (Tensor): mask_head output with shape [N, 28, 28].
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            origin_shape (Tensor): The origin shape of the input image, the tensor
                shape is [N, 2], and each row is [h, w].
        Returns:
            pred_result (Tensor): The final prediction mask results with shape
                [N, h, w] in binary mask style.
        """
        num_mask = mask_out.shape[0]
        origin_shape = origin_shape.to(torch.int32)

        if self.export_onnx:
            # Handle batch size 1 case for ONNX export
            h = origin_shape[0, 0]
            w = origin_shape[0, 1]
            # Call paste_mask with assign_on_cpu flag
            # mask_out[:, None, :, :] adds a dimension: [N, 1, 28, 28]
            # bboxes[:, 2:] gets the 4 coordinates: [N, 4]
            mask_onnx = paste_mask(mask_out[:, None, :, :], bboxes[:, 2:], h, w,
                                   self.assign_on_cpu)
            # Apply binary threshold
            mask_onnx = mask_onnx >= self.binary_thresh
            pred_result = mask_onnx.to(torch.int32)

        else:
            # Find maximum height and width across all images in the batch
            max_h = torch.max(origin_shape[:, 0]).item()
            max_w = torch.max(origin_shape[:, 1]).item()
            # Create a result tensor filled with -1
            pred_result = torch.zeros([num_mask, max_h, max_w], dtype=torch.int32, device=mask_out.device) - 1

            id_start = 0
            # Iterate over each image in the batch
            for i in range(bbox_num.shape[0]):
                # Get bboxes and mask_out for the current image
                num_boxes_i = int(bbox_num[i])
                bboxes_i = bboxes[id_start:id_start + num_boxes_i, :]
                mask_out_i = mask_out[id_start:id_start + num_boxes_i, :, :]
                # Get original image dimensions
                im_h = int(origin_shape[i, 0])
                im_w = int(origin_shape[i, 1])
                # Call paste_mask for the current image's masks and bboxes
                # mask_out_i[:, None, :, :] -> [num_boxes_i, 1, 28, 28]
                # bboxes_i[:, 2:] -> [num_boxes_i, 4]
                pred_mask = paste_mask(mask_out_i[:, None, :, :],
                                       bboxes_i[:, 2:], im_h, im_w,
                                       self.assign_on_cpu)
                pred_mask = (pred_mask >= self.binary_thresh).to(torch.int32)
                # Assign the computed mask to the result tensor
                # pred_result[id_start:id_start + num_boxes_i, :im_h, :im_w] gets a view of shape [num_boxes_i, im_h, im_w]
                # pred_mask should have shape [num_boxes_i, im_h, im_w] after paste_mask
                pred_result[id_start:id_start + num_boxes_i, :im_h, :im_w] = pred_mask
                id_start += num_boxes_i

        return pred_result


class JDEBBoxPostProcess(nn.Module):
    def __init__(self, num_classes=1, decode=None, nms=None, return_idx=True):
        super(JDEBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.return_idx = return_idx

        self.register_buffer('fake_bbox_pred',
                             torch.tensor([[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        self.register_buffer('fake_bbox_num',
                             torch.tensor([1], dtype=torch.int32))
        self.register_buffer('fake_nms_keep_idx',
                             torch.tensor([[0]], dtype=torch.int32))

        self.register_buffer('fake_yolo_boxes_out',
                             torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32))
        self.register_buffer('fake_yolo_scores_out',
                             torch.tensor([[[0.0]]], dtype=torch.float32))
        self.register_buffer('fake_boxes_idx',
                             torch.tensor([[0]], dtype=torch.int64))

    def forward(self, head_out, anchors):
        """
        Decode the bbox and do NMS for JDE model.

        Args:
            head_out (list): Bbox_pred and cls_prob of bbox_head output.
            anchors (list): Anchors of JDE model.

        Returns:
            boxes_idx (Tensor): The index of kept bboxes after decode 'JDEBox'.
            bbox_pred (Tensor): The output is the prediction with shape [N, 6]
                including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction of each batch with shape [N].
            nms_keep_idx (Tensor): The index of kept bboxes after NMS.
        """
        boxes_idx, yolo_boxes_scores = self.decode(head_out, anchors)

        if len(boxes_idx) == 0:
            boxes_idx = self.fake_boxes_idx
            yolo_boxes_out = self.fake_yolo_boxes_out
            yolo_scores_out = self.fake_yolo_scores_out
        else:
            yolo_boxes = torch.gather(yolo_boxes_scores, 0,
                                      boxes_idx.unsqueeze(-1).expand(-1, yolo_boxes_scores.size(-1)))
            # TODO: only support bs=1 now
            yolo_boxes_out = yolo_boxes[:, :4].unsqueeze(0)  # [1, len(boxes_idx), 4]
            yolo_scores_out = yolo_boxes[:, 4:5].unsqueeze(0).unsqueeze(1)  # [1, 1, len(boxes_idx)]
            boxes_idx = boxes_idx[:, 1:]

        if self.return_idx:
            bbox_pred, bbox_num, nms_keep_idx = self.nms(
                yolo_boxes_out, yolo_scores_out, self.num_classes)
            if bbox_pred.size(0) == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
                nms_keep_idx = self.fake_nms_keep_idx
            return boxes_idx, bbox_pred, bbox_num, nms_keep_idx
        else:
            bbox_pred, bbox_num, _ = self.nms(yolo_boxes_out, yolo_scores_out,
                                              self.num_classes)
            if bbox_pred.size(0) == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
            return _, bbox_pred, bbox_num, _


class CenterNetPostProcess(object):
    """
    Postprocess the model outputs to get final prediction:
        1. Do NMS for heatmap to get top `max_per_img` bboxes.
        2. Decode bboxes using center offset and box size.
        3. Rescale decoded bboxes reference to the origin image shape.
    Args:
        max_per_img(int): the maximum number of predicted objects in a image,
            500 by default.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        regress_ltrb (bool): whether to regress left/top/right/bottom or
            width/height for a box, true by default.
    """

    def __init__(self, max_per_img=500, down_ratio=4, regress_ltrb=True):
        super(CenterNetPostProcess, self).__init__()
        self.max_per_img = max_per_img
        self.down_ratio = down_ratio
        self.regress_ltrb = regress_ltrb
        # _simple_nms() _topk() are same as TTFBox in ppdet/modeling/layers.py

    @staticmethod
    def _simple_nms(heat, kernel=3):
        """ Use maxpool to filter the max score, get local peaks. """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores):
        """ Select top k scores and decode to get xy coordinates. """
        k = self.max_per_img
        batch_size, cat, height, width = scores.shape
        # batch size is 1
        scores_r = scores.view(cat, -1)
        topk_scores, topk_inds = torch.topk(scores_r, k, dim=1)

        topk_ys = topk_inds // width
        topk_xs = topk_inds % width

        topk_score_r = topk_scores.view(-1)
        topk_score, topk_ind = torch.topk(topk_score_r, k)

        k_t = torch.full(topk_ind.shape, k, dtype=torch.long, device=topk_ind.device)
        topk_clses = (topk_ind // k_t).float()

        topk_inds = topk_inds.view(-1)
        topk_ys = topk_ys.view(-1, 1)
        topk_xs = topk_xs.view(-1, 1)

        topk_inds = topk_inds[topk_ind]
        topk_ys = topk_ys[topk_ind]
        topk_xs = topk_xs[topk_ind]

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def __call__(self, hm, wh, reg, im_shape, scale_factor):
        # 1.get clses and scores, note that hm had been done sigmoid
        heat = self._simple_nms(hm)
        scores, inds, topk_clses, ys, xs = self._topk(heat)
        clses = topk_clses.unsqueeze(1)
        scores = scores.unsqueeze(1)

        # 2.get bboxes, note only support batch_size=1 now
        reg_t = reg.permute(0, 2, 3, 1)
        reg = reg_t.view(-1, reg_t.shape[-1])
        reg = reg[inds]
        xs = xs.float()
        ys = ys.float()
        xs = xs + reg[:, 0:1]
        ys = ys + reg[:, 1:2]
        wh_t = wh.permute(0, 2, 3, 1)
        wh = wh_t.view(-1, wh_t.shape[-1])
        wh = wh[inds]
        if self.regress_ltrb:
            x1 = xs - wh[:, 0:1]
            y1 = ys - wh[:, 1:2]
            x2 = xs + wh[:, 2:3]
            y2 = ys + wh[:, 3:4]
        else:
            x1 = xs - wh[:, 0:1] / 2
            y1 = ys - wh[:, 1:2] / 2
            x2 = xs + wh[:, 0:1] / 2
            y2 = ys + wh[:, 1:2] / 2
        n, c, feat_h, feat_w = hm.shape
        padw = (feat_w * self.down_ratio - im_shape[0, 1]) / 2
        padh = (feat_h * self.down_ratio - im_shape[0, 0]) / 2
        x1 = x1 * self.down_ratio
        y1 = y1 * self.down_ratio
        x2 = x2 * self.down_ratio
        y2 = y2 * self.down_ratio
        x1 = x1 - padw
        y1 = y1 - padh
        x2 = x2 - padw
        y2 = y2 - padh
        bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        scale_y = scale_factor[:, 0:1]
        scale_x = scale_factor[:, 1:2]
        scale_expand = torch.cat(
            [scale_x, scale_y, scale_x, scale_y], dim=1)
        boxes_shape = bboxes.shape
        scale_expand = scale_expand.expand(boxes_shape)
        bboxes = bboxes / scale_expand

        results = torch.cat([clses, scores, bboxes], dim=1)
        return results, torch.tensor([results.shape[0]], dtype=torch.int32,
                                     device=results.device), inds, topk_clses, ys, xs


class DETRPostProcess(object):
    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 dual_queries=False,
                 dual_groups=0,
                 use_focal_loss=False,
                 with_mask=False,
                 mask_stride=4,
                 mask_threshold=0.5,
                 use_avg_mask_score=False,
                 bbox_decode_type='origin'):
        super(DETRPostProcess, self).__init__()
        assert bbox_decode_type in ['origin', 'pad']

        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.mask_stride = mask_stride
        self.mask_threshold = mask_threshold
        self.use_avg_mask_score = use_avg_mask_score
        self.bbox_decode_type = bbox_decode_type

    def _mask_postprocess(self, mask_pred, score_pred):
        mask_score = torch.sigmoid(mask_pred)
        mask_pred = (mask_score > self.mask_threshold).float()
        if self.use_avg_mask_score:
            avg_mask_score = (mask_pred * mask_score).sum([-2, -1]) / (
                    mask_pred.sum([-2, -1]) + 1e-6)
            score_pred *= avg_mask_score

        return mask_pred.flatten(0, 1).int(), score_pred

    def __call__(self, head_out, im_shape, scale_factor, pad_shape):
        """
        Decode the bbox and mask.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            im_shape (Tensor): The shape of the input image without padding.
            scale_factor (Tensor): The scale factor of the input image.
            pad_shape (Tensor): The shape of the input image with padding.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out
        if self.dual_queries:
            num_queries = logits.shape[1]
            logits = logits[:, :int(num_queries // (self.dual_groups + 1)), :]
            bboxes = bboxes[:, :int(num_queries // (self.dual_groups + 1)), :]

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)
        # calculate the original shape of the image
        origin_shape = torch.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = origin_shape.chunk(2, dim=-1)
        if self.bbox_decode_type == 'pad':
            # calculate the shape of the image with padding
            out_shape = pad_shape / im_shape * origin_shape
            out_shape = torch.flip(out_shape, [-1]).repeat(1, 2).unsqueeze(1)
        elif self.bbox_decode_type == 'origin':
            out_shape = torch.flip(origin_shape, [-1]).repeat(1, 2).unsqueeze(1)
        else:
            raise Exception(
                f'Wrong `bbox_decode_type`: {self.bbox_decode_type}.')
        bbox_pred *= out_shape

        scores = torch.sigmoid(logits) if self.use_focal_loss else torch.softmax(
            logits, dim=-1)[:, :, :-1]

        if not self.use_focal_loss:
            scores, labels = torch.max(scores, -1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(
                    scores, self.num_top_queries, dim=-1)
                batch_ind = torch.arange(
                    scores.shape[0], device=scores.device).unsqueeze(-1).repeat(
                    1, self.num_top_queries)
                index_2d = torch.stack([batch_ind, index], dim=-1)
                labels = labels.gather(1, index)
                bbox_pred = bbox_pred.gather(1, index_2d.unsqueeze(-1).expand(-1, -1, bbox_pred.size(-1)))
        else:
            scores_flat = scores.flatten(1)
            scores, index = torch.topk(
                scores_flat, self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            batch_ind = torch.arange(scores.shape[0], device=scores.device).unsqueeze(-1).repeat(
                1, self.num_top_queries)
            index_2d = torch.stack([batch_ind, index], dim=-1)
            bbox_pred = bbox_pred.gather(1, index_2d.unsqueeze(-1).expand(-1, -1, bbox_pred.size(-1)))

        mask_pred = None
        if self.with_mask:
            assert masks is not None
            assert masks.shape[0] == 1
            # For gathering, we need to flatten the index for 4D tensor
            batch_size, num_queries_m, height, width = masks.shape
            masks_flat = masks.view(batch_size, num_queries_m, -1)
            index_expanded = index_2d[:, :, 1:2].unsqueeze(-1).expand(-1, -1, height * width)
            masks_selected = torch.gather(masks_flat, 1, index_expanded)
            masks = masks_selected.view(batch_size, -1, height, width)

            if self.bbox_decode_type == 'pad':
                masks = F.interpolate(
                    masks,
                    scale_factor=self.mask_stride,
                    mode="bilinear",
                    align_corners=False)
                # TODO: Support prediction with bs>1.
                # remove padding for input image
                h, w = im_shape.to(torch.int32)[0]
                masks = masks[..., :h, :w]
            # get pred_mask in the original resolution.
            img_h_int = img_h[0].to(torch.int32)
            img_w_int = img_w[0].to(torch.int32)
            masks = F.interpolate(
                masks,
                size=[img_h_int.item(), img_w_int.item()],
                mode="bilinear",
                align_corners=False)
            mask_pred, scores = self._mask_postprocess(masks, scores)

        bbox_pred = torch.cat(
            [
                labels.unsqueeze(-1).float(), scores.unsqueeze(-1),
                bbox_pred
            ],
            dim=-1)
        bbox_num = torch.tensor(
            self.num_top_queries, dtype=torch.int32, device=bbox_pred.device).repeat(bbox_pred.shape[0])
        bbox_pred = bbox_pred.view(-1, 6)
        return bbox_pred, bbox_num, mask_pred


class SparsePostProcess(object):
    def __init__(self,
                 num_proposals,
                 num_classes=80,
                 binary_thresh=0.5,
                 assign_on_cpu=False):
        super(SparsePostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.binary_thresh = binary_thresh
        self.assign_on_cpu = assign_on_cpu

    def __call__(self, scores, bboxes, scale_factor, ori_shape, masks=None):
        assert len(scores) == len(bboxes) == \
               len(ori_shape) == len(scale_factor)

        device = scores.device
        batch_size = len(ori_shape)

        scores = torch.sigmoid(scores)
        has_mask = masks is not None
        if has_mask:
            masks = torch.sigmoid(masks)
            masks = masks.view(batch_size, -1, *masks.shape[1:])

        bbox_pred = []
        mask_pred = [] if has_mask else None
        bbox_num = torch.zeros([batch_size], dtype=torch.int32, device=scores.device)

        for i in range(batch_size):
            score = scores[i]
            bbox = bboxes[i]
            score_flat = score.flatten(0, 1)
            topk_score, indices = torch.topk(
                score_flat, self.num_proposals, sorted=False)
            label = indices % self.num_classes

            if has_mask:
                mask = masks[i]
                mask = mask.flatten(0, 1)[indices]

            H, W = int(ori_shape[i][0]), int(ori_shape[i][1])
            bbox_indices = (indices // self.num_classes).long()
            bbox_selected = bbox[bbox_indices]
            bbox_selected = bbox_selected / scale_factor[i]
            bbox_selected[:, 0::2] = torch.clamp(bbox_selected[:, 0::2], 0, W)
            bbox_selected[:, 1::2] = torch.clamp(bbox_selected[:, 1::2], 0, H)

            keep = ((bbox_selected[:, 2] - bbox_selected[:, 0]).cpu().numpy() > 1.) & \
                   ((bbox_selected[:, 3] - bbox_selected[:, 1]).cpu().numpy() > 1.)
            keep_tensor = torch.from_numpy(keep).to(bbox_selected.device)

            if keep_tensor.sum() == 0:
                bbox_result = torch.zeros([1, 6], dtype=torch.float32, device=scores.device)
                if has_mask:
                    mask_result = torch.zeros([1, H, W], dtype=torch.uint8, device=scores.device)
            else:
                label_selected = label[keep_tensor].float().unsqueeze(-1)
                score_selected = topk_score[keep_tensor].float().unsqueeze(-1)
                bbox_result = bbox_selected[keep_tensor].float()

                if has_mask:
                    mask_selected = mask[keep_tensor].float().unsqueeze(1)
                    mask_result = paste_mask(mask_selected, bbox_result, H, W, self.assign_on_cpu)
                    mask_result = (mask_result >= self.binary_thresh).byte()

                bbox_result = torch.cat([label_selected, score_selected, bbox_result], dim=-1)

            bbox_num[i] = bbox_result.shape[0]
            bbox_pred.append(bbox_result)
            if has_mask:
                mask_pred.append(mask_result)

        bbox_pred = torch.cat(bbox_pred, dim=0)
        if has_mask:
            mask_pred = torch.cat(mask_pred, dim=0)

        if has_mask:
            return bbox_pred, bbox_num, mask_pred
        else:
            return bbox_pred, bbox_num


def multiclass_nms(bboxs, num_classes, match_threshold=0.6, match_metric='iou'):
    final_boxes = []
    for c in range(num_classes):
        idxs = bboxs[:, 0] == c
        if np.count_nonzero(idxs) == 0: continue
        r = nms(bboxs[idxs, 1:], match_threshold, match_metric)
        final_boxes.append(np.concatenate([np.full((r.shape[0], 1), c), r], 1))
    return final_boxes


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if match_metric == 'iou':
            union = areas[i] + areas[order[1:]] - inter
            match_value = inter / union
        elif match_metric == 'ios':
            smaller = np.minimum(areas[i], areas[order[1:]])
            match_value = inter / smaller
        else:
            raise ValueError()

        inds = np.where(match_value < match_threshold)[0]
        order = order[inds + 1]

    dets = dets[keep, :]
    return dets


class DETRBBoxSemiPostProcess(object):
    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 use_focal_loss=False):
        super(DETRBBoxSemiPostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    def __call__(self, head_out):
        """
        Decode the bbox.
        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out
        bbox_pred = bboxes

        scores = F.softmax(logits, dim=2)

        soft_scores = copy.deepcopy(scores)
        scores_max, _ = torch.max(scores, dim=-1)
        topk_scores, topk_indices = torch.topk(scores_max, 300, dim=-1)

        batch_size, _, _ = scores.shape
        batch_ind = torch.arange(batch_size, device=scores.device).unsqueeze(-1).repeat(
            1, 300)
        index_2d = torch.stack([batch_ind, topk_indices], dim=-1)

        # Equivalent to paddle.gather_nd
        # For labels: gather the argmax of soft_scores at the selected indices
        labels_full = torch.argmax(soft_scores, dim=-1)
        batch_idx = index_2d[:, :, 0]  # batch indices
        query_idx = index_2d[:, :, 1]  # query indices
        labels = labels_full[batch_idx, query_idx].int()

        # For score_class: gather soft_scores at the selected indices
        score_class = soft_scores[batch_idx, query_idx]

        # For bbox_pred: gather bbox_pred at the selected indices
        bbox_pred_selected = bbox_pred[batch_idx, query_idx]

        bbox_pred_final = torch.cat(
            [
                labels.unsqueeze(-1).float(),
                score_class,
                topk_scores.unsqueeze(-1),
                bbox_pred_selected
            ],
            dim=-1)
        bbox_num = torch.tensor(
            bbox_pred_final.shape[1], dtype=torch.int32, device=bbox_pred_final.device).repeat(bbox_pred_final.shape[0])
        bbox_pred_final = bbox_pred_final.view(-1, bbox_pred_final.shape[-1])
        return bbox_pred_final, bbox_num