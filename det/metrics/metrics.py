#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :metrics.py
@Author :CodeCat
@Date   :2025/11/19 17:05
"""
import abc
from typing import Any
import os
import sys
import json
import torch
import numpy as np
import typing
from collections import defaultdict
from pathlib import Path

from det.metrics.map_utils import prune_zero_padding, DetectionMAP
from det.metrics.coco_utils import get_infer_results, cocoapi_eval
from det.metrics.widerface_utils import face_eval_run, image_eval, img_pr_info, dataset_pr_info
from det.data.source.category import get_categories
from det.modeling.rbox_utils import poly2rbox_np
from loguru import logger

__all__ = [
    'Metric', 'COCOMetric', 'VOCMetric', 'WiderFaceMetric', 'get_infer_results',
    'RBoxMetric', 'SNIPERCOCOMetric'
]

# COCO关键点的标准差参数，用于OKS (Object Keypoint Similarity) 计算
# 每个值对应COCO数据集中一个关键点的归一化标准差
# 除以10.0是为了将原始值转换为相对于人体框尺寸的比例
COCO_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87,
    .89, .89
]) / 10.0

# CrowdPose数据集的关键点标准差参数
# CrowdPose包含14个关键点，对应人体的主要关节点
CROWD_SIGMAS = np.array(
    [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79,
     .79]) / 10.0


class Metric(metaclass=abc.ABCMeta):
    """
    指标基类，封装指标逻辑和API
    """

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        重置状态和结果
        """
        raise NotImplementedError(
            f"function 'reset' not implemented in {self.__class__.__name__}."
        )

    @abc.abstractmethod
    def update(self, *args: Any) -> None:
        """
        更新指标状态
        """
        raise NotImplementedError(
            f"function 'update' not implemented in {self.__class__.__name__}."
        )

    @abc.abstractmethod
    def accumulate(self) -> Any:
        """
        累积统计，计算并返回指标值
        """
        raise NotImplementedError(
            f"function 'accumulate' not implemented in {self.__class__.__name__}."
        )

    @abc.abstractmethod
    def name(self) -> str:
        """
        返回指标名称
        """
        raise NotImplementedError(
            f"function 'name' not implemented in {self.__class__.__name__}."
        )

    def compute(self, *args: Any) -> Any:
        """
        高级用法，用于加速指标计算
        """
        return args

    # 在 ppdet 中，我们还需要以下 2 个方法：

    # 用于记录指标结果的抽象方法
    def log(self):
        """
        记录指标结果
        """
        pass

    # 用于获取指标结果的抽象方法
    def get_results(self):
        """
        获取指标结果
        """
        pass


class COCOMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        """
        COCO指标类

        Args:
            anno_file: 注释文件路径
            **kwargs: 其他参数
        """
        super().__init__()
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories('COCO', anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.iou_type = kwargs.get('IouType', 'bbox')

        if not self.save_prediction_only:
            assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)

        if self.output_eval is not None:
            Path(self.output_eval).mkdir(exist_ok=True)

        self.save_threshold = kwargs.get('save_threshold', 0)

        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        # 目前仅支持bbox和mask评估
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        """
        更新指标统计值

        Args:
            inputs: 输入数据
            outputs: 输出数据
        """
        outs = {}
        # 将输出张量转换为numpy数组
        for k, v in outputs.items():
            outs[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

        # 多尺度输入：所有输入都有相同的im_id
        if isinstance(inputs, typing.Sequence):
            im_id = inputs[0]['im_id']
        else:
            im_id = inputs['im_id']
        outs['im_id'] = im_id.cpu().numpy() if isinstance(im_id,
                                                          torch.Tensor) else im_id
        if 'im_file' in inputs:
            outs['im_file'] = inputs['im_file']

        infer_results = get_infer_results(
            outs,
            self.clsid2catid,
            bias=self.bias,
            save_threshold=self.save_threshold)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.results['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []
        self.results['segm'] += infer_results[
            'segm'] if 'segm' in infer_results else []
        self.results['keypoint'] += infer_results[
            'keypoint'] if 'keypoint' in infer_results else []

    def accumulate(self):
        """
        汇总指标统计值
        """
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                print('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                print('The bbox result is saved to {} and do not '
                      'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        if len(self.results['mask']) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['mask'], f)
                print('The mask result is saved to mask.json.')

            if self.save_prediction_only:
                print('The mask result is saved to {} and do not '
                      'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['segm']) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['segm'], f)
                print('The segm result is saved to segm.json.')

            if self.save_prediction_only:
                print('The segm result is saved to {} and do not '
                      'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['keypoint']) > 0:
            output = "keypoint.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['keypoint'], f)
                print('The keypoint result is saved to keypoint.json.')

            if self.save_prediction_only:
                print('The keypoint result is saved to {} and do not '
                      'evaluate the mAP.'.format(output))
            else:
                style = 'keypoints'
                use_area = True
                sigmas = COCO_SIGMAS
                if self.iou_type == 'keypoints_crowd':
                    style = 'keypoints_crowd'
                    use_area = False
                    sigmas = CROWD_SIGMAS
                keypoint_stats = cocoapi_eval(
                    output,
                    style,
                    anno_file=self.anno_file,
                    classwise=self.classwise,
                    sigmas=sigmas,
                    use_area=use_area)
                self.eval_results['keypoint'] = keypoint_stats
                sys.stdout.flush()

    def log(self):
        """
        记录指标结果
        """
        pass

    def get_results(self):
        """
        获取指标结果

        Returns:
            指标结果字典
        """
        return self.eval_results

    def name(self):
        return self.__class__.__name__


class VOCMetric(Metric):
    def __init__(self,
                 label_list,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False,
                 output_eval=None,
                 save_prediction_only=False):
        """
        VOC指标类

        Args:
            label_list: 标签列表文件路径
            class_num: 类别数量，默认20
            overlap_thresh: IOU阈值，默认0.5
            map_type: MAP类型，默认'11point'
            is_bbox_normalized: 边界框是否归一化
            evaluate_difficult: 是否评估困难样本
            classwise: 是否按类别计算
            output_eval: 评估输出目录
            save_prediction_only: 是否只保存预测结果
        """
        super(VOCMetric, self).__init__()
        assert os.path.isfile(label_list), \
            "label_list {} not a file".format(label_list)
        self.clsid2catid, self.catid2name = get_categories('VOC', label_list)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.output_eval = output_eval
        self.save_prediction_only = save_prediction_only
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.catid2name,
            classwise=classwise)

        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.results = {'bbox': [], 'score': [], 'label': []}
        self.detection_map.reset()

    def update(self, inputs, outputs):
        """
        更新指标统计值

        Args:
            inputs: 输入数据
            outputs: 输出数据
        """
        bbox_np = outputs['bbox'].cpu().numpy() if isinstance(
            outputs['bbox'], torch.Tensor) else outputs['bbox']
        bboxes = bbox_np[:, 2:]
        scores = bbox_np[:, 1]
        labels = bbox_np[:, 0]
        bbox_lengths = outputs['bbox_num'].cpu().numpy() if isinstance(
            outputs['bbox_num'], torch.Tensor) else outputs['bbox_num']

        self.results['bbox'].append(bboxes.tolist())
        self.results['score'].append(scores.tolist())
        self.results['label'].append(labels.tolist())

        if bboxes.shape == (1, 1) or bboxes is None:
            return
        if self.save_prediction_only:
            return

        gt_boxes = inputs['gt_bbox']
        gt_labels = inputs['gt_class']
        difficults = inputs['difficult'] if not self.evaluate_difficult \
            else None

        if 'scale_factor' in inputs:
            scale_factor = inputs['scale_factor'].cpu().numpy() if isinstance(
                inputs['scale_factor'],
                torch.Tensor) else inputs['scale_factor']
        else:
            scale_factor = np.ones((gt_boxes.shape[0], 2)).astype('float32')

        bbox_idx = 0
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i].cpu().numpy() if isinstance(
                gt_boxes[i], torch.Tensor) else gt_boxes[i]
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h])
            gt_label = gt_labels[i].cpu().numpy() if isinstance(
                gt_labels[i], torch.Tensor) else gt_labels[i]
            if difficults is not None:
                difficult = difficults[i].cpu().numpy() if isinstance(
                    difficults[i], torch.Tensor) else difficults[i]
            else:
                difficult = None
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            score = scores[bbox_idx:bbox_idx + bbox_num]
            label = labels[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            self.detection_map.update(bbox, score, label, gt_box, gt_label,
                                      difficult)
            bbox_idx += bbox_num

    def accumulate(self):
        """
        汇总指标统计值
        """
        output = "bbox.json"
        if self.output_eval:
            output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results, f)
                print('The bbox result is saved to bbox.json.')
        if self.save_prediction_only:
            return

        print("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        """
        记录指标结果
        """
        map_stat = 100. * self.detection_map.get_map()
        print("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                 self.map_type, map_stat))

    def get_results(self):
        """
        获取指标结果

        Returns:
            指标结果字典
        """
        return {'bbox': [self.detection_map.get_map()]}

    def name(self):
        return self.__class__.__name__


class WiderFaceMetric(Metric):
    def __init__(self, iou_thresh=0.5):
        """
        WiderFace指标类

        Args:
            iou_thresh: IOU阈值，默认0.5
        """
        super(WiderFaceMetric, self).__init__()
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.pred_boxes_list = []
        self.gt_boxes_list = []
        self.aps = []

        self.hard_ignore_list = []
        self.medium_ignore_list = []
        self.easy_ignore_list = []

    def update(self, data, outs):
        """
        更新指标统计值

        Args:
            data: 输入数据
            outs: 输出数据
        """
        batch_pred_bboxes = outs['bbox']
        batch_pred_bboxes_num = outs['bbox_num']
        assert len(batch_pred_bboxes_num) == len(data['gt_bbox'])
        batch_size = len(data['gt_bbox'])
        box_cnt = 0
        for batch_id in range(batch_size):
            pred_bboxes_num = batch_pred_bboxes_num[batch_id]
            pred_bboxes = batch_pred_bboxes[box_cnt: box_cnt +
                                                     pred_bboxes_num].cpu().numpy()
            box_cnt += pred_bboxes_num

            det_conf = pred_bboxes[:, 1]
            det_xmin = pred_bboxes[:, 2]
            det_ymin = pred_bboxes[:, 3]
            det_xmax = pred_bboxes[:, 4]
            det_ymax = pred_bboxes[:, 5]
            det = np.column_stack((det_xmin, det_ymin, det_xmax,
                                   det_ymax, det_conf))
            self.pred_boxes_list.append(det)  # xyxy conf
            self.gt_boxes_list.append(data['gt_ori_bbox'][batch_id].cpu().numpy())  # xywh
            self.hard_ignore_list.append(
                data['gt_hard_ignore'][batch_id].cpu().numpy())
            self.medium_ignore_list.append(
                data['gt_medium_ignore'][batch_id].cpu().numpy())
            self.easy_ignore_list.append(
                data['gt_easy_ignore'][batch_id].cpu().numpy())

    def accumulate(self):
        """
        汇总指标统计值
        """
        total_num = len(self.gt_boxes_list)
        settings = ['easy', 'medium', 'hard']
        setting_ingores = [self.easy_ignore_list,
                           self.medium_ignore_list,
                           self.hard_ignore_list]
        thresh_num = 1000
        aps = []
        for setting_id in range(3):
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype(np.float32)
            gt_ignore_list = setting_ingores[setting_id]
            for i in range(total_num):
                pred_boxes = self.pred_boxes_list[i]  # xyxy conf
                gt_boxes = self.gt_boxes_list[i]  # xywh
                ignore = gt_ignore_list[i]
                count_face += np.sum(ignore)

                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue
                pred_recall, proposal_list = image_eval(pred_boxes, gt_boxes,
                                                        ignore, self.iou_thresh)
                _img_pr_info = img_pr_info(thresh_num, pred_boxes,
                                           proposal_list, pred_recall)
                pr_curve += _img_pr_info
            pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]

            ap = voc_ap(recall, propose)
            aps.append(ap)
        self.aps = aps

    def log(self):
        """
        记录指标结果
        """
        print("==================== Results ====================")
        print("Easy   Val AP: {}".format(self.aps[0]))
        print("Medium Val AP: {}".format(self.aps[1]))
        print("Hard   Val AP: {}".format(self.aps[2]))
        print("=================================================")

    def get_results(self):
        """
        获取指标结果

        Returns:
            指标结果字典
        """
        return {
            'easy_ap': self.aps[0],
            'medium_ap': self.aps[1],
            'hard_ap': self.aps[2]}

    def name(self):
        return self.__class__.__name__


class RBoxMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        """
        旋转边界框指标类

        Args:
            anno_file: 注释文件路径
            **kwargs: 其他参数
        """
        self.anno_file = anno_file
        self.clsid2catid, self.catid2name = get_categories('RBOX', anno_file)
        self.catid2clsid = {v: k for k, v in self.clsid2catid.items()}
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.overlap_thresh = kwargs.get('overlap_thresh', 0.5)
        self.map_type = kwargs.get('map_type', '11point')
        self.evaluate_difficult = kwargs.get('evaluate_difficult', False)
        self.imid2path = kwargs.get('imid2path', None)
        class_num = len(self.catid2name)
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=self.overlap_thresh,
            map_type=self.map_type,
            is_bbox_normalized=False,
            evaluate_difficult=self.evaluate_difficult,
            catid2name=self.catid2name,
            classwise=self.classwise)

        self.reset()

    def reset(self):
        """
        重置指标统计值
        """
        self.results = []
        self.detection_map.reset()

    def update(self, inputs, outputs):
        """
        更新指标统计值

        Args:
            inputs: 输入数据
            outputs: 输出数据
        """
        outs = {}
        # 将输出张量转换为numpy数组
        for k, v in outputs.items():
            outs[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

        im_id = inputs['im_id']
        im_id = im_id.cpu().numpy() if isinstance(im_id, torch.Tensor) else im_id
        outs['im_id'] = im_id

        infer_results = get_infer_results(outs, self.clsid2catid)
        infer_results = infer_results['bbox'] if 'bbox' in infer_results else []
        self.results += infer_results
        if self.save_prediction_only:
            return

        gt_boxes = inputs['gt_poly']
        gt_labels = inputs['gt_class']

        if 'scale_factor' in inputs:
            scale_factor = inputs['scale_factor'].cpu().numpy() if isinstance(
                inputs['scale_factor'],
                torch.Tensor) else inputs['scale_factor']
        else:
            scale_factor = np.ones((gt_boxes.shape[0], 2)).astype('float32')

        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i].cpu().numpy() if isinstance(
                gt_boxes[i], torch.Tensor) else gt_boxes[i]
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h, w, h, w, h])
            gt_label = gt_labels[i].cpu().numpy() if isinstance(
                gt_labels[i], torch.Tensor) else gt_labels[i]
            gt_box, gt_label, _ = prune_zero_padding(gt_box, gt_label)
            bbox = [
                res['bbox'] for res in infer_results
                if int(res['image_id']) == int(im_id[i])
            ]
            score = [
                res['score'] for res in infer_results
                if int(res['image_id']) == int(im_id[i])
            ]
            label = [
                self.catid2clsid[int(res['category_id'])]
                for res in infer_results
                if int(res['image_id']) == int(im_id[i])
            ]
            self.detection_map.update(bbox, score, label, gt_box, gt_label)

    def save_results(self, results, output_dir, imid2path):
        """
        保存结果

        Args:
            results: 检测结果
            output_dir: 输出目录
            imid2path: 图像ID到路径的映射
        """
        if imid2path:
            data_dicts = defaultdict(list)
            for result in results:
                image_id = result['image_id']
                data_dicts[image_id].append(result)

            for image_id, image_path in imid2path.items():
                basename = os.path.splitext(os.path.split(image_path)[-1])[0]
                output = os.path.join(output_dir, "{}.txt".format(basename))
                dets = data_dicts.get(image_id, [])
                with open(output, 'w') as f:
                    for det in dets:
                        catid, bbox, score = det['category_id'], det[
                            'bbox'], det['score']
                        bbox_pred = '{} {} '.format(self.catid2name[catid],
                                                    score) + ' '.join(
                            [str(e) for e in bbox])
                        f.write(bbox_pred + '\n')

            print('The bbox result is saved to {}.'.format(output_dir))
        else:
            output = os.path.join(output_dir, "bbox.json")
            with open(output, 'w') as f:
                json.dump(results, f)

            print('The bbox result is saved to {}.'.format(output))

    def accumulate(self):
        """
        汇总指标统计值
        """
        if self.output_eval:
            self.save_results(self.results, self.output_eval, self.imid2path)

        if not self.save_prediction_only:
            print("Accumulating evaluatation results...")
            self.detection_map.accumulate()

    def log(self):
        """
        记录指标结果
        """
        map_stat = 100. * self.detection_map.get_map()
        print("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                 self.map_type, map_stat))

    def get_results(self):
        """
        获取指标结果

        Returns:
            指标结果字典
        """
        return {'bbox': [self.detection_map.get_map()]}

    def name(self):
        return self.__class__.__name__


class SNIPERCOCOMetric(COCOMetric):
    def __init__(self, anno_file, **kwargs):
        """
        SNIPER COCO指标类

        Args:
            anno_file: 注释文件路径
            **kwargs: 其他参数
        """
        super(SNIPERCOCOMetric, self).__init__(anno_file, **kwargs)
        self.dataset = kwargs["dataset"]
        self.chip_results = []

    def reset(self):
        """
        重置指标统计值
        """
        # 目前仅支持bbox和mask评估
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}
        self.chip_results = []

    def update(self, inputs, outputs):
        """
        更新指标统计值

        Args:
            inputs: 输入数据
            outputs: 输出数据
        """
        outs = {}
        # 将输出张量转换为numpy数组
        for k, v in outputs.items():
            outs[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.cpu().numpy() if isinstance(im_id,
                                                          torch.Tensor) else im_id

        self.chip_results.append(outs)

    def accumulate(self):
        """
        汇总指标统计值
        """
        results = self.dataset.anno_cropper.aggregate_chips_detections(
            self.chip_results)
        for outs in results:
            infer_results = get_infer_results(
                outs, self.clsid2catid, bias=self.bias)
            self.results['bbox'] += infer_results[
                'bbox'] if 'bbox' in infer_results else []

        super(SNIPERCOCOMetric, self).accumulate()