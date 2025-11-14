#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :dataset.py
@Author :CodeCat
@Date   :2025/11/4 10:16
"""
import os
import copy
import numpy as np
from collections.abc import Sequence
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from loguru import logger


class DetDataset(Dataset):
    """
    初始化 DetDataset 类。

    Args:
        dataset_dir (str, optional): 数据集目录路径。默认为 None，如果为 None，则使用空字符串。
        image_dir (str, optional): 图像文件目录路径。默认为 None，如果为 None，则使用空字符串。
        anno_path (str, optional): 标注文件路径。默认为 None。
        data_fields (list, optional): 需要加载的数据字段列表。默认为 ['image']。
        sample_num (int, optional): 需要加载的样本数量。默认为 -1，表示加载全部样本。
        use_default_label (bool, optional): 是否使用默认标签。默认为 None。
        repeat (int, optional): 数据集重复次数。默认为 1。
        **kwargs: 其他关键字参数。
    """
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 repeat=1,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self.repeat = repeat
        self._epoch = 0
        self._curr_iter = 0

    def __len__(self):
        # return the number of samples in the dataset, roidbs is a list of dicts from parse_dataset()
        return len(self.roidbs) * self.repeat

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        """
        根据给定的索引idx获取相应的roidb数据。
        Args:
            idx (int): 索引值。
        Returns:
            roidb (dict or list of dict): 对应的roidb数据。
        """
        n = len(self.roidbs)
        if self.repeat > 1:
            idx = idx % n
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            # Perform mosaic stitching on the first 4 images, and then perform mixup(optional) with the last one
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)]) for _ in range(4)
            ]
        elif self.pre_img_epoch == 0 or self._epoch < self.pre_img_epoch:
            # Add previous image as input, only used in CenterTrack
            idx_pre_img = idx - 1
            if idx_pre_img < 0:
                idx_pre_img = idx + 1
            roidb = [roidb, copy.deepcopy(self.roidbs[idx_pre_img])]

        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
                r['curr_epoch'] = self._epoch
        else:
            roidb['curr_iter'] = self._curr_iter
            roidb['curr_epoch'] = self._epoch

        self._curr_iter += 1

        if self.transform_schedulers:
            assert isinstance(self.transform_schedulers, list)
            if isinstance(roidb, Sequence):
                for r in roidb:
                    r['transform_schedulers'] = self.transform_schedulers
            else:
                roidb['transform_schedulers'] = self.transform_schedulers

        return self.transform(roidb)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)
        self.pre_img_epoch = kwargs.get('pre_img_epoch', -1)
        self.transform_schedulers = kwargs.get('transform_schedulers', None)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self):
        raise NotImplementedError(
            "Need to implement parse_dataset method in the subclass."
        )

    def get_anno(self):
        if self.anno_path is None:
            return None
        return os.path.join(self.dataset_dir, self.anno_path)


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)


def _make_dataset(dir):
    """
    从指定目录中读取图像文件，并返回一个包含所有图像文件路径的列表。

    Args:
        dir (str): 包含图像文件的目录路径。

    Returns:
        list: 包含所有图像文件路径的列表。

    Raises:
        ValueError: 如果传入的目录路径不是一个有效的目录，则抛出此异常。
    """
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} is not a valid directory')
    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if _is_valid_file(path):
                images.append(path)
    return images


class ImageFolder(DetDataset):
    """
    Image folder dataset for detection.
     Args:
        dataset_dir (str, optional): 数据集目录. 默认值为None.
        image_dir (str, optional): 图像文件所在的目录. 默认值为None.
        anno_path (str, optional): 标注文件的路径. 默认值为None.
        sample_num (int, optional): 要采样的样本数量. 默认值为-1, 表示使用所有样本.
        use_default_label (bool, optional): 是否使用默认标签. 默认值为None.
        **kwargs: 其他关键字参数.
    """
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(ImageFolder, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            sample_num=sample_num,
            use_default_label=use_default_label)
        self._imid2path = {}
        self.roidbs = None
        self.sample_num = sample_num

    def get_anno(self):
        if self.anno_path is None:
            return None
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, self.anno_path)
        else:
            return self.anno_path

    def parse_dataset(self):
        if not self.roidbs:
            self.roidbs = self._load_images()

    def _parse(self):
        """
        解析图像目录或文件，并返回图像文件列表。
        Args:
            无
        Returns:
            list: 包含图像文件路径的列表。
        """
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def get_images(self):
        """
        获取数据集中所有图片的路径列表。
        Args:
            None
        Returns:
            images_path (list of str): 包含数据集中所有图片路径的列表。
        """
        images_path = []
        coco = COCO(os.path.join(self.dataset_dir, self.anno_path))
        imgIds = coco.getImgIds(catIds=[])
        for imgId in imgIds:
            filename = coco.loadImgs(imgId)[0]['file_name']
            images_path.append(os.path.join(self.dataset_dir, self.image_dir, filename))
        return images_path

    def _load_images(self, do_eval=False):
        """
        加载图像数据。
        Args:
            do_eval (bool): 是否处于评估模式。默认为 False。
        Returns:
            list: 包含图像记录的列表，每个记录包含图像的ID和文件路径。
        Raises:
            AssertionError: 如果在指定的目录中未找到任何图像文件，或者某个图像文件不存在。
        """
        images = self._parse()
        ct = 0
        records = []
        anno_file = self.get_anno()
        coco = COCO(anno_file)
        for image in images:
            assert image != '' and os.path.isfile(image), \
                "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            if do_eval:
                image_id = self.get_image_id(image, coco)
                ct = image_id
            rec = {
                'im_id': np.array([ct]),
                'im_file': image
            }
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No images found in {}".format(self.image_dir)
        return records

    def get_image_id(self, image, coco):
        """
        根据给定的图像名称和COCO数据集对象获取图像ID。
        Args:
            image (str): 图像的名称。
            coco (COCO): COCO数据集对象。
        Returns:
            int: 图像ID。如果未找到图像，则返回None。
        """
        image_ids = coco.getImgIds()
        for image_id in image_ids:
            img_info = coco.loadImgs(image_id)[0]
            if img_info['file_name'] in image:
                return image_id
            else:
                continue

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images, do_eval=False):
        self.image_dir = images
        self.roidbs = self._load_images(do_eval=do_eval)

    def get_slice_images(self,
                         images,
                         slice_size=[640, 640],
                         overlap_ratio=[0.25, 0.25]):
        """
        将输入的图像切片成指定大小和重叠率的子图像。
        Args:
            images (str): 存储图像的目录路径。
            slice_size (list): 子图像的大小，格式为 [高度, 宽度]，默认为 [640, 640]。
            overlap_ratio (list): 子图像之间的重叠率，格式为 [高度重叠率, 宽度重叠率]，默认为 [0.25, 0.25]。
        Returns:
            None
        Raises:
            Exception: 如果未安装sahi库，则会引发异常。
        """
        self.image_dir = images
        ori_records = self._load_images()
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            logger.error(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e

        sub_img_ids = 0
        ct = 0
        ct_sub = 0
        records = []
        for i, ori_rec in enumerate(ori_records):
            im_path = ori_rec['im_file']
            slice_image_result = sahi.slicing.slice_image(
                image_path=im_path,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1],
            )
            sub_img_num = len(slice_image_result)
            for _ind in range(sub_img_num):
                im = slice_image_result.images[_ind]
                rec = {
                    'image': im,
                    'im_id': np.array([sub_img_ids + _ind]),
                    'h': im.shape[0],
                    'w': im.shape[1],
                    'ori_im_id': np.array(ori_rec['im_id'][0]),
                    'st_pix': np.array(
                        slice_image_result.starting_pixels[_ind], dtype=np.float32),
                    'is_last': 1 if _ind == sub_img_num - 1 else 0
                } if 'image' in self.data_fields else {}
                records.append(rec)
            ct_sub += sub_img_num
            ct += 1
            sub_img_ids += sub_img_num

        logger.info(f'{ct} samples and slice to {ct_sub} sub_samples.')
        self.roidbs = records

    def get_label_list(self):
        # Only VOC dataset need label list in ImageFolder
        return self.anno_path


class CommonDataset(object):
    def __init__(self, **dataset_args):
        super(CommonDataset, self).__init__()
        dataset_args = copy.deepcopy(dataset_args)
        type = dataset_args.pop("name")
        self.dataset = getattr(source, type)(**dataset_args)

    def __call__(self):
        return self.dataset


class TrainDataset(CommonDataset):
    pass


class EvalMOTDataset(CommonDataset):
    pass


class TestMOTDataset(CommonDataset):
    pass


class EvalDataset(CommonDataset):
    pass


class TestDataset(CommonDataset):
    pass
