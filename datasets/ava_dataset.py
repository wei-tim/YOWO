#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import random
import os
import pdb

import torch
import numpy as np
import cv2

from datasets import ava_helper, cv2_transform
from datasets.dataset_utils import retry_load_images, get_frame_idx, get_sequence
from datasets import image

logger = logging.getLogger(__name__)


class Ava(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        # Basic params
        self.cfg = cfg
        self._split = split
        self._downsample = 4
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        # CAUTION: Remember to change this
        self.n_classes = cfg.MODEL.NUM_CLASSES
        
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]

        self._load_data(cfg)

    # _load_data is working fine and returns everythin (images and boxes and labels)
    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        self._image_paths, self._video_idx_to_name = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))
        
        # Loading annotations for boxes and labels.
        # boxes_and_labels: {'<video_name>': {<frame_num>: a list of [box_i, box_i_labels]} }
        boxes_and_labels = ava_helper.load_boxes_and_labels(cfg, mode=self._split)

        assert len(boxes_and_labels) == len(self._image_paths)

        # boxes_and_labels: a list of {<frame_num>: a list of [box_i, box_i_labels]}
        boxes_and_labels = [boxes_and_labels[self._video_idx_to_name[i]]for i in range(len(self._image_paths))]

        # Get indices of keyframes and corresponding boxes and labels.
        # _keyframe_indices: [video_idx, sec_idx, sec, frame_index]
        # _keyframe_boxes_and_labels: list[list[list]], outer is video_idx, middle is sec_idx,
        # inner is a list of [box_i, box_i_labels]
        self._keyframe_indices, self._keyframe_boxes_and_labels = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(self._keyframe_indices, self._keyframe_boxes_and_labels)

        self._max_objs = ava_helper.get_max_objs(self._keyframe_indices, self._keyframe_boxes_and_labels)

        self.print_summary()
        
    def print_summary(self):
        logger.info("=== MBARv1 dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(len(video_img_paths) for video_img_paths in self._image_paths)
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Here we still have the original image shape (e.g. 1944, 2592)
        # We also have boxes values in [0,1]
        height, width, _ = imgs[0].shape

        # Here we multiply boxes with shapes
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            ''' label augmentation '''
            boxes = cv2_transform.resize_boxes(self._crop_size, boxes, imgs[0].shape[0], imgs[0].shape[1])
            imgs = [cv2_transform.resize(self._crop_size, img) for img in imgs]

            #if self.random_horizontal_flip:
            #    imgs, boxes = cv2_transform.horizontal_flip_list(0.5, imgs, order="HWC", boxes=boxes)
        
        elif self._split == "val":
            imgs = [cv2_transform.resize(self._crop_size, img) for img in imgs]
            boxes = cv2_transform.resize_boxes(self._crop_size, boxes, height, width)

            #if self._test_force_flip:
            #    imgs, boxes = cv2_transform.horizontal_flip_list(1, imgs, order="HWC", boxes=boxes)

        elif self._split == "test":  # need modified
            imgs = [cv2_transform.resize(self._crop_size, img) for img in imgs]
            boxes = cv2_transform.resize_boxes(self._crop_size, boxes, height, width)
            
            #if self._test_force_flip:
            #    imgs, boxes = cv2_transform.horizontal_flip_list(1, imgs, order="HWC", boxes=boxes)

        else:
            raise NotImplementedError("Unsupported split mode {}".format(self._split))
            
        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        #imgs = [
        #    cv2_transform.color_normalization(
        #        img,
        #        np.array(self._data_mean, dtype=np.float32),
        #        np.array(self._data_std, dtype=np.float32),
        #    )
        #    for img in imgs
        #]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        bx_count = boxes[0].shape[0]
        
        #print('---1---', boxes)
        boxes = cv2_transform.clip_boxes_to_image(boxes[0], imgs[0].shape[1], imgs[0].shape[2])
        #print('---2---', boxes)
        boxes = cv2_transform.transform_cxcywh(boxes, imgs[0].shape[1], imgs[0].shape[2])
        #print('---3---', boxes)
        
        assert bx_count == boxes.shape[0]
        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        # Get the frame idxs for current clip. We can use it as center or latest
        video_idx, sec_idx, sec, frame_idx = self._keyframe_indices[idx]
        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]

        assert self.cfg.AVA.IMG_PROC_BACKEND != 'pytorch'
        sample_rate = self._sample_rate
        seq_len = self._video_length * sample_rate
        seq = get_sequence(frame_idx, seq_len // 2, sample_rate, num_frames=len(self._image_paths[video_idx]))

        image_paths = [self._image_paths[video_idx][frame - 1] for frame in seq]
        imgs = retry_load_images(image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND)
        assert len(clip_label_list) > 0
        assert len(clip_label_list) <= self._max_objs
        num_objs = len(clip_label_list)
        #keyframe_info = self._image_paths[video_idx][frame_idx - 1]
        src_height, src_width = imgs[0].shape[0], imgs[0].shape[1]

        # Get boxes and labels for current clip.
        # So, it looks like each clip (e.g. 8 frames) uses the boxes of just ONE frame
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        ori_boxes = boxes.copy()

        imgs, boxes = self._images_and_boxes_preprocessing_cv2(imgs, boxes=boxes)
        inp_height, inp_width = imgs.size(2), imgs.size(3)

        assert boxes.shape[1] == 4
        assert num_objs == len(labels)
        assert boxes.shape[0] <= self._max_objs
        assert boxes.shape[0] == len(labels)

        ret_cls   = np.zeros((self._max_objs, self.n_classes), dtype=np.float32)
        ret_boxes = np.zeros((self._max_objs, 4), dtype=np.float32)
        for i in range(num_objs):
            ret_boxes[i, :] = boxes[i]
            label = np.array(labels[i])
            assert len(label) > 0, 'Fatal Error'
            for cls_ind in label:
                    ret_cls[i, cls_ind - 1] = 1

        ret = {'clip': imgs, 'cls': ret_cls, 'boxes': ret_boxes, 'metadata':np.array([video_idx, sec, src_width, src_height])}

        return ret