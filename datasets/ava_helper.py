#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import pdb

from collections import defaultdict
from fvcore.common.file_io import PathManager
from datasets.ava_eval_helper import read_exclusions

logger = logging.getLogger(__name__)
#FPS = 20
#AVA_VALID_FRAMES = range(902, 1799)
AVA_VALID_FRAMES = range(1, 201)

def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    # frame_list_dir is /data3/ava/frame_lists/
    # contains 'train.csv' and 'val.csv'
    list_filenames = [os.path.join(cfg.AVA.FRAME_LIST_DIR, filename) for filename in (cfg.AVA.TRAIN_LISTS if is_train else cfg.AVA.TEST_LISTS)]
    pdb.set_trace()
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with PathManager.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(
                    os.path.join(cfg.AVA.FRAME_DIR, row[3])
                )

    image_paths = [image_paths[i] for i in range(len(image_paths))]

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )

    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test    ' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    if cfg.TRAIN.USE_SLOWFAST:
        gt_filename = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == 'train' else cfg.AVA.TEST_PREDICT_BOX_LISTS
    else:
        gt_filename = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == 'train' else cfg.AVA.VAL_GT_BOX_LISTS
    
    # ann_filename = /home/bill/datasets/dummy_action/annotations/ava_train_v2.2.csv
    ann_filename = os.path.join(cfg.AVA.ANNOTATION_DIR, gt_filename[0])
    all_boxes = {}
    count = 0
    unique_box_count = 0
    
    # Commented out the exclusions

    #if mode == 'train':
    #    excluded_keys = read_exclusions(
    #        os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.TRAIN_EXCLUSION_FILE)
    #    )
    #else:
    #    excluded_keys = read_exclusions(
    #        os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
    #    )
    excluded_keys = []
    
    detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH

    with PathManager.open(ann_filename, 'r') as f:
        for line in f:
            row = line.strip().split(',')

            # use same detection as slowfast
            if cfg.TRAIN.USE_SLOWFAST:
                if mode == 'val' or mode == 'test':
                    score = float(row[7])
                    if score < detect_thresh:
                        continue
            ########

            video_name, frame_sec = row[0], int(row[1])
            key = "%s,%04d" % (video_name, frame_sec)
            # if mode == 'train' and key in excluded_keys:
            if key in excluded_keys:
                print("Found {} to be excluded...".format(key))
                continue

            # Only select frame_sec % 4 = 0 samples for validation if not
            # set FULL_TEST_ON_VAL (default False)
            if mode == 'val' and not cfg.AVA.FULL_TEST_ON_VAL and frame_sec % 4 != 0:
                continue
            # Box with [x1, y1, x2, y2] with a range of [0, 1] as float
            box_key = ",".join(row[2:6])
            box = list(map(float, row[2:6]))
            # Action label
            label = -1 if row[6] == "" else int(row[6])
            
            if video_name not in all_boxes:
                all_boxes[video_name] = {}
                for sec in AVA_VALID_FRAMES:
                    all_boxes[video_name][sec] = {}

            if box_key not in all_boxes[video_name][frame_sec]:
                all_boxes[video_name][frame_sec][box_key] = [box, []]
                unique_box_count += 1

            all_boxes[video_name][frame_sec][box_key][1].append(label)
            if label != -1:
                count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    logger.info(
        "Finished loading annotations from: %s" % ", ".join([ann_filename])
    )
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    # all boxes -> 
    # {'side_near': {1: [[[0.142361, 0.552726, 0.0478395, 0.172325], [3]], [[0.342786, 0.312243, 0.0289352, 0.106996], [0]], [[0.47936, 0.258745, 0.0227623, 0.0771605], [2]]]}
    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        #return (sec - 900) * FPS
        return sec

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            if sec not in AVA_VALID_FRAMES:
                continue

            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec_to_frame(sec))
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def get_max_objs(keyframe_indices, keyframe_boxes_and_labels):
    # max_objs = 0
    # for video_idx, sec_idx, _, _ in keyframe_indices:
    #     num_boxes = len(keyframe_boxes_and_labels[video_idx][sec_idx])
    #     if num_boxes > max_objs:
    #         max_objs = num_boxes

    # return max_objs
    return 50 #### MODIFICATION FOR NOW! TODO: FIX LATER!
