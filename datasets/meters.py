#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import time
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
import json

from datasets import logging
from datasets import ava_helper
from datasets.ava_eval_helper import (
    run_evaluation,
    read_csv,
    read_exclusions,
    read_labelmap,
    write_results
)

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    def __init__(self, cfg, mode, output_json):
        self.cfg = cfg
        self.all_preds = []
        self.mode = mode
        self.output_json = os.path.join(self.cfg.BACKUP_DIR, output_json)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)
        _, self.video_idx_to_name = ava_helper.load_image_lists(cfg, self.mode == 'train')

    def update_stats(self, preds):
        self.all_preds.extend(preds)

    def evaluate_ava(self):
        eval_start = time.time()
        detections = self.get_ava_eval_data()
        if self.mode == 'test' or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth
        logger.info("Evaluating with %d unique GT frames." % len(groundtruth[0]))
        logger.info("Evaluating with %d unique detection frames" % len(detections[0]))

        name = "latest"
        write_results(detections, os.path.join(self.cfg.BACKUP_DIR, "detections_%s.csv" % name))
        write_results(groundtruth, os.path.join(self.cfg.BACKUP_DIR, "groundtruth_%s.csv" % name))
        results = run_evaluation(self.categories, groundtruth, detections, self.excluded_keys)
        with open(self.output_json, 'w') as fp:
            json.dump(results, fp)
        logger.info("Save eval results in {}".format(self.output_json))

        logger.info("AVA eval done in %f seconds." % (time.time() - eval_start))

        return results["PascalBoxes_Precision/mAP@0.5IOU"]

    def get_ava_eval_data(self):
        out_scores = defaultdict(list)
        out_labels = defaultdict(list)
        out_boxes = defaultdict(list)
        count = 0

        # each pred is [[x1, y1, x2, y2], [scores], [video_idx, src]]
        for i in range(len(self.all_preds)):
            pred = self.all_preds[i]
            assert len(pred) == 3
            video_idx = int(np.round(pred[-1][0]))
            sec = int(np.round(pred[-1][1]))
            box = pred[0]
            scores = pred[1]
            assert len(scores) == 80
            # try:
            #     assert len(scores) == len(labels)
            # except TypeError:
            #     pdb.set_trace()

            video = self.video_idx_to_name[video_idx]
            key = video + ',' + "%04d" % (sec)
            box = [box[1], box[0], box[3], box[2]]  # turn to y1,x1,y2,x2

            for cls_idx, score in enumerate(scores):
                if cls_idx + 1 in self.class_whitelist:
                    out_scores[key].append(score)
                    out_labels[key].append(cls_idx + 1)
                    out_boxes[key].append(box)
                    count += 1

        return out_boxes, out_labels, out_scores


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
