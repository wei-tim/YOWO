import os
import cv2
import sys
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets import list_dataset, cv2_transform
from datasets.ava_dataset import Ava 
from datasets.ava_eval_helper import read_labelmap 
from datasets.meters import AVAMeter
from core.optimization import *
from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss, RegionLoss_Ava
from core.model import YOWO, get_fine_tuning_parameters



####### Load configuration arguments
# ---------------------------------------------------------------
args  = parser.parse_args()
cfg   = parser.load_config(args)


####### Create model
# ---------------------------------------------------------------
model = YOWO(cfg)
model = model.cuda()
model = nn.DataParallel(model, device_ids=None) # in multi-gpu case


####### Load resume path if necessary
# ---------------------------------------------------------------
if cfg.TRAIN.RESUME_PATH:
    print("===================================================================")
    print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch'] + 1
    best_score = checkpoint['score']
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded model score: ", checkpoint['score'])
    print("===================================================================")
    del checkpoint


####### Test parameters
# ---------------------------------------------------------------

labelmap, _       = read_labelmap("/usr/home/sut/datasets/AVA/annotations/ava_action_list_v2.2.pbtxt")
num_classes       = cfg.MODEL.NUM_CLASSES
clip_length		  = cfg.DATA.NUM_FRAMES
crop_size 		  = cfg.DATA.TEST_CROP_SIZE
anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
num_anchors       = cfg.SOLVER.NUM_ANCHORS
nms_thresh        = 0.5
conf_thresh_valid = 0.5 # For more stable results, this threshold is increased!

meter = AVAMeter(cfg, cfg.TRAIN.MODE, 'latest_detection.json')

model.eval()

# 9Y_l9NsnYE0.mp4
# CMCPhm2L400.mkv
# CZ2NP8UsPuE.mkv
# KVq6If6ozMY.mkv

####### Data preparation and inference 
# ---------------------------------------------------------------
video_path = '/usr/home/sut/datasets/AVA/video_done/9Y_l9NsnYE0.mp4'
cap = cv2.VideoCapture(video_path)

cnt = 1
queue = []
while(cap.isOpened()):
    ret, frame = cap.read()

    if len(queue) <= 0: # At initialization, populate queue with initial frame
    	for i in range(clip_length):
    		queue.append(frame)

    # Add the read frame to last and pop out the oldest one
    queue.append(frame)
    queue.pop(0)

    # Resize images
    imgs = [cv2_transform.resize(crop_size, img) for img in queue]
    frame = img = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    # Convert image to CHW keeping BGR order.
    imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

    # Image [0, 255] -> [0, 1].
    imgs = [img / 255.0 for img in imgs]

    imgs = [
        np.ascontiguousarray(
            img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
        ).astype(np.float32)
        for img in imgs
    ]

    # Normalize images by mean and std.
    imgs = [
        cv2_transform.color_normalization(
            img,
            np.array(cfg.DATA.MEAN, dtype=np.float32),
            np.array(cfg.DATA.STD, dtype=np.float32),
        )
        for img in imgs
    ]

    # Concat list of images to single ndarray.
    imgs = np.concatenate(
        [np.expand_dims(img, axis=1) for img in imgs], axis=1
    )

    imgs = np.ascontiguousarray(imgs)
    imgs = torch.from_numpy(imgs)
    imgs = torch.unsqueeze(imgs, 0)


    # Model inference
    with torch.no_grad():
        output = model(imgs)

        preds = []
        all_boxes = get_region_boxes_ava(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            
            for box in boxes:
                x1 = float(box[0]-box[2]/2.0)
                y1 = float(box[1]-box[3]/2.0)
                x2 = float(box[0]+box[2]/2.0)
                y2 = float(box[1]+box[3]/2.0)
                det_conf = float(box[4])
                cls_out = [det_conf * x.cpu().numpy() for x in box[5]]
                preds.append([[x1,y1,x2,y2], cls_out])

    # for line in preds:
    # 	print(line)
    for dets in preds:
        x1 = int(dets[0][0] * crop_size)
        y1 = int(dets[0][1] * crop_size)
        x2 = int(dets[0][2] * crop_size)
        y2 = int(dets[0][3] * crop_size) 
        cls_scores = np.array(dets[1])
        indices = np.where(cls_scores>0.4)
        scores = cls_scores[indices]
        indices = list(indices[0])
        scores = list(scores)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        if len(scores) > 0:
            blk   = np.zeros(frame.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []
            # scores, indices  = [list(a) for a in zip(*sorted(zip(scores,indices), reverse=True))] # if you want, you can sort according to confidence level
            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(labelmap[cls_ind]['name']))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                coord.append((x1+3, y1+7+10*_))
                cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)



    cv2.imshow('frame',frame)
    # cv2.imwrite('{:05d}.jpg'.format(cnt), frame) # save figures if necessay
    cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
