#!/usr/bin/python
# encoding: utf-8

import os
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.clip import *


class UCF_JHMDB_Dataset(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset='ucf24', shape=None,
                 transform=None, target_transform=None, 
                 train=False, clip_duration=16, sampling_rate=1):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base
        self.dataset = dataset
        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.clip_duration = clip_duration
        self.sampling_rate = sampling_rate

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train: # For Training
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            clip, label = load_data_detection(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.shape, self.dataset, jitter, hue, saturation, exposure)

        else: # For Testing
            frame_idx, clip, label = load_data_detection(self.base_path, imgpath, False, self.clip_duration, self.sampling_rate, self.shape, self.dataset)
            clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (clip, label)
        else:
            return (frame_idx, clip, label)