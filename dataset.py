#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from clip import *
import glob

class listDataset(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset_use='ucf101-24', shape=None, shuffle=True,
                 transform=None, target_transform=None, 
                 train=False, seen=0, batch_size=64,
                 clip_duration=16, num_workers=4):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.base_path = base
        self.dataset_use = dataset_use
        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.clip_duration = clip_duration
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        self.shape = (224, 224)

        if self.train: # For Training
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            clip, label = load_data_detection(self.base_path, imgpath,  self.train, self.clip_duration, self.shape, self.dataset_use, jitter, hue, saturation, exposure)

        else: # For Testing
            frame_idx, clip, label = load_data_detection(self.base_path, imgpath, False, self.clip_duration, self.shape, self.dataset_use)
            clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        if self.train:
            return (clip, label)
        else:
            return (frame_idx, clip, label)

class testData(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, root, shape=None, shuffle=False,
                 transform=None, target_transform=None,
                 train=False, seen=0, batch_size=64,
                 clip_duration=16, num_workers=4):
       self.root = root
       self.imgpaths = sorted(glob.glob(os.path.join(root, '*.jpg')))

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.imgpaths)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.clip_duration = clip_duration
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.imgpaths[index]

        clip,label = load_data_detection_test(self.root, imgpath, self.clip_duration, self.nSamples)
        clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return clip,label
