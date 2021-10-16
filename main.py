from __future__ import print_function
import os
import sys
import time
import math
import random
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets.ava_dataset import Ava 
from core.optimization import *
from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss_Ava
from core.model import YOWO, get_fine_tuning_parameters


# Load configuration arguments
# Here we use cfg/parser.py which uses defaults.py
args  = parser.parse_args()
cfg   = parser.load_config(args)

# Check backup directory, create if necessary
if not os.path.exists(cfg.BACKUP_DIR):
    os.makedirs(cfg.BACKUP_DIR)

# Create model
# Here we use core/model.py
model = YOWO(cfg)
model = model.cuda()
model = nn.DataParallel(model, device_ids=None) # in multi-gpu case
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging('Total number of trainable parameters: {}'.format(pytorch_total_params))

# Set seed, use cuda
seed = int(time.time())
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

# Define optimizer
parameters = get_fine_tuning_parameters(model, cfg)
optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
# optimizer = optim.SGD(parameters, lr=cfg.TRAIN.LEARNING_RATE/batch_size, momentum=cfg.SOLVER.MOMENTUM, dampening=0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
best_score   = 0 # initialize best score

# Load resume path if necessary
if cfg.TRAIN.RESUME_PATH:
    print("===================================================================")
    print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch'] + 1
    try:
        best_score = checkpoint['score']
    except KeyError:
        best_score = checkpoint['fscore']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded model score: ", best_score)
    print("===================================================================")
    del checkpoint

# Create backup directory if necessary
if not os.path.exists(cfg.BACKUP_DIR):
    os.mkdir(cfg.BACKUP_DIR)

# Data loader, training scheme and loss function for AVA
dataset = cfg.TRAIN.DATASET

# Make sure the correct dataset is chosen
assert dataset == 'ava', 'invalid dataset'

# Set dataset
# Here we use datasets/ava_dataset which uses ava_helper, cv2_transform, dataset_utils
train_dataset = Ava(cfg, split='train')
test_dataset  = Ava(cfg, split='val')

# For debugging try train_dataset[0]
#pdb.set_trace()

# Set trainloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, 
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=True, pin_memory=True)

test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)

# Set loss function
# Here we use core/region_loss.py which uses utils
loss_module   = RegionLoss_Ava(cfg).cuda()

# Import train and test functions
# Import using getattr (no reason for this...)
train = getattr(sys.modules[__name__], 'train_ava')
test  = getattr(sys.modules[__name__], 'test_ava')
# This is the same as
from core.optimization import train_ava as train
from core.optimization import test_ava as test

# Training and Testing Schedule
if cfg.TRAIN.EVALUATE:
    logging('evaluating ...')
    test(cfg, 0, model, test_loader)
else:
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH + 1):
        # Adjust learning rate
        lr_new = adjust_learning_rate(optimizer, epoch, cfg)
        
        # Train and test model
        logging('training at epoch %d, lr %f' % (epoch, lr_new))
        train(cfg, epoch, model, train_loader, loss_module, optimizer)
        logging('testing at epoch %d' % (epoch))
        score = test(cfg, epoch, model, test_loader)

        # Save the model to backup directory
        is_best = score > best_score
        if is_best:
            print("New best score is achieved: ", score)
            print("Previous score was: ", best_score)
            best_score = score

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'score': score
            }
        save_checkpoint(state, is_best, cfg.BACKUP_DIR, cfg.TRAIN.DATASET, cfg.DATA.NUM_FRAMES)
        logging('Weights are saved to backup directory: %s' % (cfg.BACKUP_DIR))