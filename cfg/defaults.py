#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
#_C.BN.NUM_BATCHES_PRECISE = 200
_C.BN.NUM_BATCHES_PRECISE = 25

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SplitBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "ava"

# Total mini-batch size.
#_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.BATCH_SIZE = 10

# Total batch size to accumulate loss
#_C.TRAIN.TOTAL_BATCH_SIZE = 128
_C.TRAIN.TOTAL_BATCH_SIZE = 20

_C.TRAIN.LEARNING_RATE = 1e-4

_C.TRAIN.RESUME_PATH = ""
_C.TRAIN.DETECT_PATH = ""
_C.TRAIN.ONLY_DETECT = False

_C.TRAIN.EVALUATE = False
_C.TRAIN.BEGIN_EPOCH = 1
_C.TRAIN.END_EPOCH = 6
# use groundtruth for validation
_C.TRAIN.USE_GROUNDTRUTH = False
# use slowfast detection for validation
_C.TRAIN.USE_SLOWFAST = False
# finetune or not (for jhmdb)
_C.TRAIN.FINE_TUNE = False

# train mode: `train`, `val` or `test`
_C.TRAIN.MODE = "train"

# categories_ratio file for loss
_C.TRAIN.CLASS_RATIO_FILE = "cfg/categories_ratio.json"

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Dissected version of ResNet
_C.RESNET.DISSECTED = False

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"
_C.MODEL.BACKBONE_2D = "dla34"  # [`res_dcn18`, `dla34`]
_C.MODEL.BACKBONE_3D = "dla34"  # [`res_dcn18`, `dla34`]


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256  # original is 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 20

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# If True, load frame info
_C.DATA.DEBUG = True


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Instead if LRS, LR decay can be set
_C.SOLVER.LR_DECAY_RATE = 0.1

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Anchors added
_C.SOLVER.ANCHORS = [0.70458, 1.18803, 1.26654, 2.55121, 1.59382, 4.08321, 2.30548, 4.94180, 3.52332, 5.91979]

# The number of anchors
_C.SOLVER.NUM_ANCHORS = 5

# Scales for different parts of loss
_C.SOLVER.OBJECT_SCALE = 5
_C.SOLVER.NOOBJECT_SCALE = 1
_C.SOLVER.CLASS_SCALE = 1
_C.SOLVER.COORD_SCALE = 1

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

_C.WEIGHTS = CfgNode()
_C.WEIGHTS.BACKBONE_3D = ""
_C.WEIGHTS.BACKBONE_2D = ""
_C.WEIGHTS.FREEZE_BACKBONE_3D = False
_C.WEIGHTS.FREEZE_BACKBONE_2D = False


# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# output path
_C.DEMO.OUT_PATH = ""

# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
_C.DEMO.LABEL_FILE_PATH = ""

# Specify a camera device as input. This will be prioritized
# over input video if set.
# If -1, use input video instead.
_C.DEMO.WEBCAM = -1

# Path to input video for demo.
_C.DEMO.INPUT_VIDEO = ""
# Custom width for reading input video data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input video data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Path to Detectron2 object detection model configuration,
# only used for detection tasks.
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# Path to Detectron2 object detection model pre-trained weights.
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# Threshold for choosing predicted bounding boxes by Detectron2.
_C.DEMO.DETECTRON2_THRESH = 0.9
# Number of overlapping frames between 2 consecutive clips.
# Increase this number for more frequent action predictions.
# The number of overlapping frames cannot be larger than
# half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
_C.DEMO.BUFFER_SIZE = 0
# If specified, the visualized outputs will be written this a video file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output video file.
# If not set (-1), use fps rate from input file.
_C.DEMO.OUTPUT_FPS = -1
# Input format from demo video reader ("RGB" or "BGR").
_C.DEMO.INPUT_FORMAT = "BGR"
# Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
_C.DEMO.CLIP_VIS_SIZE = 10
# Number of processes to run video visualizer.
_C.DEMO.NUM_VIS_INSTANCES = 2

# Path to pre-computed predicted boxes
_C.DEMO.PREDS_BOXES = ""
# Whether to run in with multi-threaded video reader.
_C.DEMO.THREAD_ENABLE = False
# Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
# This is used for fast demo speed by reducing the prediction/visualiztion frequency.
# If -1, take the most recent read clip for visualization. This mode is only supported
# if `DEMO.THREAD_ENABLE` is set to True.
_C.DEMO.NUM_CLIPS_SKIP = 0
# Path to ground-truth boxes and labels (optional)
_C.DEMO.GT_BOXES = ""
# The starting second of the video w.r.t bounding boxes file.
_C.DEMO.STARTING_SECOND = 900
# Frames per second of the input video/folder of images.
_C.DEMO.FPS = 30
# Visualize with top-k predictions or predictions above certain threshold(s).
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.
_C.DEMO.COMMON_CLASS_THRES = 0.7
# Theshold for uncommon class names. This will not be
# used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# This is chosen based on distribution of examples in
# each classes in AVA dataset.
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# Slow-motion rate for the visualization. The visualized portions of the
# video will be played `_C.DEMO.SLOWMO` times slower than usual speed.
_C.DEMO.SLOWMO = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# backup dir
_C.BACKUP_DIR = "backup"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# Directory path of frames.
_C.AVA.FRAME_DIR = "/home/bill/datasets/avalike_mbarv1/frames/"

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = ("/home/bill/datasets/avalike_mbarv1/frame_lists/")

# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = ("/home/bill/datasets/avalike_mbarv1/annotations/")

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]  # change to 2.1 if need
_C.AVA.VAL_GT_BOX_LISTS = ["ava_val_v2.2.csv"]  # change to 2.1 if need
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.AVA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.AVA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.AVA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of exclusion file for train
_C.AVA.TRAIN_EXCLUSION_FILE = "ava_train_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "cv2"

# Add custom config with default values.
custom_config.add_custom_config(_C)

def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
