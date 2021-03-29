import os
import glob
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat

from cfg import parser
from core.model import YOWO
from core.utils import *
from core.eval_results import *



####### Load configuration arguments
# ---------------------------------------------------------------
args  = parser.parse_args()
cfg   = parser.load_config(args)

dataset = cfg.TRAIN.DATASET
assert dataset == 'ucf24' or dataset == 'jhmdb21', 'invalid dataset'

gt_file       = 'cfg/ucf24_finalAnnots.mat' # Necessary for ucf
base_path     = cfg.LISTDATA.BASE_PTH
testlist      = os.path.join(base_path, 'testlist_video.txt')

clip_duration = cfg.DATA.NUM_FRAMES
sampling_rate = cfg.DATA.SAMPLING_RATE
anchors       = [float(i) for i in cfg.SOLVER.ANCHORS]
num_anchors   = cfg.SOLVER.NUM_ANCHORS
num_classes   = cfg.MODEL.NUM_CLASSES

# Test parameters
conf_thresh   = 0.005
nms_thresh    = 0.4
eps           = 1e-5



####### Create model
# ---------------------------------------------------------------
model = YOWO(cfg)
model = model.cuda()
model = nn.DataParallel(model, device_ids=None) # in multi-gpu case
# print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging('Total number of trainable parameters: {}'.format(pytorch_total_params))


# Load resume path 

if cfg.TRAIN.RESUME_PATH:
    print("===================================================================")
    print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded!")
    print("===================================================================")
    del checkpoint


def get_clip(root, imgpath, train_dur, sampling_rate, dataset):
    im_split = imgpath.split('/')
    num_parts = len(im_split)
    class_name = im_split[-3]
    file_name = im_split[-2]
    im_ind = int(im_split[num_parts - 1][0:5])
    if dataset == 'ucf24':
        img_name = os.path.join(class_name, file_name, '{:05d}.jpg'.format(im_ind))
    elif dataset == 'jhmdb21':
        img_name = os.path.join(class_name, file_name, '{:05d}.png'.format(im_ind))
    labpath = os.path.join(base_path, 'labels', class_name, file_name, '{:05d}.txt'.format(im_ind))
    img_folder = os.path.join(base_path, 'rgb-images', class_name, file_name)
    max_num = len(os.listdir(img_folder))
    clip = [] 

    d = sampling_rate
    for i in reversed(range(train_dur)):
        i_img = im_ind - i * d
        if i_img < 1:
            i_img = 1
        elif i_img > max_num:
            i_img = max_num

        if dataset == 'ucf24':
            path_tmp = os.path.join(base_path, 'rgb-images', class_name, file_name, '{:05d}.jpg'.format(i_img))
        elif dataset == 'jhmdb21':
            path_tmp = os.path.join(base_path, 'rgb-images', class_name, file_name, '{:05d}.png'.format(i_img))      
        clip.append(Image.open(path_tmp).convert('RGB'))

    label = torch.zeros(50 * 5)
    try:
        tmp = torch.from_numpy(read_truths_args(labpath, 8.0 / clip[0].width).astype('float32'))
    except Exception:
        tmp = torch.zeros(1, 5)

    tmp = tmp.view(-1)
    tsz = tmp.numel()

    if tsz > 50 * 5:
        label = tmp[0:50 * 5]
    elif tsz > 0:
        label[0:tsz] = tmp

    return clip, label, img_name

class testData(Dataset):
    def __init__(self, root, shape=None, transform=None, clip_duration=16, sampling_rate=1):

        self.root = root
        if dataset == 'ucf24':
            self.label_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
        elif dataset == 'jhmdb21':
            self.label_paths = sorted(glob.glob(os.path.join(root, '*.png')))

        self.shape = shape
        self.transform = transform
        self.clip_duration = clip_duration
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label_path = self.label_paths[index]

        clip, label, img_name = get_clip(self.root, label_path, self.clip_duration, self.sampling_rate, dataset)
        clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        return clip, label, img_name

def video_mAP_ucf():
    """
    Calculate video_mAP over the test dataset
    """
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    CLASSES = ('Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 
               'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
               'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
               'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
               'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')
    
    video_testlist = []
    with open(testlist, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            video_testlist.append(line)

    detected_boxes = {}
    gt_videos = {}

    gt_data = loadmat(gt_file)['annot']
    n_videos = gt_data.shape[1]
    for i in range(n_videos):
        video_name = gt_data[0][i][1][0]
        if video_name in video_testlist:
            n_tubes = len(gt_data[0][i][2][0])
            v_annotation = {}
            all_gt_boxes = []
            for j in range(n_tubes):  
                gt_one_tube = [] 
                tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                tube_class = gt_data[0][i][2][0][j][2][0][0]
                tube_data = gt_data[0][i][2][0][j][3]
                tube_length = tube_end_frame - tube_start_frame + 1
            
                for k in range(tube_length):
                    gt_boxes = []
                    gt_boxes.append(int(tube_start_frame+k))
                    gt_boxes.append(float(tube_data[k][0]))
                    gt_boxes.append(float(tube_data[k][1]))
                    gt_boxes.append(float(tube_data[k][0]) + float(tube_data[k][2]))
                    gt_boxes.append(float(tube_data[k][1]) + float(tube_data[k][3]))
                    gt_one_tube.append(gt_boxes)
                all_gt_boxes.append(gt_one_tube)

            v_annotation['gt_classes'] = tube_class
            v_annotation['tubes'] = np.array(all_gt_boxes)
            gt_videos[video_name] = v_annotation

    for line in lines:
        print(line)
        line = line.rstrip()
        test_loader = torch.utils.data.DataLoader(
                          testData(os.path.join(base_path, 'rgb-images', line),
                          shape=(224, 224), transform=transforms.Compose([
                          transforms.ToTensor()]), clip_duration=clip_duration, sampling_rate=sampling_rate),
                          batch_size=64, shuffle=False, num_workers= 8, pin_memory= True)

        for batch_idx, (data, target, img_name) in enumerate(test_loader):
            data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = model(data).data

                all_boxes = get_region_boxes_video(output, conf_thresh, num_classes, anchors, num_anchors, 0, 1)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    n_boxes = len(boxes)

                    # generate detected tubes for all classes
                    # save format: {img_name: {cls_ind: array[[x1,y1,x2,y2, cls_score], [], ...]}}
                    img_annotation = {}
                    for cls_idx in range(num_classes):
                        cls_idx += 1    # index begins from 1
                        cls_boxes = np.zeros([n_boxes, 5], dtype=np.float32)
                        for b in range(n_boxes):
                            cls_boxes[b][0] = max(float(boxes[b][0]-boxes[b][2]/2.0) * 320.0, 0.0)
                            cls_boxes[b][1] = max(float(boxes[b][1]-boxes[b][3]/2.0) * 240.0, 0.0)
                            cls_boxes[b][2] = min(float(boxes[b][0]+boxes[b][2]/2.0) * 320.0, 320.0)
                            cls_boxes[b][3] = min(float(boxes[b][1]+boxes[b][3]/2.0) * 240.0, 240.0)
                            cls_boxes[b][4] = float(boxes[b][5+(cls_idx-1)*2])
                        img_annotation[cls_idx] = cls_boxes
                    detected_boxes[img_name[i]] = img_annotation


    iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    for iou_th in iou_list:
        print('iou is: ', iou_th)
        print(evaluate_videoAP(gt_videos, detected_boxes, CLASSES, iou_th, True))



def video_mAP_jhmdb():
    """
    Calculate video_mAP over the test set
    """
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    CLASSES = ('brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 
                    'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                    'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                    'stand', 'swing_baseball', 'throw', 'walk', 'wave')

    with open(testlist, 'r') as file:
        lines = file.readlines()

    detected_boxes = {}
    gt_videos = {}
    for line in lines:
        print(line)

        line = line.rstrip()

        test_loader = torch.utils.data.DataLoader(
                          testData(os.path.join(base_path, 'rgb-images', line),
                          shape=(224, 224), transform=transforms.Compose([
                          transforms.ToTensor()]), clip_duration=clip_duration, sampling_rate=sampling_rate),
                          batch_size=1, shuffle=False, num_workers= 8, pin_memory= True)

        video_name = ''
        v_annotation = {}
        all_gt_boxes = []
        t_label = -1

        for batch_idx, (data, target, img_name) in enumerate(test_loader):
            path_split = img_name[0].split('/')
            if video_name == '':
                video_name = os.path.join(path_split[0], path_split[1])

            data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = model(data).data
                all_boxes = get_region_boxes_video(output, conf_thresh, num_classes, anchors, num_anchors, 0, 1)

                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    n_boxes = len(boxes)
                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)

                    if t_label == -1:
                        t_label = int(truths[0][0]) + 1

                    # generate detected tubes for all classes
                    # save format: {img_name: {cls_ind: array[[x1,y1,x2,y2, cls_score], [], ...]}}
                    img_annotation = {}
                    for cls_idx in range(num_classes):
                        cls_idx += 1    # index begins from 1
                        cls_boxes = np.zeros([n_boxes, 5], dtype=np.float32)
                        for b in range(n_boxes):
                            cls_boxes[b][0] = max(float(boxes[b][0]-boxes[b][2]/2.0) * 320.0, 0.0)
                            cls_boxes[b][1] = max(float(boxes[b][1]-boxes[b][3]/2.0) * 240.0, 0.0)
                            cls_boxes[b][2] = min(float(boxes[b][0]+boxes[b][2]/2.0) * 320.0, 320.0)
                            cls_boxes[b][3] = min(float(boxes[b][1]+boxes[b][3]/2.0) * 240.0, 240.0)
                            cls_boxes[b][4] = float(boxes[b][5+(cls_idx-1)*2])
                        img_annotation[cls_idx] = cls_boxes
                    detected_boxes[img_name[0]] = img_annotation

                    # generate corresponding gts
                    # save format: {v_name: {tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}} 
                    gt_boxes = []
                    for g in range(num_gts):
                        gt_boxes.append(int(path_split[2][:5]))
                        gt_boxes.append(float(truths[g][1]-truths[g][3]/2.0) * 320.0)
                        gt_boxes.append(float(truths[g][2]-truths[g][4]/2.0) * 240.0)
                        gt_boxes.append(float(truths[g][1]+truths[g][3]/2.0) * 320.0)
                        gt_boxes.append(float(truths[g][2]+truths[g][4]/2.0) * 240.0)
                        all_gt_boxes.append(gt_boxes)
                    
        v_annotation['gt_classes'] = t_label
        v_annotation['tubes'] = np.expand_dims(np.array(all_gt_boxes), axis=0)
        gt_videos[video_name] = v_annotation

    iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    for iou_th in iou_list:
        print('iou is: ', iou_th)
        print(evaluate_videoAP(gt_videos, detected_boxes, CLASSES, iou_th, True))


if __name__ == '__main__':
    if cfg.TRAIN.DATASET == 'ucf24':
        video_mAP_ucf()
    elif cfg.TRAIN.DATASET == 'jhmdb21':
        video_mAP_jhmdb()
    
