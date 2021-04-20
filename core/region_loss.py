import time
import json
import torch
import torch.nn.functional as F

import math
import torch.nn as nn
from torch.autograd import Variable
from core.utils import *
from builtins import range as xrange
import numpy as np
from core.FocalLoss import *

# this function works for building the groud truth 
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh):
    # nH, nW here are number of grids in y and x directions (7, 7 here)
    nB = target.size(0) # batch size
    nA = num_anchors    # 5 for our case
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    # for each grid there are nA anchors
    # nAnchors is the number of anchor for one image
    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    # for each image
    for b in xrange(nB):
        # get all anchor boxes in one image
        # (4 * nAnchors)
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        # initialize iou score for each anchor
        cur_ious = torch.zeros(nAnchors)
        for t in xrange(50):
            # for each anchor 4 coordinate parameters, already in the coordinate system for the whole image
            # this loop is for anchors in each image
            # for each anchor 5 parameters are available (class, x, y, w, h)
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            # groud truth boxes
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            # bbox_ious is the iou value between orediction and groud truth
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        # if iou > a given threshold, it is seen as it includes an object
        # conf_mask[b][cur_ious>sil_thresh] = 0
        conf_mask_t = conf_mask.view(nB, -1)
        conf_mask_t[b][cur_ious>sil_thresh] = 0
        conf_mask_tt = conf_mask_t[b].view(nA, nH, nW)
        conf_mask[b] = conf_mask_tt

    # number of ground truth
    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        # anchors for one batch (at least batch size, and for some specific classes, there might exist more than one anchor)
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            # the values saved in target is ratios
            # times by the width and height of the output feature maps nW and nH
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3] * nW
            gh = target[b][t*5+4] * nH
            gt_box = [0, 0, gw, gh]
            for n in xrange(nA):
                # get anchor parameters (2 values)
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                # only consider the size (width and height) of the anchor box
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                # get the best anchor form with the highest iou
                if iou > best_iou:
                    best_iou = iou
                    best_n = n


            # then we determine the parameters for an anchor (4 values together)
            gt_box = [gx, gy, gw, gh]
            # find corresponding prediction box
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            # only consider the best anchor box, for each image
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            # in this cell of the output feature map, there exists an object
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            # confidence equals to iou of the corresponding anchor
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            # if ious larger than 0.5, we justify it as a correct prediction
            if iou > 0.5:
                nCorrect = nCorrect + 1

    # true values are returned
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    # for our model anchors has 10 values and number of anchors is 5
    # parameters: 24, 10 float values, 24, 5
    def __init__(self, cfg):
        super(RegionLoss, self).__init__()
        self.num_classes    = cfg.MODEL.NUM_CLASSES
        self.batch          = cfg.TRAIN.BATCH_SIZE
        self.anchors        = [float(i) for i in cfg.SOLVER.ANCHORS]
        self.num_anchors    = cfg.SOLVER.NUM_ANCHORS
        self.anchor_step    = len(self.anchors)//self.num_anchors    # each anchor has 2 parameters
        self.object_scale   = cfg.SOLVER.OBJECT_SCALE
        self.noobject_scale = cfg.SOLVER.NOOBJECT_SCALE
        self.class_scale    = cfg.SOLVER.CLASS_SCALE
        self.coord_scale    = cfg.SOLVER.COORD_SCALE
        self.focalloss      = FocalLoss(class_num=self.num_classes, gamma=2, size_average=False)
        self.thresh = 0.6
        self.l_x = AverageMeter()
        self.l_y = AverageMeter()
        self.l_w = AverageMeter()
        self.l_h = AverageMeter()
        self.l_conf = AverageMeter()
        self.l_cls = AverageMeter()
        self.l_total = AverageMeter()

    def reset_meters(self):
        self.l_x.reset()
        self.l_y.reset()
        self.l_w.reset()
        self.l_h.reset()
        self.l_conf.reset()
        self.l_cls.reset()
        self.l_total.reset()


    def forward(self, output, target, epoch, batch_idx, l_loader):
        # output : B*A*(4+1+num_classes)*H*W
        # B: number of batches
        # A: number of anchors
        # 4: 4 parameters for each bounding box
        # 1: confidence score
        # num_classes
        # H: height of the image (in grids)
        # W: width of the image (in grids)
        # for each grid cell, there are A*(4+1+num_classes) parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # resize the output (all parameters for each anchor can be reached)
        output   = output.view(nB, nA, (5+nC), nH, nW)
        # anchor's parameter tx
        x    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        # anchor's parameter ty
        y    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        # anchor's parameter tw
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        # anchor's parameter th
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        # confidence score for each anchor
        conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        # anchor's parameter class label
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        # resize the data structure so that for every anchor there is a class label in the last dimension
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        # for the prediction of localization of each bounding box, there exist 4 parameters (tx, ty, tw, th)
        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        # tx and ty
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        # for each anchor there are anchor_step variables (with the structure num_anchor*anchor_step)
        # for each row(anchor), the first variable is anchor's width, second is anchor's height
        # pw and ph
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        # for each pixel (grid) repeat the above process (obtain width and height of each grid)
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        # prediction of bounding box localization
        # x.data and y.data: top left corner of the anchor
        # grid_x, grid_y: tx and ty predictions made by yowo

        x_data = x.data.view(-1)
        y_data = y.data.view(-1)
        w_data = w.data.view(-1)
        h_data = h.data.view(-1)

        pred_boxes[0] = x_data + grid_x    # bx
        pred_boxes[1] = y_data + grid_y    # by
        pred_boxes[2] = torch.exp(w_data) * anchor_w    # bw
        pred_boxes[3] = torch.exp(h_data) * anchor_h    # bh
        # the size -1 is inferred from other dimensions
        # pred_boxes (nB*nA*nH*nW, 4)
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh)
        cls_mask = (cls_mask == 1)
        #  keep those with high box confidence scores (greater than 0.25) as our final predictions
        nProposals = int((conf > 0.25).sum().data.item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        # losses between predictions and targets (ground truth)
        # In total 6 aspects are considered as losses: 
        # 4 for bounding box location, 2 for prediction confidence and classification seperately
        loss_x = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/2.0

        # try focal loss with gamma = 2
        loss_cls = self.class_scale * self.focalloss(cls, tcls)

        # sum of loss
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        #print(loss)
        t4 = time.time()

        self.l_x.update(loss_x.data.item(), self.batch)
        self.l_y.update(loss_y.data.item(), self.batch)
        self.l_w.update(loss_w.data.item(), self.batch)
        self.l_h.update(loss_h.data.item(), self.batch)
        self.l_conf.update(loss_conf.data.item(), self.batch)
        self.l_cls.update(loss_cls.data.item(), self.batch)
        self.l_total.update(loss.data.item(), self.batch)


        if batch_idx % 20 == 0: 
            print('Epoch: [%d][%d/%d]:\t nGT %d, recall %d, proposals %d, loss: x %.2f(%.2f), '
                  'y %.2f(%.2f), w %.2f(%.2f), h %.2f(%.2f), conf %.2f(%.2f), '
                  'cls %.2f(%.2f), total %.2f(%.2f)'
                   % (epoch, batch_idx, l_loader, nGT, nCorrect, nProposals, self.l_x.val, self.l_x.avg,
                    self.l_y.val, self.l_y.avg, self.l_w.val, self.l_w.avg,
                    self.l_h.val, self.l_h.avg, self.l_conf.val, self.l_conf.avg,
                    self.l_cls.val, self.l_cls.avg, self.l_total.val, self.l_total.avg))
        return loss





########################### AVA ###############################
###############################################################

class binary_FocalLoss(nn.Module):
    def __init__(self, gamma, class_num, class_count_json, size_average=True):
        super(binary_FocalLoss, self).__init__()
        with open(class_count_json, 'r') as fb:
            self.class_ratio = json.load(fb)
        self.gamma = gamma
        self.class_num = class_num
        # self.beta = 0.999
        self.size_average = size_average
        self._init_class_weight()

    def _init_class_weight(self):
        self.register_buffer('class_weight', torch.zeros(80))
        for i in range(1, 81):
            self.class_weight[i - 1] = 1 - self.class_ratio[str(i)]
            # n = self.class_ratio[str(i)]
            # self.class_weight[i - 1] = (1 - self.beta) / (1 - self.beta ** n)

    def forward(self, inputs, targets):
        '''
        inputs: (N, C) -- result of sigmoid
        targets: (N, C) -- one-hot variable
        '''
        assert self.class_num == targets.size(1)
        assert self.class_num == inputs.size(1)
        assert inputs.size(0) == targets.size(0)

        weight_matrix = self.class_weight.expand(inputs.size(0), self.class_num)
        weight_p1 = torch.exp(weight_matrix[targets == 1])
        weight_p0 = torch.exp(1 - weight_matrix[targets == 0])
        # weight_p1 = weight_matrix[targets == 1]
        # weight_p0 = 1 - weight_matrix[targets == 0]
        p_1 = inputs[targets == 1]
        p_0 = inputs[targets == 0]

        # loss = torch.sum(torch.log(p_1)) + torch.sum(torch.log(1 - p_0))  # origin bce loss
        loss1 = torch.pow(1 - p_1, self.gamma) * torch.log(p_1) * weight_p1
        loss2 = torch.pow(p_0, self.gamma) * torch.log(1 - p_0) * weight_p0
        loss = -torch.sum(loss1) - torch.sum(loss2)
        if self.size_average:
            loss /= inputs.size(0)

        return loss


def _sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)

def _softmax(x):
    return torch.clamp(F.softmax(x, dim=-1), min=1e-4, max=1 - 1e-4)


# this function works for building the groud truth 
def build_targets_Ava(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh):
    # nH, nW here are number of grids in y and x directions (7, 7 here)
    target_cls = target['cls']
    target_boxes = target['boxes']
    nB = target_cls.size(0) # batch size
    nA = num_anchors    # 5 for our case
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW, nC) 

    # for each grid there are nA anchors
    # nAnchors is the number of anchor for one image
    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    # for each image
    for b in xrange(nB):
        # get all anchor boxes in one image
        # (4 * nAnchors)
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        # initialize iou score for each anchor
        cur_ious = torch.zeros(nAnchors)
        for t in xrange(50):
            # for each anchor 4 coordinate parameters, already in the coordinate system for the whole image
            # this loop is for anchors in each image
            # for each anchor 5 parameters are available (class, x, y, w, h)
            if target_boxes[b,t,2] == 0:
                break
            gx = target_boxes[b,t,0]*nW
            gy = target_boxes[b,t,1]*nH
            gw = target_boxes[b,t,2]*nW
            gh = target_boxes[b,t,3]*nH
            # groud truth boxes
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            # bbox_ious is the iou value between orediction and groud truth
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        # if iou > a given threshold, it is seen as it includes an object
        # conf_mask[b][cur_ious>sil_thresh] = 0
        conf_mask_t = conf_mask.view(nB, -1)
        conf_mask_t[b][cur_ious>sil_thresh] = 0
        conf_mask_tt = conf_mask_t[b].view(nA, nH, nW)
        conf_mask[b] = conf_mask_tt

    # number of ground truth
    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        # anchors for one batch (at least batch size, and for some specific classes, there might exist more than one anchor)
        for t in xrange(50):
            if target_boxes[b,t,2] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            # the values saved in target is ratios
            # times by the width and height of the output feature maps nW and nH
            gx = target_boxes[b,t,0]*nW
            gy = target_boxes[b,t,1]*nH
            gi = int(gx)
            gj = int(gy)
            gw = target_boxes[b,t,2]*nW
            gh = target_boxes[b,t,3]*nH
            gt_box = [0, 0, gw, gh]
            for n in xrange(nA):
                # get anchor parameters (2 values)
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                # only consider the size (width and height) of the anchor box
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                # get the best anchor form with the highest iou
                if iou > best_iou:
                    best_iou = iou
                    best_n = n


            # then we determine the parameters for an anchor (4 values together)
            gt_box = [gx, gy, gw, gh]
            # find corresponding prediction box
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            # only consider the best anchor box, for each image
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            # in this cell of the output feature map, there exists an object
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target_boxes[b,t,0]*nW - gi
            ty[b][best_n][gj][gi] = target_boxes[b,t,1]*nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            # confidence equals to iou of the corresponding anchor
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi][:] = target_cls[b,t,:]
            # if ious larger than 0.5, we justify it as a correct prediction
            if iou > 0.5:
                nCorrect = nCorrect + 1

    # true values are returned
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls



class RegionLoss_Ava(nn.Module):
    # for our model anchors has 10 values and number of anchors is 5
    # parameters: 24, 10 float values, 24, 5
    def __init__(self, cfg):
        super(RegionLoss_Ava, self).__init__()
        self.num_classes    = cfg.MODEL.NUM_CLASSES
        self.batch          = cfg.TRAIN.BATCH_SIZE
        self.anchors        = [float(i) for i in cfg.SOLVER.ANCHORS]
        self.num_anchors    = cfg.SOLVER.NUM_ANCHORS
        self.anchor_step    = len(self.anchors)//self.num_anchors    # each anchor has 2 parameters
        self.object_scale   = cfg.SOLVER.OBJECT_SCALE
        self.noobject_scale = cfg.SOLVER.NOOBJECT_SCALE
        self.class_scale    = cfg.SOLVER.CLASS_SCALE
        self.coord_scale    = cfg.SOLVER.COORD_SCALE
        self.loss_func      = binary_FocalLoss(0.5, self.num_classes, cfg.TRAIN.CLASS_RATIO_FILE)
        self.thresh = 0.6
        self.l_x = AverageMeter()
        self.l_y = AverageMeter()
        self.l_w = AverageMeter()
        self.l_h = AverageMeter()
        self.l_conf = AverageMeter()
        self.l_cls = AverageMeter()
        self.l_total = AverageMeter()

    def reset_meters(self):
        self.l_x.reset()
        self.l_y.reset()
        self.l_w.reset()
        self.l_h.reset()
        self.l_conf.reset()
        self.l_cls.reset()
        self.l_total.reset()


    def forward(self, output, target, epoch, batch_idx, l_loader):
        # output : B*A*(4+1+num_classes)*H*W
        # B: number of batches
        # A: number of anchors
        # 4: 4 parameters for each bounding box
        # 1: confidence score
        # num_classes
        # H: height of the image (in grids)
        # W: width of the image (in grids)
        # for each grid cell, there are A*(4+1+num_classes) parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # resize the output (all parameters for each anchor can be reached)
        output   = output.view(nB, nA, (5+nC), nH, nW)
        # anchor's parameter tx
        x    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        # anchor's parameter ty
        y    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        # anchor's parameter tw
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        # anchor's parameter th
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        # confidence score for each anchor
        conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        # anchor's parameter class label
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        # resize the data structure so that for every anchor there is a class label in the last dimension
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        # for the prediction of localization of each bounding box, there exist 4 parameters (tx, ty, tw, th)
        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        # tx and ty
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        # for each anchor there are anchor_step variables (with the structure num_anchor*anchor_step)
        # for each row(anchor), the first variable is anchor's width, second is anchor's height
        # pw and ph
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        # for each pixel (grid) repeat the above process (obtain width and height of each grid)
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        # prediction of bounding box localization
        # x.data and y.data: top left corner of the anchor
        # grid_x, grid_y: tx and ty predictions made by yowo

        x_data = x.data.view(-1)
        y_data = y.data.view(-1)
        w_data = w.data.view(-1)
        h_data = h.data.view(-1)

        pred_boxes[0] = x_data + grid_x    # bx
        pred_boxes[1] = y_data + grid_y    # by
        pred_boxes[2] = torch.exp(w_data) * anchor_w    # bw
        pred_boxes[3] = torch.exp(h_data) * anchor_h    # bh
        # the size -1 is inferred from other dimensions
        # pred_boxes (nB*nA*nH*nW, 4)
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets_Ava(pred_boxes, target, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh)
        cls_mask = (cls_mask == 1)
        #  keep those with high box confidence scores (greater than 0.25) as our final predictions
        nProposals = int((conf > 0.25).sum().data.item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls = Variable(tcls.view(-1, nC)[cls_mask.view(-1), :].long().cuda()) # TODO: CHECKOUT THIS ONE

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        # losses between predictions and targets (ground truth)
        # In total 6 aspects are considered as losses: 
        # 4 for bounding box location, 2 for prediction confidence and classification seperately
        loss_x = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/2.0

        # try binary_FocalLoss
        pose_output = _softmax(cls[:, :14])
        inter_output = _sigmoid(cls[:, 14:])
        total_output = torch.cat([pose_output, inter_output], dim=1)
        loss_cls = self.class_scale * self.loss_func(total_output, tcls)  # cls_loss

        # sum of loss
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        #print(loss)
        t4 = time.time()

        self.l_x.update(loss_x.data.item(), self.batch)
        self.l_y.update(loss_y.data.item(), self.batch)
        self.l_w.update(loss_w.data.item(), self.batch)
        self.l_h.update(loss_h.data.item(), self.batch)
        self.l_conf.update(loss_conf.data.item(), self.batch)
        self.l_cls.update(loss_cls.data.item(), self.batch)
        self.l_total.update(loss.data.item(), self.batch)


        if batch_idx % 20 == 0: 
            print('Epoch: [%d][%d/%d]:\t nGT %d, recall %d, proposals %d, loss: x %.2f(%.2f), '
                  'y %.2f(%.2f), w %.2f(%.2f), h %.2f(%.2f), conf %.2f(%.2f), '
                  'cls %.2f(%.2f), total %.2f(%.2f)'
                   % (epoch, batch_idx, l_loader, nGT, nCorrect, nProposals, self.l_x.val, self.l_x.avg,
                    self.l_y.val, self.l_y.avg, self.l_w.val, self.l_w.avg,
                    self.l_h.val, self.l_h.avg, self.l_conf.val, self.l_conf.avg,
                    self.l_cls.val, self.l_cls.avg, self.l_total.val, self.l_total.avg))

        return loss
