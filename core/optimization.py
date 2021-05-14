import os
import torch
import time
from core.utils import *
from datasets.meters import AVAMeter



def train_ava(cfg, epoch, model, train_loader, loss_module, optimizer):
    t0 = time.time()
    loss_module.reset_meters()
    l_loader = len(train_loader)

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data = batch['clip'].cuda()
        target = {'cls': batch['cls'], 'boxes': batch['boxes']}
        output = model(data)
        loss = loss_module(output, target, epoch, batch_idx, l_loader)

        loss.backward()
        steps = cfg.TRAIN.TOTAL_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
        if batch_idx % steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # save result every 1000 batches
        if batch_idx % 2000 == 0: # From time to time, reset averagemeters to see improvements
            loss_module.reset_meters()

    t1 = time.time()
    logging('trained with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    print('')



def train_ucf24_jhmdb21(cfg, epoch, model, train_loader, loss_module, optimizer):
    t0 = time.time()
    loss_module.reset_meters()
    l_loader = len(train_loader)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        output = model(data)
        loss = loss_module(output, target, epoch, batch_idx, l_loader)

        loss.backward()
        steps = cfg.TRAIN.TOTAL_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
        if batch_idx % steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # save result every 1000 batches
        if batch_idx % 2000 == 0: # From time to time, reset averagemeters to see improvements
            loss_module.reset_meters()

    t1 = time.time()
    logging('trained with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    print('')



@torch.no_grad()
def test_ava(cfg, epoch, model, test_loader):
     # Test parameters
    num_classes       = cfg.MODEL.NUM_CLASSES
    anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors       = cfg.SOLVER.NUM_ANCHORS
    nms_thresh        = 0.5
    conf_thresh_valid = 0.005

    nbatch = len(test_loader)
    meter = AVAMeter(cfg, cfg.TRAIN.MODE, 'latest_detection.json')

    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        data = batch['clip'].cuda()
        target = {'cls': batch['cls'], 'boxes': batch['boxes']}

        with torch.no_grad():
            output = model(data)
            metadata = batch['metadata'].cpu().numpy()

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
                    preds.append([[x1,y1,x2,y2], cls_out, metadata[i][:2].tolist()])

        meter.update_stats(preds)
        logging("[%d/%d]" % (batch_idx, nbatch))

    mAP = meter.evaluate_ava()
    logging("mode: {} -- mAP: {}".format(meter.mode, mAP))

    return mAP



@torch.no_grad()
def test_ucf24_jhmdb21(cfg, epoch, model, test_loader):

    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Test parameters
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    num_classes = cfg.MODEL.NUM_CLASSES
    anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors = cfg.SOLVER.NUM_ANCHORS
    conf_thresh_valid = 0.005
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0

    correct_classification = 0.0
    total_detected = 0.0

    nbatch = len(test_loader)

    model.eval()

    for batch_idx, (frame_idx, data, target) in enumerate(test_loader):
        data = data.cuda()
        with torch.no_grad():
            output = model(data).data
            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                if cfg.TRAIN.DATASET == 'ucf24':
                    detection_path = os.path.join('ucf_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('ucf_detections', 'detections_'+str(epoch))
                    if not os.path.exists('ucf_detections'):
                        os.mkdir('ucf_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)
                else:
                    detection_path = os.path.join('jhmdb_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('jhmdb_detections', 'detections_'+str(epoch))
                    if not os.path.exists('jhmdb_detections'):
                        os.mkdir('jhmdb_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = round(float(box[0]-box[2]/2.0) * 320.0)
                        y1 = round(float(box[1]-box[3]/2.0) * 240.0)
                        x2 = round(float(box[0]+box[2]/2.0) * 320.0)
                        y2 = round(float(box[1]+box[3]/2.0) * 240.0)

                        det_conf = float(box[4])
                        for j in range((len(box)-5)//2):
                            cls_conf = float(box[5+2*j].item())
                            prob = det_conf * cls_conf

                            f_detect.write(str(int(box[6])+1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)
        
                total = total + num_gts
                pred_list = [] # LIST OF CONFIDENT BOX INDICES
                for i in range(len(boxes)):
                    if boxes[i][4] > 0.25:
                        proposals = proposals+1
                        pred_list.append(i)

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = -1
                    for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > iou_thresh:
                        total_detected += 1
                        if int(boxes[best_j][6]) == box_gt[6]:
                            correct_classification += 1

                    if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                        correct = correct+1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            logging("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))

    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    locolization_recall = 1.0 * total_detected / (total + eps)

    print("Classification accuracy: %.3f" % classification_accuracy)
    print("Locolization recall: %.3f" % locolization_recall)

    return fscore
