import os
import torch
import time
import pdb
from core.utils import *
from datasets.meters import AVAMeter



def train_ava(cfg, epoch, model, train_loader, loss_module, optimizer):
    t0 = time.time()
    loss_module.reset_meters()
    l_loader = len(train_loader)

    model.train()

    for batch_idx, batch in enumerate(train_loader):
        #pdb.set_trace()

        data = batch['clip'].cuda()
        target = {'cls': batch['cls'], 'boxes': batch['boxes']}

        output = model(data)
        """
        for k in range(batch['clip'][0].shape[1]):
            # That's a debugging function that visualizes the first frame of a batch
            import cv2
            # Select the first frame, detach and *255
            example = batch['clip'][0][:,k,:,:].detach().numpy()*255
            # Move axis to be correctly oriented
            example = np.moveaxis(example, 0, -1)
            # Colorize correctly
            example = cv2.cvtColor(np.float32(example), cv2.COLOR_RGB2BGR)
        
            #cv2.imwrite('batch_debugging/batch_{}_example_{}.jpg'.format(batch_idx, k), example)

            crop_size = cfg.DATA.TEST_CROP_SIZE
        
            for i, dets in enumerate(batch['boxes'][0]):
                x = dets[0].item() * crop_size
                y = dets[1].item() * crop_size
                w = dets[2].item() * crop_size
                h = dets[3].item() * crop_size
                #print(x, y, w, h)
                
                x1 = int(x-w/2)
                y1 = int(y-w/1)
                x2= int(x+w/2)
                y2= int(y+w/2)
                #print(x1, x2, y1, y2)
                
                cv2.rectangle(example, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('batch_debugging/batch_{}_example_{}_with_labels.jpg'.format(batch_idx, k), example)
            
            
            '''
            from datasets.ava_eval_helper import read_labelmap 
            labelmap, _       = read_labelmap(os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.TEST_LABEL_MAP_FILE))

            cls_scores = np.array(batch['cls'][i])
            indices = np.where(cls_scores==1)
            scores = cls_scores[indices]
            indices = list(indices[0])
            scores = list(scores)
            
            if len(scores) > 0:
                blk   = np.zeros(example.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                for _, cls_ind in enumerate(indices):
                    pdb.set_trace()
                    text.append("[{:.2f}] ".format(scores[_]) + str(labelmap[cls_ind]['name']))
                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                    coord.append((x1+3, y1+7+10*_))
                    #cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                #malakia = cv2.addWeighted(malakia, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(example, text[t], coord[t], font, 0.25, (0, 0, 0), 1)
                '''
        """
        #pdb.set_trace()
        
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
    nms_thresh        = 0.2
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