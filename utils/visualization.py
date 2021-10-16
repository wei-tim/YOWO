import csv
from pathlib import Path
import argparse
import cv2 as cv
from ast import literal_eval
from collections import defaultdict
import json
import pdb
import os

class _Labels_(object):
    """
    Mapping from class labels to colors
    """

    ACTION_CLASSES_TO_CLRS= {
        "Passing": (255,0,0),
        "Dribbling": (0,255,0),
        "Shooting": (0,0,255),
        "Ball possession": (255,255,0),
        "No action": (0,255,255),
        "Shoot in": (255,255,255),
        "Shoot out": (0,0,0),
        "No action ball": (255,0,255)
    }


def draw_box(img, bbox, label, mode):
    if mode=='action_recognition1':
        bbox_dict = literal_eval(bbox)
        label = label.split('"')[-2]
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2)
        cv.rectangle(img,(bbox_dict['x'],bbox_dict['y']),(bbox_dict['x']+bbox_dict['width'],bbox_dict['y']+bbox_dict['height']),color=_Labels_.ACTION_CLASSES_TO_CLRS[label],thickness=3)
        #cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv.putText(img, label, (bbox_dict['x'],bbox_dict['y']),0,tl/3,color=_Labels_.ACTION_CLASSES_TO_CLRS[label],thickness=max(tl, 1), lineType=cv.LINE_AA)
    
    elif mode=='action_recognition2':
        bbox = [int(x) for x in bbox]
        label = label['action_class']
        print(label)
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2)
        cv.rectangle(img,(bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]),color=_Labels_.ACTION_CLASSES_TO_CLRS[label],thickness=3)
        cv.putText(img, label, (bbox[0], bbox[1]),0,tl/3,color=_Labels_.ACTION_CLASSES_TO_CLRS[label],thickness=max(tl, 1), lineType=cv.LINE_AA)
    
    else:
        print('Please specify a mode from: action_recognition1 | action_recognition2.')
    
    return img

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Visualize annotation results from Mindys Support')
    parser.add_argument('--input', type = str, help='path to annotation csv')
    parser.add_argument('--output', type = str, help='path to where the results will be saved')
    parser.add_argument('--mode', type = str, default='action_recognition2', help='convert mode, e.g. action_recognition1 or action_recognition2')
    args = parser.parse_args()

    data = defaultdict(list)
    annotation_file = Path(args.input)
    out_path = Path(args.output)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if args.mode=='action_recognition1':
        with open(annotation_file,'r') as csvfile:
            annotation_data = csv.reader(csvfile,delimiter=',')
            next(annotation_data,None)
            for img,_,_,_,_,bbox,label in annotation_data:
                data[img].append([bbox,label])
            img_names = list(data.keys())
            for counter in range(len(img_names)):
                if img_names[counter].endswith('.jpg'):
                    img_path = annotation_file.parent / img_names[counter]
                    img = cv.imread(str(img_path))
                    [draw_box(img,x[0],x[1],args.mode) for x in data[img_names[counter]]]
                    cv.imwrite(str(out_path / ('annotated_'+img_names[counter])),img)
    
    elif args.mode=='action_recognition2':
        with open(annotation_file, 'r') as file:
            annotation_data = json.load(file)
            for annotation in annotation_data['annotations']:
                img= annotation_data['images'][annotation['image_id']-1]['file_name']
                bbox = annotation['bbox']
                object_class = annotation['category_id']
                action_class = annotation['attributes']
                action_class['object_class'] = object_class
                
                #if object_class == 4:
                #    if action_class['action_class']!='Shoot in':
                #        action_class['action_class']='Shoot out' 
                
                label = action_class

                data[img].append([bbox, label])
                # annotation['image_id'], annotation['category_id'], annotation['bbox'], annotation['attributes']
                # annotation_data['images'][annotation['image_id']-1]
            img_names = list(data.keys())
            for counter in range(len(img_names)):
                if img_names[counter].endswith('.jpg'):
                    #img_path = annotation_file.parent / img_names[counter]
                    img_path = annotation_file.parent / 'frames' / 'far' / img_names[counter]
                    img = cv.imread(str(img_path))
                    [draw_box(img,x[0],x[1],args.mode) for x in data[img_names[counter]]]
                    cv.imwrite(str(out_path / ('annotated_'+img_names[counter])),img)
