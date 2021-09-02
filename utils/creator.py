import glob
import os
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import json
import random
import pdb

class ActionLabels(object):
    """
    Mapping from action string labels to ids
    """

    ACTION_CLASSES_TO_IDS= {
        "Passing": 0,
        "Dribbling": 1,
        "Shooting": 2,
        "Ball possession": 3,
        "No action": 4,
        "Shoot in": 5,
        "Shoot out": 6,
        "No action ball": 7
    }

def write_frames(video_frame_ids, video_frame_boxes, csv_file):
    with open(os.path.join(output_path, csv_file), 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar="'")
        for video_frame in video_frame_ids:
            video = video_frame[0]
            frame = video_frame[1]
            id = video_frame[2]
            for bb in video_frame_boxes[video][0][frame]:
                writer.writerow([video, id, bb[0], bb[1], bb[2], bb[3], bb[4]])

    return None

def read_csv(csv_file):
    '''
    Given an AVA-like CSV file for training or validation, extract info about video/frame paths/ids, append to list and return
    Args:
        csv_file: The file to read from (i.e. train.csv or val.csv)
    '''
    #original_video_ids = []
    #original_frame_ids = []
    #frame_ids = []
    video_frame_ids = []
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for i, row in enumerate(csv_reader):
            if i > 0:
                video_frame_ids.append([row[0], row[4].split('/')[1], row[2]])
                #original_video_ids.append(row[0])
                #original_frame_ids.append(row[4].split('/')[1])
                #frame_ids.append(row[2])

    return video_frame_ids

def read_json(json_file):
    with open(os.path.join(json_path, json_file), 'r') as file:
        annotation_data = json.load(file)

        boxes = defaultdict(list)
        for annotation in annotation_data['annotations']:
            img = annotation_data['images'][annotation['image_id']-1]['file_name']
            bbox = annotation['bbox']
            #object_class = annotation['category_id']
            action_class = ActionLabels.ACTION_CLASSES_TO_IDS[annotation['attributes']['action_class']]
            
            boxes[img].append([bbox[0], bbox[1], bbox[2], bbox[3], action_class])
            
            #boxes.append([json_file, img, bbox[0], bbox[1], bbox[2], bbox[3], action_class])
    
    return boxes


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Annotation files creator. Takes as input the JSON files of Mindy and the CSV files of AVA dataset. Outputs ava_train_v2.2 and ava_val_v2.2 CSV files')
    parser.add_argument('--input_json_path', type = str, default = '/home/bill/datasets/avalike_mbarv1/annotations/',  help='Path to JSON files. i.e. /home/bill/datasets/avalike_mbarv1/annotations/')
    parser.add_argument('--input_csv_path', type = str, default = '/home/bill/datasets/avalike_mbarv1/frame_lists/', help='Path to train and val CSV files of AVA dataset, i.e. /home/bill/datasets/avalike_mbarv1/frame_lists/')
    parser.add_argument('--output_path', type = str, default = '/home/bill/datasets/avalike_mbarv1/annotations', help='Path to train and val CSV files of AVA dataset, i.e. /home/bill/datasets/avalike_mbarv1/annotations')
    args = parser.parse_args()

    json_path = args.input_json_path
    csv_path = args.input_csv_path
    output_path = args.output_path

    # Read CSVs
    print('Reading training and validation CSVs')
    train_video_frame_ids = read_csv(os.path.join(csv_path, 'train.csv'))
    val_video_frame_ids = read_csv(os.path.join(csv_path, 'val.csv'))

    # Read JSONs
    jsons = [x for x in os.listdir(json_path) if x.endswith('.json')]
    print('Found {} JSON files in given directory.'.format(len(jsons)))
    
    video_frame_boxes = defaultdict(list)
    for json_file in jsons:
        print('Reading {} file...'.format(json_file))
        per_json_boxes = read_json(json_file)
        print('Found {} boxes in it.'.format(len(per_json_boxes)))
        video_frame_boxes[json_file.split('.')[0]].append(per_json_boxes)

    write_frames(train_video_frame_ids, video_frame_boxes, 'ava_train_v2.2.csv')
    write_frames(val_video_frame_ids, video_frame_boxes, 'ava_val_v2.2.csv')


