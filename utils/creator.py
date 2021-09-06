import os
import argparse
import csv
import json
import pdb

from collections import defaultdict

class ActionLabels(object):
    """
    Mapping from action string labels to ids
    """

    ACTION_CLASSES_TO_IDS= {
        "Passing": 1,
        "Dribbling": 2,
        "Shooting": 3,
        "Ball possession": 4,
        "No action": 5,
        "Shoot in": 6,
        "Shoot out": 7,
        "No action ball": 8
    }

def read_csv(csv_file):
    '''
    Given an AVA-like CSV file for training or validation, extract info about video/frame paths/ids, append to list and return
    Args:
        csv_file: The CDV file to read from (i.e. train.csv or val.csv)
    '''
    video_frame_ids = []
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for i, row in enumerate(csv_reader):
            if i > 0:
                video_frame_ids.append([row[0], row[3].split('/')[1], row[2]])

    return video_frame_ids

def read_json(json_file):
    '''
    Given a Mindy Support anotation JSON file, create a dictionary having the bounding boxes infomation for each frame of each video
    Args:
        json_file: The JSON file to read from (i.e. 2021-04-27_10-01-15.json)
    '''
    with open(os.path.join(json_path, json_file), 'r') as file:
        annotation_data = json.load(file)

        boxes = defaultdict(list)
        for annotation in annotation_data['annotations']:
            img = annotation_data['images'][annotation['image_id']-1]['file_name']
            bbox = annotation['bbox']
            action_class = ActionLabels.ACTION_CLASSES_TO_IDS[annotation['attributes']['action_class']]
            
            boxes[img].append([bbox[0]/2592, bbox[1]/1944, bbox[2]/2592, bbox[3]/1944, action_class])
    
    return boxes

def write_frames(video_frame_ids, video_frame_boxes, csv_file):
    '''
    Given a list having the video/frame paths/ids (from read_csv) and a dictionary having the bounding boxes infomation for each frame of each video (from read_json),
    writes all the needed info as it should to csv_file 
    Args:
        video_frame_ids: list having the video/frame paths/ids (from read_csv)
        video_frame_boxes: dictionary having the bounding boxes infomation for each frame of each video (from read_json)
        csv_file: the file to write to (i.e. ava_train_v2.2, ava_val_v2.2)
    '''
    with open(os.path.join(output_path, csv_file), 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar="'")
        for video_frame in video_frame_ids:
            video = video_frame[0]
            frame = video_frame[1]
            id = video_frame[2]
            for bb in video_frame_boxes[video][0][frame]:
                writer.writerow([video, id, bb[0], bb[1], bb[2], bb[3], bb[4]])

    return None

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
    print('Found {} frames for training and {} frames for validation.'.format(len(train_video_frame_ids), len(val_video_frame_ids)))
    print('Total number of frames from CSVs: {}'.format(len(train_video_frame_ids) + len(val_video_frame_ids)))

    # Read JSONs
    jsons = [x for x in os.listdir(json_path) if x.endswith('.json')]
    print('Found {} JSON files in given directory.'.format(len(jsons)))
    
    video_frame_boxes = defaultdict(list)
    sum_boxes = 0
    for json_file in jsons:
        print('Reading {} file...'.format(json_file))
        per_json_boxes = read_json(json_file)
        print('Found {} frames in it.'.format(len(per_json_boxes)))
        video_frame_boxes[json_file.split('.')[0]].append(per_json_boxes)

        sum_boxes += len(per_json_boxes)
    
    print('Total number of frames from JSONs: {}'.format(sum_boxes))

    assert len(train_video_frame_ids) + len(val_video_frame_ids) == sum_boxes, "Oh no! Number of frames in CSVs and JSONs not consistent..."

    # Write to CSV files
    print('Writing to CSV files...')
    write_frames(train_video_frame_ids, video_frame_boxes, 'ava_train_v2.2.csv')
    write_frames(val_video_frame_ids, video_frame_boxes, 'ava_val_v2.2.csv')