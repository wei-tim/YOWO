import glob
import os
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import random
import pdb

def write_frames(frames, frame_ids, video_ids, csv_file):
    '''
    Given a txt file and a list of frame paths, writes the paths on the text
    Args:
        frames (list): List of frame paths
        txt_file (file object): a text file object where the image paths will be saved
    '''
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, delimiter=' ', quotechar="'")
        writer.writerow(['original_video_id', 'video_id', 'frame_id', 'path', 'labels'])
        for id, frame in enumerate(frames):
            video = frame.split('/')[0]
            writer.writerow([video, video_ids[video], frame_ids[id], frame, '""'])

    return None

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Custom splitter for data')
    parser.add_argument('--input', type = str, help='Path to frames. i.e. /home/bill/datasets/mbarv1/frames/')
    parser.add_argument('--split', type = int, default = 0.1, help='Split percentage between train and validation set')
    args = parser.parse_args()

    current_dir = Path(args.input)
    csvs_dir = Path(os.path.dirname(args.input)) / "frame_lists"
    
    file_train = os.path.join(csvs_dir,'train.csv')
    file_val = os.path.join(csvs_dir,'val.csv')
    #file_test = os.path.join(str(Path(args.input)),'test.csv')

    videos = []
    frames = []
    lengths = []
    for folder in os.listdir(os.path.abspath(str(current_dir))):
        videos.append(folder)
        for id, frame in enumerate(os.listdir(os.path.join(os.path.abspath(str(current_dir)), folder))):
            frames.append(os.path.join(folder, frame))
        lengths.append(id+1)

    video_ids = {v:i for i,v in enumerate(videos)}
    num_frames = {k:v for k,v in zip(video_ids, lengths)}
    
    frames.sort()
    videos.sort()

    frame_ids = []
    for video in videos:
        frame_ids.extend(range(num_frames[video]))
    
    '''
    A DIFFERENT APPROACH - WE'LL VISIT THIS AGAIN
    min_value_lengths = min(lengths)
    min_index_lengths = lengths.index(min_value_lengths)

    train_frames = frames[sum(lengths[:min_index_lengths])+min_value_lengths:]
    train_frame_ids = frame_ids[sum(lengths[:min_index_lengths])+min_value_lengths:]
    
    write_frames(train_frames, train_frame_ids, video_ids, file_train)

    val_frames = frames[:sum(lengths[:min_index_lengths])+min_value_lengths]
    val_frame_ids = frame_ids[:sum(lengths[:min_index_lengths])+min_value_lengths]
    
    write_frames(val_frames, val_frame_ids, video_ids, file_val)
    '''

    train_frames = frames[:round((1-args.split)*len(frames))]
    train_frame_ids = frame_ids[:round((1-args.split)*len(frames))]
    write_frames(train_frames, train_frame_ids, video_ids, file_train)

    val_frames = frames[round((1-args.split)*len(frames)):]
    val_frame_ids = frame_ids[round((1-args.split)*len(frames)):]
    write_frames(val_frames, val_frame_ids, video_ids, file_val)