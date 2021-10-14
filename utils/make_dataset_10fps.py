import os
import pdb
from collections import defaultdict

mbarv1_10_fps_path = '/home/bill/datasets/mbarv1_10fps/'
videos = os.listdir(os.path.join(mbarv1_10_fps_path, 'frames'))

videos_frames_dict = defaultdict(list)
for video in videos:
    for image in os.listdir(os.path.join(mbarv1_10_fps_path, 'frames', video)):
        videos_frames_dict[video].append(image)

for video in videos_frames_dict.keys():
    videos_frames_dict[video] = sorted(videos_frames_dict[video])

for video in videos_frames_dict.keys():
    del videos_frames_dict[video][::2]

for video in videos:
    for image in videos_frames_dict[video]:
        os.remove(os.path.join(mbarv1_10_fps_path, 'frames', video, image))