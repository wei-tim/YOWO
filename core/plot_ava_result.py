import json
import os
import operator
import pdb
import matplotlib.pyplot as plt
import numpy as np

#import _init_paths
#from config.defaults import get_cfg
#from utils.ava_eval_helper import read_labelmap
#from datasets.ava_dataset import Ava


def main(json_file):
    with open('categories_count.json', 'r') as fb:
        categories_count = json.load(fb)

    with open(json_file, 'r') as fb:
        detection_result = json.load(fb)

    prefix = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/'
    categories = list(categories_count.keys())
    mAP_list = []
    for category in categories:
        mAP = detection_result[prefix + category]
        print(mAP)
        mAP_list.append(mAP)

    # width = np.diff(mAP_list).min()
    fig, ax = plt.subplots(figsize=(20, 8))
    x = list(range(len(categories)))
    ax.bar(x, mAP_list, align='center', width=0.8)
    for i, y in enumerate(mAP_list):
        ax.text(x[i] - 0.5, y + 0.01, '{:.2f}'.format(y), fontsize='x-small')
    ax.set(xticks=list(range(0, len(categories))), xticklabels=categories)
    plt.xticks(rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.34)
    #fig.autofmt_xdate()
    plt.savefig('ava_output_histogram.eps', format='eps')
    #plt.xticks(list(range(len(categories))), categories, rotation='vertical')
    #plt.show()
    


if __name__ == '__main__':
    main('latest_detection.json')
