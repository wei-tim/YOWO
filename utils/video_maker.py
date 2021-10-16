import cv2
import numpy as np
import glob
import pdb

filenames = []
img_array = []
for filename in glob.glob('/home/bill/code/yowo/inference/random_inference/*.jpg'):
    filenames.append(filename)

filenames.sort()

for filename in filenames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('/home/bill/code/yowo/inference/videos/random_inference.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()