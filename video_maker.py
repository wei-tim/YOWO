import cv2
import numpy as np

img=[]
for i in range(0,20):
    img.append(cv2.imread('/home/bill/datasets/ucf24/rgb-images/Basketball/v_Basketball_g01_c01/' + '0000' + str(i+1) + '.jpg'))

height,width,layers=img[0].shape

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter('video.avi', fourcc, 1,(width,height))

for im in img:
    video.write(im)

cv2.destroyAllWindows()
video.release()