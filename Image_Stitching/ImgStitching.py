import numpy as np
import cv2

#warp to cylindrical coordinate
def WarpToCylindrical(img,f):
    np.shape(img)

#main
images = []
focal_length = [704.916,706.286,705.849,706.645,706.587,705.645,705.327,704.696,703.794,704.325,704.696,703.895,704.289,704.676,704.847,704.537,705.102,705.576]
#read img
for i in range(0,18):
    if(i < 10):
        img = cv2.imread('parrington/prtn0'+str(i)+'.jpg')
    else:
        img = cv2.imread('parrington/prtn'+str(i)+'.jpg')
    images.append(img)
#warp to cylindrical coordinate
for i in range(0,1):
    WarpToCylindrical(images[i],focal_length[i])