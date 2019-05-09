import numpy as np
import cv2
import math
import random

#warp to cylindrical coordinate
def WarpToCylindrical(img,f):
    print(np.shape(img)) #(512,384,3)
    (height,width,color) = np.shape(img)
    img_transformed = np.zeros((height,width,color)) #x,y,f,rgb
    for x in range(width): #y
        for y in range(height): #x
            angle = math.atan((x-192)/f)
            h = y/math.sqrt(math.pow((x-192),2)+math.pow(f,2))
            x_new = int(f*angle)
            y_new = int(f*h) 
            img_transformed[y_new,x_new+192,:] = img[y,x,:]
    return img_transformed        

def WANSAC(array): #image stitching : WANSAC
    n = 6
    P = 0.99
    k = 293
    e = 9 # canculate inliear

    for i in range(k):
        sample = np.zeros((n))
        random.shuffle(array)
        sample = array[:n]
        other_points = array[n:]
        sample_model = model.fit(sample)
        err_threshold = model.get_error(other_points,sample_model)
        

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
img_cylindrical = np.zeros((17,512,384,3)) #store transformed images (project to cylindrical pos)
for i in range(0,1):
    img_cylindrical[i,:,:,:] = WarpToCylindrical(images[i],int(focal_length[i]))
    cv2.imwrite("test.jpg",img_cylindrical[i])

#image stitching : WANSAC
