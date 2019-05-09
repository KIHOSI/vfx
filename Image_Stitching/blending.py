import cv2
import numpy as np

#read image
img1 = cv2.imread("parrington/prtn01.jpg")
img2 = cv2.imread("parrington/prtn00.jpg")

alpha = 0.5
beta = 1 - alpha

#create clip windows from translation matrixs
height,width,color = img1.shape
windows_height = height
windows_width = 100

#clip images
num = width - windows_width
windows1 = img1[:,num:]
windows2 = img2[:,:windows_width]
img1 = img1[:,:num]
img2 = img2[:,windows_width:]
cv2.imwrite("img2.jpg",img2)

#blending with two windows
window = cv2.addWeighted(windows1,alpha,windows2,beta,0.0)

#spice images
image = np.concatenate([img1,window,img2],axis=1)
cv2.imwrite('image.jpg',image)
