import cv2
import numpy as np
from scipy import ndimage, signal
import scipy
import math
import matplotlib.pyplot as plt
from skimage import draw
import sklearn.preprocessing

def detect_local_minima(arr):
    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape),7)
    local_min = (ndimage.filters.minimum_filter(arr, footprint=neighborhood)==arr)
    background = (arr==0)
    eroded_background = ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = np.logical_xor(local_min ,eroded_background)
    return np.where(detected_minima) 

def warpImgToCylindrical(img,f): #warp img
    (height,width,color) = np.shape(img)
    img_transformed = np.zeros((height,width,color)) #x,y,f,rgb
    for x in range(width): #x
        for y in range(height): #y
            angle = math.atan((x-int(width/2))/f)
            h_new = (y-int(height/2))/math.sqrt(math.pow((x-int(width/2)),2)+math.pow(f,2))
            x_new = int(f*angle)
            y_new = int(f*h_new) 
            img_transformed[y_new+int(height/2),x_new+int(width/2),:] = img[y,x,:]

    return img_transformed  

def warpToCylindrical(feature,focal_length, mode): #warp feature
    feature_warp = np.zeros((n, 2000, 2))
    #feature
    if(mode == False):
        for i in range(n): #img num
            for j in range(int(feature_num[i])): # feature point
                x = feature[i][j][1] #0:height,1:width
                y = feature[i][j][0]
                angle = math.atan((x-int(w/2))/focal_length[i])
                height = (y-int(h/2))/math.sqrt(math.pow((x-int(w/2)),2)+math.pow(focal_length[i],2))
                x_new = int(focal_length[i]*angle)
                y_new = int(focal_length[i]*height) 
                feature_warp[i][j][0] = y_new + int(h/2)
                feature_warp[i][j][1] = x_new + int(w/2)
    else:
        for i in range(n): #img num
            for j in range(int(final_feature_num[i])): # feature point
                x = feature[i][j][1] #0:height,1:width
                y = feature[i][j][0]
                angle = math.atan((x-int(w/2))/focal_length[i])
                height = (y-int(h/2))/math.sqrt(math.pow((x-int(w/2)),2)+math.pow(focal_length[i],2))
                x_new = int(focal_length[i]*angle)
                y_new = int(focal_length[i]*height) 
                feature_warp[i][j][0] = y_new + int(h/2)
                feature_warp[i][j][1] = x_new + int(w/2)
    return feature_warp

def ransac(data,tranformed_data,model,data_num,ransac_n,k,t,d,debug=False,return_all=False): 
#n = 5 要取幾個點, k = 5000 iterations, t = 7e4 threshold for inlier selection, d = 50 numbers of inliers
    iterations = 0
    #best_fit = 0
    #best_err = np.inf
    #best_inlier_idxs = None
    vote = np.zeros((k))
    vote_model = np.zeros((k, 2, 3))
    maybe_inliers = np.zeros((ransac_n, 2))
    test_points = np.zeros((data_num - ransac_n, 2))

    while(iterations < k):
        maybe_idxs,test_idxs = random_partition(ransac_n,data_num)
        for i in range(ransac_n):
            maybe_inliers[i] = data[int(maybe_idxs[i])]
        for i in range(data_num - ransac_n):
            test_points[i] = data[int(test_idxs[i])]
        maybe_inliers_output = tranformed_data[maybe_idxs]
        test_points_output = tranformed_data[test_idxs]
        maybe_model = model.fit(maybe_inliers,maybe_inliers_output)
        vote_model[iterations] = maybe_model
        vote_idxs, vote[iterations] = model.get_error_idxs(test_points,test_points_output,maybe_model,t,data_num-2)
 
        iterations += 1

    best_model_idxs = np.argmax(vote)
    best_model = vote_model[best_model_idxs]

    return best_model


def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1,idxs2    

class LinearLeastSquaresModel:
    """
    linear system solved using linear least squares
    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """  
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self,data,transformed_data):
        h_translate = transformed_data[0][0] - data[0][0]
        w_translate = transformed_data[0][1] - data[0][1]
        #要算translation matrix, M * A = B 
        M = np.array([[1,0,h_translate],
                      [0,1,w_translate]])
        return M
    def get_error_idxs(self,data,transformed_data,model,threshold, data_num):
        vote_idxs = [] #store index which tranformed point is also active
        for i in range(data_num):
            h_test = data[i][0]
            w_test = data[i][1]
            matrix = np.array([[h_test,w_test,1]]).T
            matrix_predict = np.dot(model,matrix)

            h_correct = transformed_data[i][0]
            w_correct = transformed_data[i][1]
            matrix_correct = np.array([[h_correct,w_correct]]).T

            err_per_point = np.sum(np.power(matrix_correct - matrix_predict, 2))
            if(err_per_point < threshold):
                vote_idxs.append(i)

        return vote_idxs, len(vote_idxs)

    def get_error(self,data,transformed_data,model,data_num):
        error = [] #count error 
        for i in range(data_num):
            h_test = data[i][0]
            w_test = data[i][1]
            matrix = np.array([[h_test,w_test,1]]).T
            matrix_predict = np.dot(model,matrix)

            h_correct = transformed_data[i][0]
            w_correct = transformed_data[i][1]
            matrix_correct = np.array([[h_correct,w_correct]]).T

            error.append(np.sum(np.power(matrix_correct - matrix_predict, 2)))

        error = np.array(error)
        error_sum = np.sum(error)
        return error_sum

def blendImages(img1, img2, width, ori_height, new_height):
    result = np.zeros((h+60, width, 3),dtype = np.uint8)
    for i in range(width):
        rate = (i+1) / width
        if (i < int(width/2)):
            for j in range(h):
                for k in range(3):
                    if(img1[j, w-width+i, k] < 10 or img2[j, i, k] < 10):
                        result[ori_height+j, i, k] = img1[j, w-width+i, k]
                    else:
                        result[ori_height+j, i, k] = int((1 - rate) * img1[j, w-width+i, k] + rate * img2[j, i, k])
        else:
            for j in range(h):
                for k in range(3):
                    if(img1[j, w-width+i, k] < 10 or img2[j, i, k] < 10):
                        result[new_height+j, i, k] = img2[j, i, k]
                    else:    
                        result[new_height+j, i, k] = int((1 - rate) * img1[j, w-width+i, k] + rate * img2[j, i, k])
    return result

'''
h = 512
w = 384
n = 18  # number of images
'''
h = 600
w = 400
n = 12

# Read all images
images = []
images_color = []
for i in range(n): 
    #read img
    #img = cv2.imread("csie/"+str(i)+'.JPG', 0)
    img = cv2.imread("img/"+str(i)+'.JPG', 0)
    images.append(img)
    #img_color = cv2.imread("csie/"+str(i)+'.JPG')
    img_color = cv2.imread("img/"+str(i)+'.JPG')
    images_color.append(img_color)

ori_img = np.array(images)
ori_img_color = np.array(images_color) 

# DoG filter
Sigma = 1.3
s = 3
k = math.pow(2, 1/s)
DoG = np.zeros((n, 3, h, w))
for i in range(n):
    s0 = ndimage.gaussian_filter(ori_img[i], sigma = Sigma)
    s1 = ndimage.gaussian_filter(ori_img[i], sigma = math.pow(k, 1) * Sigma)
    s2 = ndimage.gaussian_filter(ori_img[i], sigma = math.pow(k, 2) * Sigma)
    s3 = ndimage.gaussian_filter(ori_img[i], sigma = math.pow(k, 3) * Sigma)
    # multiply by sigma to get scale invariance
    DoG[i][0] = s1 - s0
    DoG[i][1] = s2 - s1
    DoG[i][2] = s3 - s2

feature = np.zeros((n, 2000, 2))
feature_num = np.zeros(n)
HoG = np.zeros((n, 2000, 128))
feature_vector = np.zeros((n, 2000, 128))
feature_match = np.zeros((n, 2000, 2))
for j in range(n):
    detect_point = np.zeros((h*w, 2))
    point = detect_local_minima(DoG[j])
    num ,num_point = np.shape(point)
    num_detect_point = 0
    for i in range(num_point):
        if(point[0][i] == 1):
            detect_point[num_detect_point] = [point[1][i], point[2][i]]
            num_detect_point = num_detect_point + 1

    gradient = np.zeros((3,1))
    gradient_2 = np.zeros((3,3))
    r = 5

    feature_point = []
    for i in range(num_detect_point):
        x = int(detect_point[i][0])
        y = int(detect_point[i][1])
        if( 0 < x and x < h-1 and 1 < y and y < w-1):
            gradient_2[0][0] = DoG[j][1][x+1][y] - 2 * DoG[j][1][x][y] + DoG[j][1][x-1][y]
            gradient_2[0][1] = (DoG[j][1][x+1][y+1] + DoG[j][1][x-1][y-1] - DoG[j][1][x-1][y+1] - DoG[j][1][x+1][y-1]) / 4
            gradient_2[0][2] = (DoG[j][2][x+1][y] + DoG[j][0][x-1][y] - DoG[j][2][x-1][y] - DoG[j][0][x+1][y]) / 4
            gradient_2[1][0] = gradient_2[0][1]
            gradient_2[1][1] = DoG[j][1][x][y+1] - 2 * DoG[j][1][x][y] + DoG[j][1][x][y-1]
            gradient_2[1][2] = (DoG[j][2][x][y+1] + DoG[j][0][x][y-1] - DoG[j][0][x][y+1] - DoG[j][2][x+1][y-1]) / 4
            gradient_2[2][0] = gradient_2[0][2]
            gradient_2[2][1] = gradient_2[1][2]
            gradient_2[2][2] = DoG[j][2][x][y] - 2 * DoG[j][1][x][y] + DoG[j][0][x][y]

            gradient[0] = (DoG[j][1][x+1][y] - DoG[j][1][x-1][y]) / 2
            gradient[1] = (DoG[j][1][x][y+1] - DoG[j][1][x][y-1]) / 2
            gradient[2] = (DoG[j][2][x][y] - DoG[j][0][x][y]) / 2

            if(np.linalg.det(gradient_2) != 0):
                real_x = -np.dot(np.linalg.inv(gradient_2) , gradient)
                if(real_x[0] < 0.5 and real_x[1] < 0.5 and real_x[2] < 0.5):
                    Difference = DoG[j][1][x][y] +  np.dot(gradient.T , real_x) / 2
                    if(Difference > 10):                    
                        Tr_H = gradient_2[0][0] + gradient_2[1][1]
                        Det_H = (gradient_2[0][0] * gradient_2[1][1]) - math.pow(gradient_2[0][1], 2)
                        v = math.pow(r+1, 2) / r
                        if(Det_H != 0):
                            if((math.pow(Tr_H, 2) / Det_H) < v):
                                feature_point.append([x, y]) 

    feature_num[j] = len(feature_point)
    for a in range(int(feature_num[j])):
        feature[j][a] = np.array(feature_point[a])

    L = ndimage.gaussian_filter(ori_img[j], sigma = Sigma)
    L_pad = np.pad(L, ((1, 1),(1, 1)), 'edge')
    magnitude = np.zeros((h, w))
    theta = np.zeros((h, w))
    z = 0
    for a in range(1, h+1):
        for b in range(1, w+1):
            magnitude[a-1][b-1] = math.sqrt(math.pow(L_pad[a+1][b] - L_pad[a-1][b], 2) + math.pow(L_pad[a][b+1] - L_pad[a][b-1], 2))
            if(math.isnan((L_pad[a][b+1] - L_pad[a][b-1]) / (L_pad[a+1][b] - L_pad[a-1][b]))):
                theta[a-1][b-1] = int((np.arctan(float('inf')) * 180) / math.pi)
            else:
                theta[a-1][b-1] = int((np.arctan((L_pad[a][b+1] - L_pad[a][b-1]) / (L_pad[a+1][b] - L_pad[a-1][b])) * 180) / math.pi)
            
            if((L_pad[a][b+1] < L_pad[a][b-1]) and (L_pad[a+1][b] > L_pad[a-1][b])):
                theta[a-1][b-1] = 180 - theta[a-1][b-1]
            elif((L_pad[a][b+1] < L_pad[a][b-1]) and (L_pad[a+1][b] < L_pad[a-1][b])):
                theta[a-1][b-1] = 180 + theta[a-1][b-1]
            elif((L_pad[a][b+1] > L_pad[a][b-1]) and (L_pad[a+1][b] < L_pad[a-1][b])):
                theta[a-1][b-1] = 360 - theta[a-1][b-1]


    # 取 windows 16*16  the center is (8,8)
    area_mag = np.zeros((16, 16))
    area_theta = np.zeros((16, 16))
    magnitude_pad = np.pad(magnitude, ((8, 8),(8, 8)), 'edge')
    theta_pad = np.pad(theta, ((8, 8),(8, 8)), 'edge')
    for i in range(len(feature_point)):
        x, y = feature_point[i]
        x = x + 8
        y = y + 8
        a1 = 0
        for a in range(x-8, x+8):
            b1 = 0
            for b in range(y-8, y+8): 
                #area[a1][b1] = array_pad[a][b]
                area_theta[a1][b1] = theta_pad[a][b]
                area_mag[a1][b1] = magnitude_pad[a][b]
                b1 = b1 + 1
            a1 = a1 + 1

        for a in range(16):
            for b in range(16):
                # divide to 16 blocks
                index_block = 4 * int(int(a) / 4) + int(int(b) / 4)
                index_theta = int(area_theta[a][b] / 45) % 8
                index = 8 * index_block + index_theta
                HoG[j][i][index] += area_mag[a][b]
        
    feature_vector[j] = sklearn.preprocessing.normalize(HoG[j], norm='l2',axis=1)  
for j in range(n):
    # j images, i feature points
    for i in range(int(feature_num[j])):
        Defference_0 = np.sum(np.power(feature_vector[(j+1)%n][0] - feature_vector[j][i], 2))
        Defference_1 = np.sum(np.power(feature_vector[(j+1)%n][1] - feature_vector[j][i], 2))
        if(Defference_0 < Defference_1):
            index_min = feature[(j+1)%n][0]
            index_second_min = feature[(j+1)%n][1]
            min_Defference = Defference_0
            min_second_Defference = Defference_1
        else:
            index_min = feature[(j+1)%n][1]
            index_second_min = feature[(j+1)%n][0]
            min_Defference = Defference_1
            min_second_Defference = Defference_0

        for a in range(2, int(feature_num[(j+1)%n])):
            Defference = np.sum(np.power(feature_vector[(j+1)%n][a] - feature_vector[j][i], 2))
            if(Defference < min_Defference):
                min_second_Defference = min_Defference
                index_second_min = index_min
                min_Defference = Defference
                index_min = feature[(j+1)%n][a]
            elif(Defference < min_second_Defference):
                min_second_Defference = Defference
                index_second_min = feature[(j+1)%n][a]

        dis_min = abs(index_min[0] - feature[j][i][0])
        dis_second_min = abs(index_second_min[0] - feature[j][i][0])
        if(abs(dis_min - dis_second_min) > 0.8):
            if(abs(index_min[0] - feature[j][i][0]) < 10):
                feature_match[j][i] = index_min

            
#img_rotate = np.zeros((n, w, h, 3), dtype = np.uint8)

final_feature_num = np.zeros(n)
final_feature = np.zeros((n, 200, 2))
final_feature_matching = np.zeros((n, 200, 2))
for j in range(n):
    a = []
    b = []
    #combine = np.zeros((h, w*2 + 50, 3), dtype = np.uint8)
    #combine[0:h, 0:w] = ori_img_color[j]
    #combine[0:h, w+ 50:w*2 + 50] = ori_img_color[(j+1)%n]
    for i in range(int(feature_num[j])):
        if(feature_match[j][i][0] != 0 and feature_match[j][i][1] != 0):
            if((int(feature[j][i][1])) > 250 and int(feature_match[j][i][1]) < 150 ):
                a.append([int(feature[j][i][0]), int(feature[j][i][1])])
                b.append([int(feature_match[j][i][0]), int(feature_match[j][i][1])])

                #rr, cc = draw.line(int(feature[j][i][0]), int(feature[j][i][1]), int(feature_match[j][i][0]), w + 50 + int(feature_match[j][i][1]))
                #draw.set_color(combine, [rr, cc], [0, 0,255])
    
    final_feature_num[j] = len(a)
    for i in range(int(final_feature_num[j])):
        final_feature[j][i] = np.array(a[i])
        final_feature_matching[j][i] = np.array(b[i])
    '''
    for i in range(int(feature_num[j])):
        rr, cc = draw.circle_perimeter(int(feature[j][i][0]),int(feature[j][i][1]), 5)
        draw.set_color(combine, [rr, cc], [0, 0, 255])
    
    for i in range(int(feature_num[(j+1)%n])):
        rr, cc = draw.circle_perimeter(int(feature[(j+1)%n][i][0]), w + 50 + int(feature[(j+1)%n][i][1]), 5)
        draw.set_color(combine, [rr, cc], [0, 0, 255])

    imwrite('feature_detection_+'+str(j), combine)
    cv2.imshow("rotate", combine)
    cv2.waitKey(0)
    '''
      


#warp to cylindrical coordinate
focal_length = [460.678, 455.056, 458.013, 461.428, 461.46, 463.49, 464.426, 465.268, 462.351, 463.327, 463.803, 463.678]
img_cylindrical = np.zeros((n,h,w,3)) #store transformed images (project to cylindrical pos)
for i in range(n):
    #img_cylindrical[i,:,:,:] = warpImgToCylindrical(ori_img_color[i],int(focal_length[i]))
    img_cylindrical[i,:,:,:] = warpImgToCylindrical(ori_img_color[i], float(focal_length[i]))

#feature_warp = warpToCylindrical(final_feature,list(map(int,focal_length)))
#feature_match_warp = warpToCylindrical(final_feature_matching,list(map(int,focal_length)))
feature_warp = warpToCylindrical(feature,list(map(int,focal_length)), False)
final_feature_warp = warpToCylindrical(final_feature,list(map(int,focal_length)), True)
feature_match_warp = warpToCylindrical(final_feature_matching,list(map(int,focal_length)), True)

'''
# 作圖 (顯示兩張 warped image 的 feature matching)
for j in range(n):
    combine = np.zeros((h, w*2 + 50, 3), dtype = np.uint8)
    combine[0:h, 0:w] = img_cylindrical[j]
    combine[0:h, w+ 50:w*2 + 50] = img_cylindrical[(j+1)%n]
    for i in range(int(final_feature_num[j])):
        rr, cc = draw.line(int(final_feature_warp[j][i][0]), int(final_feature_warp[j][i][1]), int(feature_match_warp[j][i][0]), w + 50 + int(feature_match_warp[j][i][1]))
        draw.set_color(combine, [rr, cc], [0, 0,255])

    for i in range(int(feature_num[j])):
        rr, cc = draw.circle_perimeter(int(feature_warp[j][i][0]),int(feature_warp[j][i][1]), 5)
        draw.set_color(combine, [rr, cc], [0, 0, 255])
    
    for i in range(int(feature_num[(j+1)%n])):
        rr, cc = draw.circle_perimeter(int(feature_warp[(j+1)%n][i][0]), w + 50 + int(feature_warp[(j+1)%n][i][1]), 5)
        draw.set_color(combine, [rr, cc], [0, 0, 255])

    cv2.imshow("warp", combine)
    cv2.waitKey(0)
'''

#ransac
translation_matrix = np.zeros((n,2,3))
# translation_data = np.zeros((18,2,3))
debug = True
for i in range(n):
    input_columns = final_feature_warp[i].shape[0] # original position
    output_columns = feature_match_warp[i].shape[0] # paired position      
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)
    ransac_fit = ransac(final_feature_warp[i],feature_match_warp[i],model,int(final_feature_num[i]),
                                    2, 5000, 5, 50, # misc. parameters
                                    debug=debug,return_all=True)
    translation_matrix[i] = ransac_fit


ori_height = 30
padding = 0

for j in range(n):
    dis_h = int(translation_matrix[j][0][2]) # 正的為向下
    if(abs(dis_h) > 5):
        dis_h = 0
    dis_w = w - abs(int(translation_matrix[j][1][2])) # 正的為向左
    if(j == 0):
        if(dis_w % 2 == 1):
            new_img = np.zeros((h+60, w-int(dis_w/2)-1, 3),dtype = np.uint8)
            new_width = new_img.shape[1]
            new_height = 30
            new_img[new_height:h+new_height, 0:new_width, :] = img_cylindrical[j][:, 0:new_width ,:]

        else:
            new_img = np.zeros((h+60, w-int(dis_w/2), 3),dtype = np.uint8)
            new_width = new_img.shape[1]
            new_height = 30
            new_img[new_height:h+new_height, 0:new_width, :] = img_cylindrical[j][:, 0:new_width ,:]
    else:
        if(dis_w % 2 == 1):
            add_img = np.zeros((h+60, w-int(dis_w/2)-1 - padding, 3), dtype=np.uint8)
            new_img = np.concatenate((new_img, add_img), axis = 1)
            new_width = new_img.shape[1]
            new_height = ori_height - dis_h
            new_img[new_height:h+new_height, ori_width:new_width, :] = img_cylindrical[j][:, padding:w-int(dis_w/2)-1, :]

        else:
            add_img = np.zeros((h+60, w-int(dis_w/2) - padding, 3), dtype=np.uint8)
            new_img = np.concatenate((new_img, add_img), axis = 1)
            new_width = new_img.shape[1]
            new_height = ori_height - dis_h
            new_img[new_height:h+new_height, ori_width:new_width, :] = img_cylindrical[j][:, padding:w-int(dis_w/2), :]
    
    # blending
    if(j > 0):
        before_width = w - abs(int(translation_matrix[j-1][1][2])) # 正的為向下
        #before_width = 80
        blend = np.zeros((h+60, before_width, 3))
        blend = blendImages(img_cylindrical[j-1], img_cylindrical[j], before_width, ori_height, new_height)
        if(before_width % 2 == 1):    
            new_img[:, ori_width-int(before_width/2)+1:ori_width+int(before_width/2), :] = blend[:, 1:before_width-1, :]
        else:
            new_img[:, ori_width-int(before_width/2)+1:ori_width+int(before_width/2)-1, :] = blend[:, 1:before_width-1, :]
    
    ori_height = new_height
    ori_width = new_width
    if(dis_w % 2 == 1):
        padding = int(dis_w/2) + 1
    else:
        padding = int(dis_w/2)


cv2.imwrite("result.jpg", new_img)



