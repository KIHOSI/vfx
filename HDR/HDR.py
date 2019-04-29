import numpy as np
import cv2
import math
from matplotlib.pyplot import plot, show, scatter, title, xlabel, ylabel, savefig
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

z_min = 0
z_max = 255

# construct radiance map
def RadianceMap(img, B, g, weight):   
    # img 是同一個channel的15張照片(15, 3448, 4592)
    rad_map = np.zeros((3448, 4592), dtype = np.float64)
    G = np.zeros(15)
    W = np.zeros(15)
    for i in range(3448):
        for j in range(4592):
            G = np.array([g[int(img[k,i, j])] for k in range(15)])
            W = np.array([weight[int(img[k,i, j])] for k in range(15)])
            sum_w = np.sum(W)
            if(sum_w > 0):
                rad_map[i, j] = np.sum(W * (G - B) / sum_w)
            else:
                rad_map[i, j] = G[8] - B[8]
    
    return rad_map

# convert RGB to Y
def gernerate_w():
    w = np.zeros(256, dtype = np.float64)
    for i in range(256):
        if(i <= 0.5 * (z_max + z_min)):
            w[i] = i - z_min
        else:
            w[i] = z_max - i
    
    w = w / np.max(w)
    
    return w

def gslove(Z, B, l, w):
    n = 256
    a1 = np.size(Z, 0)                      # a1 = N = 100
    a2 = np.size(Z, 1)                      # a2 = P = 15
    raw = a1 * a2 + (z_max - z_min)
    column = n + a1
    A = np.zeros((raw, column), dtype = np.float64)
    b = np.zeros((raw, 1), dtype = np.float64)

    # construct matrix A and matrix b
    k = 0
    for i in range(a1):
        for j in range(a2):
            temp = int(Z[i][j])
            wij = w[temp]
            A[k][temp] = wij
            A[k][n + i] = -wij
            b[k][0] = wij * B[j]
            k = k + 1

    A[k][128] = 1
    k = k + 1

    for i in range(n-2):
        A[k][i] = l * w[i]
        A[k][i + 1] = (-2) * l * w[i]
        A[k][i + 2] = l * w[i]
        k = k + 1

    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, b).reshape(356)
    
    g = x[: n]
    IE = x[n :]
    return g , IE
    
def globalToneMapping(image,gamma):    
    image_corrected = cv2.pow(image/255,1.0/gamma)
    return image_corrected

def intensityAdjustment(image, template):
    m, n, channel = image.shape
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.average(image[:, :, ch]), np.average(template[:, :, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)

    return output



# main
images = []
#讀取圖檔
for i in range(4,19):
    #read img
    img = cv2.imread('img/P10304'+str(i)+'.JPG')
    images.append(img)
    
alignMTB = cv2.createAlignMTB()
alignMTB.process(images,images)

array = np.array(images) # images shape = (15, 3448, 4592, 3)

p = 15
N = 100
w = 4592
h = 3448
inv_shutter_speed = [1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 0.125, 0.25, 0.5, 1, 2, 4, 8, 15]
lamdba = 10

# calculate weight function
weight = np.zeros(256)              # weight function for pixel value z
weight = gernerate_w()

# calculate B
ln_t = np.zeros(15)
shutter_speed = np.array(inv_shutter_speed)
for i in range(10):
    ln_t[i] = math.log(shutter_speed[i]) 

# sample point
temp = np.zeros((10, 10))
res = np.zeros((10, 10, 3, 15))
point = np.zeros((100, 15, 3))          # (number of picked pixels, image number, channel)

for i in range(p):
    res[:, :, :, i] = cv2.resize(images[i], (10, 10), interpolation = cv2.INTER_CUBIC)
    for j in range(3):
        temp = res[:, :, j, i]
        point[:, i, j] = temp.reshape(-1)

g = np.zeros((256, 3))
lnE = np.zeros((100, 3))
for i in range(3):
    g[:, i] , lnE[:, i] = gslove(point[:, :, i], ln_t, lamdba, weight)

temp = np.zeros((3448, 4592, 3))
img_rad_map = np.zeros((3, 3448, 4592))
channel_images = np.zeros((3, 15, 3448, 4592))
for i in range(3):
    for j in range(15):
        temp = images[j]
        channel_images[i, j, :, :] = temp[:, :, i]

HDR_image = np.zeros((3, 3448, 4592), dtype = np.float32)
HDR = np.zeros((3448, 4592, 3), dtype = np.float32)
for i in range(3):
    img_rad_map[i] = RadianceMap(channel_images[i], ln_t, g[:, i], weight)
    HDR_image[i] = cv2.normalize(img_rad_map[i], dst = HDR_image[i], alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    HDR[:, :, i] = HDR_image[i]

# output hdr image
cv2.imwrite('Radiance.JPG',HDR)

# global tone mapping (gamma correction)
output = np.zeros((3448, 4592),dtype=np.uint8)
img_2 = images[len(images)//2]
gamma = 0.6
image_mapped = globalToneMapping(HDR , gamma)
template = img_2
image_tuned = intensityAdjustment(image_mapped, template)
output = cv2.normalize(image_tuned,dst=output , alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imwrite("output.jpg", output)

# Tonemap using Reinhard's method to obtain 24-bit color image
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(HDR)
cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)

#plot response curve
plot(g, np.arange(256))
title('RGB Response function')
xlabel('log exposure')
ylabel('Z value')
savefig('response_curve.png')
