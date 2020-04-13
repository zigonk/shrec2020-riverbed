import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np
import cv2
import seaborn as sns
import math
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from preprocess_data import *
from find_maximum_square import *
from skimage import img_as_float
from skimage import io, color, morphology


out = open('./Data/stat_size.txt', "w")

def find_local_maxima(img, save_path):
    im = img_as_float(img)
    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    # image_max = ndi.maximum_filter(im, size=1, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=3)

    # display results
    # fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    # ax=axes.ravel()
    plt.clf()
    plt.imshow(im, cmap=plt.cm.gray)
    # ax[0].autoscale(False)
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # ax[0].axis('off')
    # ax[0].set_title('Peak local max')

    # fig.tight_layout()

    plt.savefig(save_path)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def angle_between_vector(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    return angle

def orientation(p1, p2, p3):
    # to find the orientation of
    # an ordered triplet (p1,p2,p3)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    val = (float(p2.y - p1.y) * (p3.x - p2.x)) - \
           (float(p2.x - p1.x) * (p3.y - p2.y))
    if (val > 0):
        return 1
    elif (val < 0):
        return 2
    else:
        return 0

def morphologyEx(img):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  # erosion = cv2.erode(closing,kernel,iterations = 1)
  return img

def color_quantization(img, bins=16):
  # img = cv2.cvtColor(img , cv2.COLOR_GRAY2RGB)

  # convert to np.float32
  Z = np.float32(img)
  Z = Z.flatten()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = bins
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((img.shape))
  # res2 = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
  return res2

def remove_border(img):
  h, w = img.shape
  crop_top = 0
  crop_bot = h
  crop_left = 0
  crop_right = w
  max_val = img.max()
  border_w = w // 4
  border_h = h // 4
  for j in range(border_w, w - border_w):
    for i in range(1, h):
      if (img[i][j] != max_val) and (img[i - 1][j] == max_val):
        crop_top = max(crop_top , i)
        break
  for j in range(border_w, w - border_w):
    for i in range(h - 2, 0, -1):
      if (img[i][j] != max_val ) and (img[i+1][j] == max_val):
        crop_bot = min(crop_bot, i)
        break
  for i in range(border_h, h - border_h):
    for j in range(1, w):
      if (img[i][j] != max_val ) and (img[i][j - 1] == max_val):
        crop_left = max(crop_left, j)
        break
  for i in range(border_h, h - border_h):
    for j in range(w - 2, 0, -1):
      if (img[i][j] != max_val ) and (img[i][j + 1] == max_val):
        crop_right = min(crop_right, j)
        break
  return img[crop_top:crop_bot, crop_left:crop_right]

def cvtBinary(img):
  max_val = img.max()
  min_val = img.min()
  img[img == max_val] = 255
  img[img == min_val] = 0
  return img

def blur_multiple_time(img = [],kernel = 3, num_iters = 3):
  for i in range(num_iters):
    img = cv2.medianBlur(img, kernel)
  return img

def fill_by_circle(img, sz, thresh_hold):
  # Calculate number position that can filled by a circle
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sz,sz))
  accept_num_points = kernel.sum() * (1 - thresh_hold)
  h, w = img.shape
  res = 0
  for i in range(h - sz):
    for j in range(w - sz):
      corr = np.bitwise_and(kernel, img[i:i+sz, j:j+sz])
      if (corr.sum() >= accept_num_points):
        res += 1
  return res

def fill_by_circle_topdown(img, sz, thresh_hold):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sz,sz))
  accept_num_points = kernel.sum() * (1 - thresh_hold)
  h, w = img.shape
  res = 0
  for i in range(h - sz):
    for j in range(w - sz):
      corr = np.bitwise_and(kernel, img[i:i+sz, j:j+sz])
      if (corr.sum() >= accept_num_points):
        res += 1
  return res

def classify_1(x):
  if (len(x) <= 3 or x[1] < 1e-3):
    class_num = 1
  elif (len(x) <= 4 or x[2] < 1e-3):
    class_num = 4
  elif (len(x) <= 6 or x[4] < 1e-3):
    class_num = 6
  else:
    class_num = 8

def generate_depth_image(model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, _, files in os.walk(model_path):
        cur_path = os.path.join(save_path, root[len(model_path):])
        if not os.path.exists(cur_path):
          os.makedirs(cur_path)
        files.sort()
        for f in files:
            file_path = os.path.join(root, f)
            fpath  = os.path.join(cur_path, f[:f.find('.')])
            np_path  = os.path.join(cur_path, f[:f.find('.')] + '.npy')
            img_path  = os.path.join(cur_path, f[:f.find('.')] + '_quantize.png')
            origin_img_path = os.path.join(cur_path, f[:f.find('.')] + '_origin.png')
            print(f)
            matrix = np.load(file_path)
            matrix = matrix.astype(np.float32)
            matrix = cv2.medianBlur(matrix, 5)
            # matrix = straightening_img(matrix)
            # matrix = remove_black_border_numpy(matrix)
            blur   = cv2.GaussianBlur(matrix ,(11, 11), 1, borderType=cv2.BORDER_CONSTANT)
            hpass  = matrix - blur
            hpass *= 1000000
            # hpass /= hpass.max()
            hpass[hpass > 127] = 127
            hpass[hpass < -127] = -127
            hpass += 127
            hpass = hpass.astype(np.uint8)
            hpass = remove_border(hpass)
            quant = hpass.copy()
            # quant = remove_border(hpass)
            quant = cv2.medianBlur(quant, 3)
            # quant = cv2.GaussianBlur(quant, (3, 3), 1, borderType=cv2.BORDER_CONSTANT)
            # hpass = cv2.Canny(hpass, 50, 1, L2gradient=True)
            # quant = color_quantization(quant, 16)
            # quant = color_quantization(quant, 8)
            # quant = color_quantization(quant, 4)
            quant = color_quantization(quant, 2)
            quant = cvtBinary(quant)
            quant = blur_multiple_time(quant, 3, 2)

            x = []
            i = 31
            while True:
              i -= 2
              num_c = fill_by_circle(quant, i, 0.01)
              if (num_c == 0):
                continue
              x.append(fill_by_circle(quant, i, 0.01), i)
            w, h = quant.shape
            sz = w * h
            for num_c, type_c in x:
              print(num_c, type_c, num_c*type_c / sz)


            # x = []
            # i = 7
            # while True:
            #   i += 2
            #   x.append(fill_by_circle(quant, i, 0.01))
            #   if x[-1] == 0:
            #     break
            # print(x)
            # class_num = classify_1(x)


            # plt.plot(x)
            # plt.show()
            # quant = morphologyEx(quant)
            # quant = morphologyEx(quant)
            # thin = np.asarray((1 - morphology.thin(1 - img_as_float(quant)).astype(np.uint8)) * 255)
            # print(thin.shape)
            # hist, visualize = find_maximum_square(thin)
            # plt.clf()
            # plt.hist(hist, range=(5,30), bins=27, density=True)
            # skeleton = (1 - morphology.skeletonize(1 - img_as_float(quant)).astype(int)) * 255
            # hpass = morphologyEx(hpass)
            # quant = cv2.medianBlur(quant, 5)
            # matrix += abs(matrix.min())
            # print(matrix.min())
            # print(matrix.max())
            # matrix /= matrix.max()
            # bins = 1
            # x = np.zeros((int(matrix.shape[0]/bins) + 1, matrix.shape[1]))
            # sz = matrix.shape
            # matrix = cv2.GaussianBlur(matrix ,(5, 5), 1, borderType=cv2.BORDER_CONSTANT)
            # for t in range(matrix.shape[0]):
            #     x[int(t/bins)] += matrix[t]
            # x /= bins
            # plt.show()
            # matrix = cv2.blur(matrix ,(7, 7),  borderType=cv2.BORDER_CONSTANT)
            # matrix = cv2.blur(matrix ,(9, 9), borderType=cv2.BORDER_CONSTANT)
            # matrix = cv2.blur(matrix ,(11, 11), borderType=cv2.BORDER_CONSTANT)
            # matrix = cv2.blur(matrix ,(17, 17), borderType=cv2.BORDER_CONSTANT)
            # sobelx = cv2.Sobel(hpass,cv2.CV_64F,1,0,ksize=5)
            # sobely = cv2.Sobel(hpass,cv2.CV_64F,0,1,ksize=5)
            # print(sobelx.min())
            # sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
            # sobel = sobel[50:sz[0] - 50, 50: sz[1] - 50]
            # print(sobel.shape)
            # x = sobel.flatten()
            # x /= (sobel.shape[0] * sobel.shape[1])
            # # plt.clf()
            # # x = x[(x >= 0) & (x < 0.15)]
            # if (fpath.find("Class1_1") != -1 or fpath.find("Class4_1") != -1 or fpath.find("Class6_1") != -1 or fpath.find("Class8_1") != -1):
            #     sns.lineplot(data = x[10], label = f)
            # # # filtered_img = matrix - blur
            # filtered_img = np.copy(matrix)
            # sobel *= 255/sobel.max()
            # sobel[sobel > 255] = 255
            # ratio = 255 / matrix.max()
            # matrix *= ratio
            # # filtered_img *= 128 / max(abs(filtered_img.min()), filtered_img.max())
            # # filtered_img += max(abs(filtered_img.min()), filtered_img.max())
            # filtered_img += abs(filtered_img.min())
            # filtered_img *= 255 / filtered_img.max()
            # filtered_img = filtered_img.astype('uint8')
            # filtered_img = np.minimum(filtered_img, 255)
            # filtered_img = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #             cv2.THRESH_BINARY,11,1)

            # filtered_img = 255-filtered_img
            # filtered_img = morphologyEx(filtered_img)
            # hist, demo_img = find_maximum_square(filtered_img)
            # hist = np.asarray(hist)
            # filtered_img = cv2.medianBlur(filtered_img, 3)
            # _, contours, _ = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS )
            # filtered_img = cv2.drawContours(filtered_img, contours, -1, (0,255,0), 1)
            # np.save(np_path, hist)
            class_path = os.path.join(cur_path, str(class_num))
            if not os.path.exists(class_path):
              os.makedirs(class_path)
            img_path = os.path.join(class_path, f[:f.find('.')])
            plt.clf()
            plt.subplot(211)
            plt.imshow(quant, cmap='gray')
            plt.subplot(212)
            plt.imshow(hpass, cmap='gray')
            plt.savefig(img_path)
            # cv2.imwrite(img_path + '.png' , quant)
            # cv2.imwrite(fpath + '_thin.png', thin)
            # cv2.imwrite(fpath + '_skeleton.png', skeleton)
            # plt.savefig(fpath + '_hist.png')
            # cv2.imwrite(fpath + '.png', hpass)
            # cv2.imwrite(fpath + '_sq.png', visualize)
            # cv2.imwrite(fpath + '_.png', quant)
            # cv2.imwrite(origin_img_path, hpass)

generate_depth_image('./DepthImage/Train', './Data/ReTest_HighPass/Circle/Train')