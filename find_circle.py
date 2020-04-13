import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np
import cv2
import math

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
  return img[crop_top + 10 : crop_bot - 10, crop_left + 10 : crop_right - 10]

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
  for i in range(1, h - sz - 1):
    for j in range(1, w - sz - 1):
      corr = np.bitwise_and(kernel, img[i:i+sz, j:j+sz])
      if (corr.sum() >= accept_num_points):
        if (img[i - 1: i+sz+1, j:j-1+sz+1].max() == 2):
          img[i:i+sz, j:j+sz] = corr * 2
          continue
        else:
          res += 1
          img[i:i+sz, j:j+sz] = corr * 2
  return res, img

def classify_1(x):
  if (len(x) <= 3 or x[1] < 1e-3):
    class_num = 1
  elif (len(x) <= 4 or x[2] < 1e-3):
    class_num = 4
  elif (len(x) <= 6 or x[4] < 1e-3):
    class_num = 6
  else:
    class_num = 8
  return class_num

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
            blur   = cv2.GaussianBlur(matrix ,(11, 11), 1, borderType=cv2.BORDER_CONSTANT)
            hpass  = matrix - blur
            hpass *= 1000000
            hpass[hpass > 127] = 127
            hpass[hpass < -127] = -127
            hpass += 127
            hpass = hpass.astype(np.uint8)
            hpass = remove_border(hpass)
            quant = hpass.copy()
            quant = cv2.medianBlur(quant, 3)
            quant = color_quantization(quant, 2)
            quant = cvtBinary(quant)
            quant = blur_multiple_time(quant, 3, 1)
            quant //= 255

            x = []
            i = 50
            while i > 5:
              i -= 1
              num_c, quant = fill_by_circle_topdown(quant, i, 0.05)
              if (num_c == 0 or i > 35):
                continue
              x.append((num_c, i))
            w, h = quant.shape
            sz = w * h
            class1 = class4 = class6 = class8 = 0
            for num_c, type_c in x:
              kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(type_c,type_c))
              total = kernel.sum()
              fill_ratio = round(num_c*total / sz, 4)
              if (type_c >= 19):
                class8 += fill_ratio
              elif (type_c > 15):
                class6 += fill_ratio
              elif (type_c > 13):
                class4 += fill_ratio
              elif (type_c >= 7):
                class1 += fill_ratio

            # model_num = int((f.split('.')[0]).split('_')[1])
            if (class8 > 0.02):
              class_num = 8
            elif (class6 > 0):
              class_num = 6
            elif (class4 > 0):
              class_num = 4
            elif (class1 > 0):
              class_num = 1
            # print(class1, class4, class6, class8)
            quant *= 255

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

# result = np.zeros(241)
generate_depth_image('./DepthImage/Test', './Circle_TopDown/Test')