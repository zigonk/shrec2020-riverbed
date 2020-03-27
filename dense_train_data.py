import os
import cv2
import numpy as np

train_data_path = './PureData/Train'
valid_data_path = './PureData/Valid'

crop_size = 256
image_idx = 0

def remove_black_border(gray):
  _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
  contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnt = contours[0]
  x,y,w,h = cv2.boundingRect(cnt)
  crop = gray[y:y+h,x:x+w]
  return crop

def check_quality(ratio_black_pix, ratio_size):
  if (ratio_black_pix > 0.7): return False
  if (ratio_size > 1.8): return False
  if (ratio_size > 1.5 and ratio_black_pix > 0.4): return False
  return True

def straightening_img(image):
  coords = np.column_stack(np.where(image > 0))
  angle = cv2.minAreaRect(coords)[-1]

  if angle < -45:
      angle = -(90 + angle)
  else:
      angle = -angle

  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=0)
  return rotated

def dense_sampling(data_path, img):
  global image_idx
  height = img.shape[0]
  width = img.shape[1]
  for i in range(0, height - crop_size, 50):
    for j in range(0, width - crop_size, 50):
      image_idx += 1
      crop_img = img[i : i + crop_size, j : j + crop_size]
      img_path = os.path.join(data_path, '{:05d}.png'.format(image_idx))
      cv2.imwrite(img_path, crop_img)

def pre_processing(data_path):
  global image_idx
  for f in os.listdir(data_path):
    class_data_path = os.path.join(data_path, f)
    print(class_data_path)
    image_idx = 0
    list_img = os.listdir(class_data_path)
    for img_file in list_img:
      img_path = os.path.join(class_data_path, img_file)
      img = cv2.imread(img_path, 0)
      straight_img = straightening_img(img)
      cutted_img = remove_black_border(straight_img)
      # remove image that have low quality (side view or back view)
      np_img = np.asarray(cutted_img)
      ratio = np_img.shape[0] / np_img.shape[1]
      ratio = max(ratio, 1/ratio)
      np_img = np_img.flatten()
      np_img = np_img == 0
      num_black_pix = np_img.sum()
      total_pix = np_img.shape[0]
      if (check_quality(num_black_pix/total_pix, ratio)):
        dense_sampling(class_data_path, cutted_img)
      os.remove(img_path)

pre_processing(valid_data_path)