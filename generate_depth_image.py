import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np
import cv2
import math

def generate_depth_image(model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, _, files in os.walk(model_path):
        cur_path = os.path.join(save_path, root[root.find('/') + 1:])
        if not os.path.exists(cur_path):
          os.makedirs(cur_path)
        for f in files:
            file_path = os.path.join(root, f)
            np_path  = os.path.join(cur_path, f[:f.find('.')] + '.npy')
            # origin_img_path = os.path.join(cur_path, f[:f.find('.')] + '_origin.png')
            print(file_path)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            points = points + abs(points.min(axis = 0))
            points *= [1000, 1000, 1]

            size = points.max(axis = 0)[:-1].astype(int) + 1

            matrix = np.zeros((size[1], size[0]))
            cnt = np.zeros((size[1], size[0]))

            for p in points:
                x = int(p[0])
                y = int(p[1])
                z = p[2]
                matrix[y][x] = matrix[y][x] + z
                cnt[y][x] += 1
            cnt = np.maximum(cnt, 1)
            matrix /= cnt
            ratio = 255 / matrix.max()
            matrix *= ratio
            matrix = matrix.astype('uint8')
            matrix = cv2.medianBlur(matrix, 5)
            sobelx = cv2.Sobel(matrix,cv2.CV_64F,1,0,ksize=5)
            sobely = cv2.Sobel(matrix,cv2.CV_64F,0,1,ksize=5)
            sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
            sobel *= 5000 / sobel.max()
            sobel[sobel > 255] = 255
            # blur = cv2.blur(matrix ,(21, 21), borderType=cv2.BORDER_CONSTANT)
            # filtered_img = matrix - blur
            # # filtered_img = np.copy(matrix)
            # filtered_img *= 128 / max(abs(filtered_img.min()), filtered_img.max())
            # filtered_img += max(abs(filtered_img.min()), filtered_img.max())
            # filtered_img = filtered_img.astype('uint8')
            # # filtered_img = np.minimum(filtered_img, 255)
            # filtered_img = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #             cv2.THRESH_BINARY,51,1)

            # _, contours, _ = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS )
            # filtered_img = cv2.drawContours(filtered_img, contours, -1, (0,255,0), 1)
            np.save(np_path, matrix)
            # cv2.imwrite(origin_img_path, matrix)

generate_depth_image('./PointsPoisson/', './Data/')