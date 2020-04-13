import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np
import cv2
import math
from preprocess_data import *

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
            img_path  = os.path.join(cur_path, f[:f.find('.')] + '.png')
            origin_img_path = os.path.join(cur_path, f[:f.find('.')] + '_origin.png')
            print(f)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            print(points)
            # matrix = np.load(file_path)
            points = points + abs(points.min(axis = 0))
            points *= [1000, 1000, 1]

            size = points.max(axis = 0)[:-1].astype(int) + 1

            matrix = np.zeros((size[1], size[0]))
            cnt = np.zeros((size[1], size[0]))

            for p in points:
                x = int(p[0])
                y = int(p[1])
                z = p[2]
                # matrix[y][x] = max(matrix[y][x], z)
                matrix[y][x] = matrix[y][x] + z
                cnt[y][x] += 1
            cnt = np.maximum(cnt, 1)
            matrix /= cnt
            matrix = matrix.astype(np.float32)
            matrix = cv2.medianBlur(matrix, 5)
            matrix = straightening_img(matrix)
            matrix = remove_black_border_numpy(matrix)

            np.save(np_path, matrix)
            # cv2.imwrite(img_path, sobel)
            # cv2.imwrite(origin_img_path, matrix)

generate_depth_image('./PointsPoisson/Test', './DepthImage/Test')