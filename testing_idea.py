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
    val = ((p2[1] - p1[1]) * (p3[0] - p2[0])) - \
           ((p2[0] - p1[0]) * (p3[1] - p2[1]))
    if (val > 0):
        return 1
    elif (val < 0):
        return -1
    else:
        return 1

def count_angle_orientation_and_distance(x = [], start_x = 5, end_x = 0, start_y = 5, end_y = 0):
    a = 0.3
    lr = -1

    cnt = 0
    momentum = 0
    dist_list = []
    angle_list = []
    if (end_x == 0): end_x = x.shape[0] - 5
    if (end_y == 0): end_y = x.shape[1] - 5
    last_angle = 0
    sum = 0
    # plt.plot(x[0], label=)
    for idx in range(start_x, end_x):
        for idy in range(start_y, end_y):
            p1 = np.asarray([idy - 1, x[idx][idy - 1]])
            p2 = np.asarray([idy    , x[idx][idy]])
            p3 = np.asarray([idy + 1, x[idx][idy + 1]])
            angle = angle_between_vector(p1 - p2, p3 - p2)
            orient = orientation(p1, p2, p3)
            # if (orient == -1):
            # else:
    return angle_list, dist_list

def generate_depth_image(model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, _, files in os.walk(model_path):
        cur_path = os.path.join(save_path, root[len(model_path):])
        if not os.path.exists(cur_path):
          os.makedirs(cur_path)
        files.sort()
        files = ['Class1_1.pcd', 'Class1_2.pcd', 'Class4_1.pcd', 'Class4_2.pcd', 'Class6_1.pcd', 'Class6_2.pcd', 'Class8_1.pcd', 'Class8_2.pcd']
        for f in files:
            file_path = os.path.join(root, f)
            fpath  = os.path.join(cur_path, f[:f.find('.')])
            np_path  = os.path.join(cur_path, f[:f.find('.')] + '.npy')
            img_path  = os.path.join(cur_path, f[:f.find('.')] + '.png')
            origin_img_path = os.path.join(cur_path, f[:f.find('.')] + '_origin.png')
            print(f)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            # matrix = np.load(file_path)
            points = points + abs(points.min(axis = 0))
            points *= [1000, 1000, 1]

            size = points.max(axis = 0)[:-1].astype(int) + 1
            out.write(f + '\n')
            out.write(str(size) + '\n')

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
            matrix = matrix.astype(np.float32)
            matrix = straightening_img(matrix)
            matrix = remove_black_border_numpy(matrix)
            matrix = cv2.medianBlur(matrix, 3)
            # matrix += abs(matrix.min())
            # print(matrix.min())
            # print(matrix.max())
            # matrix /= matrix.max()
            bins = 10
            n_bins = int(matrix.shape[0]/bins)
            x = np.zeros((n_bins + 1, matrix.shape[1]))
            # sz = matrix.shape
            # matrix = cv2.GaussianBlur(matrix ,(5, 5), 1, borderType=cv2.BORDER_CONSTANT)
            for t in range(matrix.shape[0]):
                x[int(t/bins)] += matrix[t]
            x /= bins
            # x *= 1000
            # plt.clf()
            # angle_list, dist_list = count_angle_orientation_and_distance(np.asarray([x[9]]), 0, 1)
            # plt.figure()
            # plt.subplot(211)
            # plt.hist(angle_list, bins=100)
            # plt.subplot(212)
            # plt.hist(dist_list, bins=7, density=True)
            # plt.axis([0, 35, 0, 1])
            # plt.savefig(fpath + '.png')
            # plt.show()
            # break
            # # # filtered_img = matrix - blur
            # # filtered_img = np.copy(matrix)
            sobelx = cv2.Sobel(matrix,cv2.CV_64F,1,0,ksize=5)
            sobely = cv2.Sobel(matrix,cv2.CV_64F,0,1,ksize=5)
            sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
            sobel *= 500
            sobel[sobel > 255] = 255
            plt.figure()
            plt.subplot(211)
            sns.lineplot(data = x[10], label = f)
            sns.lineplot(data = x[9], label = f)
            plt.subplot(212)
            plt.imshow(sobel)
            plt.hlines(105, 0, 300)
            plt.show()
            # matrix *= ratio
            # # filtered_img *= 128 / max(abs(filtered_img.min()), filtered_img.max())
            # # filtered_img += max(abs(filtered_img.min()), filtered_img.max())
            # filtered_img += abs(filtered_img.min())
            # filtered_img *= 255 / filtered_img.max()
            # filtered_img = filtered_img.astype('uint8')
            # filtered_img = np.minimum(filtered_img, 255)
            # filtered_img = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #             cv2.THRESH_BINARY,21,1)

            # _, contours, _ = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS )
            # filtered_img = cv2.drawContours(filtered_img, contours, -1, (0,255,0), 1)
            # np.save(np_path, matrix)
            # cv2.imwrite(img_path, sobel)
            # cv2.imwrite(origin_img_path, matrix)

generate_depth_image('./PointsPoisson/PointCloud/Train', './Data/CutSize/Train')
plt.show()
