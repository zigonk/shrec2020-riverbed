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

def find_sobel(matrix):
    sobelx = cv2.Sobel(matrix,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(matrix,cv2.CV_64F,0,1,ksize=5)
    sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
    sobel *= 500
    sobel[sobel > 255] = 255
    return sobel

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

    return angle / math.pi

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

def find_roi(x = [], start_x = 5, end_x = 0, start_y = 5, end_y = 0):
    eps = 0.01
    thresh_hold = 0.00007
    if (end_x == 0): end_x = x.shape[0] - 5
    if (end_y == 0): end_y = x.shape[1] - 5
    roi = np.zeros(x.shape)
    for idx in range(start_x, end_x):
        for idy in range(start_y, end_y):
            p0 = np.asarray([idy - 1, x[idx][idy - 1]])
            p1 = np.asarray([idy    , x[idx][idy]])
            p2 = np.asarray([idy + 1, x[idx][idy + 1]])
            p3 = np.asarray([idy + 2, x[idx][idy + 2]])
            if (abs(p3[1] - p2[1]) >= eps or abs(p2[1] - p1[1]) >= eps):
                continue
            orient_cur = orientation(p0, p1, p2)
            orient_nxt = orientation(p1, p2, p3)
            if (orient_nxt == -1 and orient_cur == -1 and p1[1] <= p2[1] + 0.001):
                roi[idx][idy] = 0
            else:
                roi[idx][idy] = 1
    return roi

def create_depth_image(points):
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
    matrix = matrix.astype(np.float32)
    matrix = straightening_img(matrix)
    matrix = remove_black_border_numpy(matrix)
    matrix = cv2.medianBlur(matrix, 5)
    # matrix = cv2.GaussianBlur(matrix ,(5, 5), 1, borderType=cv2.BORDER_CONSTANT)
    return matrix

def median_blur_row_only(matrix, ksize):
    for idx in range(matrix.shape[0]):
        row_x = matrix[idx].copy()
        for idy in range(ksize//2, matrix.shape[1] - ksize//2):
            left  = idy - ksize // 2
            right = idy + ksize // 2
            count = matrix[idx][left:right].sum()
            if (count > ksize // 2):
                row_x[idy] = 1
            else:
                row_x[idy] = 0
        matrix[idx] = row_x
    return matrix

def expand_mask_by_row(matrix, ratio = 0.1):
    wide = 0
    l = 0
    r = 0
    for idx in range(matrix.shape[0]):
        row_x = matrix[idx].copy()
        for idy in range(matrix.shape[1]):
            if (matrix[idx][idy] == 0):
                if wide == 0: continue
                expand = int(wide * ratio)
                l = l - expand
                r = r + expand
                row_x[l:r] = 1
                wide = 0
            else:
                if wide == 0:
                    l = idy
                    wide = 1
                else:
                    r = idy
                    wide += 1
        matrix[idx] = row_x
    return matrix

def get_wide_of_soil(roi = [[]], wide_threshhold = 3, height_threshhold = 0.003, iou_ratio = 0.7, disparity_threshhold = 0.01):
    expand_ratio = 1 - iou_ratio
    wide = 0
    max_height = 0
    min_height = 1
    l = 0
    r = 0
    res = np.asarray([])
    for idx in range(1, roi.shape[0] - 1):
        for idy in range(roi.shape[1]):
            if (roi[idx][idy] == 0):
                if (wide < wide_threshhold or max_height - min_height < height_threshhold):
                    wide = 0
                    max_height = 0
                    min_height = 1
                    continue
                elif abs(roi[idx][l] - roi[idx][r]) > disparity_threshhold:
                    wide = 0
                    max_height = 0
                    min_height = 1
                    continue
                else:
                    l = max(int(l - wide * expand_ratio), 0)
                    r = min(int(r + wide * expand_ratio), roi.shape[1])
                    prev_max = roi[idx - 1][l:r].max()
                    next_max = roi[idx + 1][l:r].max()
                    if (max_height > prev_max and max_height > next_max):
                        res = np.append(res, wide)
                    wide = 0
                    max_height = 0
                    min_height = 1
            else:
                if (wide == 0):
                    l = idy
                    wide = 1
                else:
                    r = idy
                    wide += 1
                max_height = max(max_height, roi[idx][idy])
                min_height = min(min_height, roi[idx][idy])
    return res

def generate_depth_image(model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, _, files in os.walk(model_path):
        cur_path = os.path.join(save_path, root[len(model_path):])
        if not os.path.exists(cur_path):
          os.makedirs(cur_path)
        files.sort()
        # files = ['Class1_1.pcd', 'Class1_2.pcd', 'Class4_1.pcd', 'Class4_2.pcd', 'Class6_1.pcd', 'Class6_2.pcd', 'Class8_1.pcd', 'Class8_2.pcd']
        # files = ['Class1_2.pcd', 'Class4_2.pcd', 'Class6_2.pcd', 'Class8_2.pcd']
        # files = ['Class1_1.pcd', 'Class8_1.pcd']
        for f in files:
            file_path = os.path.join(root, f)
            fpath  = os.path.join(cur_path, f[:f.find('.')])
            np_path  = os.path.join(cur_path, f[:f.find('.')] + '.npy')
            img_path  = os.path.join(cur_path, f[:f.find('.')] + '.png')
            origin_img_path = os.path.join(cur_path, f[:f.find('.')] + '_origin.png')
            print(f)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            matrix = create_depth_image(points)
            bins = 5
            n_bins = int(matrix.shape[0]/bins)
            avg_side_cut = np.zeros((n_bins + 1, matrix.shape[1]))
            for t in range(matrix.shape[0]):
                avg_side_cut[int(t/bins)] += matrix[t]
            avg_side_cut /= bins
            roi = find_roi(avg_side_cut)
            # roi = expand_mask_by_row(roi)
            roi = median_blur_row_only(roi, 3)
            roi *= avg_side_cut
            cnt = get_wide_of_soil(roi)
            # plt.clf()
            # plt.plot(roi[5])
            plt.plot(avg_side_cut[5])
            plt.show()
            # plt.clf()
            # plt.hist(cnt, range=(0, 35), bins = 35, density=True)
            # plt.savefig(img_path)
            # x *= 1000
            # plt.clf()
            # angle_list, dist_list = count_angle_orientation_and_distance(np.asarray([x[9]]), 0, 1)
            # plt.figure()
            # plt.subplot(211)
            # plt.show()
            # plt.subplot(212)
            # plt.hist(dist_list, bins=7, density=True)
            # plt.axis([0, 35, 0, 1])
            # plt.savefig(fpath + '.png')
            # plt.show()
            # break
            # # # filtered_img = matrix - blur
            # # filtered_img = np.copy(matrix)
            # plt.figure()
            # plt.subplot(111)
            # sns.lineplot(data = x[9], label = f)
            # plt.subplot(212)
            # sns.lineplot(data = x[9], label = f)
            # count_angle_orientation_and_distance(np.asarray([x[9]]), 0, 1)
            # plt.imshow(sobel)
            # plt.hlines(105, 0, 300)
            # plt.show()
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

generate_depth_image('./PointsPoisson/PointCloud/Train', './Data/CutSide/Train')
plt.show()
