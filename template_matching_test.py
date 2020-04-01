import cv2
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import open3d as o3d

from preprocess_data import *

def generate_template_matching_image(model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, _, files in os.walk(model_path):
        cur_path = os.path.join(save_path, root[root.find('/') + 1:])
        if not os.path.exists(cur_path):
          os.makedirs(cur_path)
        for f in files:
            file_path = os.path.join(root, f)
            np_path  = os.path.join(cur_path, f[:f.find('.')] + '.npy')
            img_path  = os.path.join(cur_path, f[:f.find('.')] + '.png')
            origin_img_path = os.path.join(cur_path, f[:f.find('.')] + '_origin.png')
            print(file_path)
            # pcd = o3d.io.read_point_cloud(file_path)
            # points = np.asarray(pcd.points)
            # points = points + abs(points.min(axis = 0))
            # points *= [2000, 2000, 1]

            # size = points.max(axis = 0)[:-1].astype(int) + 1

            # matrix = np.zeros((size[1], size[0]))
            # cnt = np.zeros((size[1], size[0]))

            # for p in points:
            #     x = int(p[0])
            #     y = int(p[1])
            #     z = p[2]
            #     matrix[y][x] = max(matrix[y][x], z)
            # matrix = matrix.astype(np.float32)
            # matrix = cv2.medianBlur(matrix, 5)
            # matrix = straightening_img(matrix)
            img = np.load(file_path)
            img = img.astype(np.float32)
            img -= img.mean()
            img = img.astype(np.float32)
            # img -= img.mean()
            template = gkern(7,1)
            # template *= -1
            # template -= template.min()
            template -= template.mean()

            template = template.astype(np.float32)
            # print(template)

            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            x = res.flatten()
            # res += 1
            res *= 255 / res.max()
            # res += max(abs(res.min()), res.max())
            res = res.astype('uint8')

            cv2.imwrite(img_path, res)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


generate_template_matching_image('./PointsPoisson/', './Data/TemplateMatching')