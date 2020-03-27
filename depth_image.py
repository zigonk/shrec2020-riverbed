import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2

from find_pca import *

model_path = './Test.ply'

mesh = o3d.io.read_point_cloud(model_path)
# mesh = o3d.io.read_triangle_mesh(model_path)
# mesh = mesh.sample_points_poisson_disk(500000)
# o3d.io.write_point_cloud('./Test.ply', mesh)
# mesh = projection_model(mesh)
mesh = projection_model_point_cloud(mesh)

# points = np.asarray(mesh.vertices)
points = np.asarray(mesh.points)
points = points + abs(points.min(axis = 0))
points *= [1000, 1000, 1]

size = points.max(axis = 0)[:-1].astype(int) + 1

matrix = np.zeros((size[1], size[0]))
cnt = np.zeros((size[1], size[0]))

for p in points:
    x = int(p[0])
    y = int(p[1])
    z = p[2]
    matrix[y][x] += z
    cnt[y][x] += 1

cnt = np.maximum(cnt, 1)

matrix /= cnt
ratio = 255 / matrix.max()
matrix *= ratio
matrix = matrix.astype('float32')
blur = cv2.medianBlur(matrix, 5)
blur = cv2.GaussianBlur(blur ,(51, 51), 1, borderType=cv2.BORDER_CONSTANT)
# print(matrix.min())
lp_filter = matrix - blur
# print(lp_filter.min())
# print(lp_filter.max())
# lp_filter[lp_filter > 0] = 0
# lp_filter[lp_filter < 0] = abs(lp_filter[lp_filter < 0])
lp_filter += max(abs(lp_filter.min()), lp_filter.max())
# print(matrix.shape)
# matrix = np.maximum(matrix, blur)
cv2.imwrite('blur.png', blur)
cv2.imwrite('origin.png', matrix)
cv2.imwrite('lpfilter.png', lp_filter)

