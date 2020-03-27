import open3d as o3d
import numpy as np
import os

from find_pca import *

trainset_model3d_path = './Train'
testset_model3d_path = './Test'

train_point_path = './PointsPoisson/Train'
test_point_path = './PointsPoisson/Test'

def create_sample_poisson_point(mesh):
    points = np.asarray(mesh.vertices)
    points = points + abs(points.min(axis = 0))
    points *= [1000, 1000, 1]
    size = points.max(axis = 0)[:-1].astype(int) + 1
    num_points = size[0] * size[1] * 10
    pcd = mesh.sample_points_poisson_disk(num_points)
    return pcd

def generate_poisson_point(model3d_path, poisson_point_path):
    if not os.path.exists(poisson_point_path):
        os.makedirs(poisson_point_path)
    for f in os.listdir(model3d_path):
        path = os.path.join(model3d_path, f)
        pos_ext = f.find('.')
        file_name = f[:pos_ext]
        mesh = o3d.io.read_triangle_mesh(path)
        mesh = projection_model(mesh)
        pcd = create_sample_poisson_point(mesh)
        o3d.io.write_point_cloud(os.path.join(poisson_point_path, file_name + '.pcd'), pcd)

generate_poisson_point(trainset_model3d_path, train_point_path)
generate_poisson_point(testset_model3d_path, test_point_path)