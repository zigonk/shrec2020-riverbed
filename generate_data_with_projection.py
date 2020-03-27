import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from find_pca import *
from pynput.keyboard import Key, Controller

train_image_path = './Data/Train'
test_image_path = './Data/Test'
valid_image_path = './Data/Valid'
trainset_model3d_path = './Train'
testset_model3d_path = './Test'
groundtruth_path = './ground_truth.txt'

def custom_draw_geometry(data_path, pcd, class_name, file_name):
    class_path = os.path.join(data_path, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.background_color = (0, 0, 0)
    # render_option.mesh_show_wireframe = True
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("{}/{}.png".format(class_path, file_name))
    vis.destroy_window()

def generate_train_image():
    for f in os.listdir(trainset_model3d_path):
        pos = f.find('_')
        class_name = f[:pos]
        print(f)
        path = os.path.join(trainset_model3d_path, f)
        pos_ext = f.find('.')
        file_name = f[:pos_ext]
        mesh = o3d.io.read_triangle_mesh(path)
        mesh = projection_model(mesh)
        custom_draw_geometry(train_image_path, mesh, class_name, file_name)

def generate_image(list_path, model_path, image_path):
    for class_name in list_path:
        print('Current class name {}'.format(class_name))
        for f in list_path[class_name]:
            pos = f.find('_')
            path = os.path.join(model_path, f)
            pos_ext = f.find('.')
            file_name = f[:pos_ext]
            mesh = o3d.io.read_triangle_mesh(path)
            mesh = projection_model(mesh)
            custom_draw_geometry(image_path, mesh, class_name, file_name)

def generate_image_without_class(model_path, image_path):
    class_name = 'model'
    for f in os.listdir(model_path):
        print(f)
        path = os.path.join(model_path, f)
        pos_ext = f.find('.')
        file_name = f[:pos_ext]
        mesh = o3d.io.read_triangle_mesh(path)
        mesh = projection_model(mesh)
        custom_draw_geometry(image_path, mesh, class_name, file_name)

def generate_image_by_model(path, image_path, class_name, file_name):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    # mesh = projection_model(mesh)
    custom_draw_geometry(image_path, mesh, class_name, file_name)


# generate_image_without_class(testset_model3d_path, test_image_path)
# generate_image_by_model('./Train/Class6_1.ply', test_image_path, 'model', 'model_1')
generate_train_image()