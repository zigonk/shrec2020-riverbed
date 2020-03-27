import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Controller

train_image_path = './Data/Train'
test_image_path = './Data/Test'
valid_image_path = './Data/Valid'
trainset_model3d_path = './Train'
testset_model3d_path = './Test'
groundtruth_path = './ground_truth.txt'

keyboard = Controller()

def custom_draw_geometry_with_camera_trajectory(data_path, pcd, class_name, f):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "./camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    class_path = os.path.join(data_path, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            vis.capture_screen_image("{}/{}{:02d}.png".format(class_path,f,glb.index))
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
            custom_draw_geometry_with_camera_trajectory.vis.\
                    close()
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json("./renderoption.json")
    vis.register_animation_callback(move_forward)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
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
        mesh.compute_vertex_normals()
        custom_draw_geometry_with_camera_trajectory(train_image_path,mesh, class_name, file_name)

def generate_image(list_path, model_path, image_path):
    for class_name in list_path:
        print('Current class name {}'.format(class_name))
        for f in list_path[class_name]:
            pos = f.find('_')
            path = os.path.join(model_path, f)
            pos_ext = f.find('.')
            file_name = f[:pos_ext]
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            custom_draw_geometry_with_camera_trajectory(image_path, mesh, class_name, file_name)

def create_list_from_groundtruth():
    f = open(groundtruth_path, 'r')
    list_path_valid = {}
    for idx, class_id in enumerate(f):
        class_id = class_id.strip()
        class_name = 'Class' + class_id
        file_name = 'model_' + str(idx + 1) + '.ply'
        if (class_name in list_path_valid):
            list_path_valid[class_name].append(file_name)
        else:
            list_path_valid[class_name] = []
            list_path_valid[class_name].append(file_name)
    return list_path_valid

list_path_valid = create_list_from_groundtruth()
generate_image(list_path_valid, testset_model3d_path, valid_image_path)
