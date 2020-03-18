import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def custom_draw_geometry_with_key_callback(pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "./renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def save_view_point(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('./pinhole.json', param)

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("T")] = save_view_point
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "./camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    if not os.path.exists("./image/"):
        os.makedirs("./image/")
    if not os.path.exists("./depth/"):
        os.makedirs("./depth/")

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
            print("Capture image {:05d}".format(glb.index))
            # depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(True)
            # plt.imsave("./depth/{:05d}.png".format(glb.index),\
            #         np.asarray(depth), dpi = 1)
            img = np.asarray(image)
            plt.imsave("./image/{:05d}.jpg".format(glb.index),\
                    img, dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json("./renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


mesh = o3d.io.read_triangle_mesh('./Train/Class1_1.ply')
mesh.compute_vertex_normals()
# custom_draw_geometry_with_key_callback(mesh)
custom_draw_geometry_with_camera_trajectory(mesh)
opt.background_color = np.asarray([0, 0, 0])