import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def custom_draw_geometry_with_key_callback(pcd):
    def save_view_point(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('./pinhole.json', param)

    key_to_callback = {}
    key_to_callback[ord("T")] = save_view_point
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


sample_3dpcd_path = './Test/model_8.ply'

mesh = o3d.io.read_triangle_mesh(sample_3dpcd_path)
mesh.compute_vertex_normals()
custom_draw_geometry_with_key_callback(mesh)