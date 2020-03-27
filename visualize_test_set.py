import open3d as o3d
import os
import cv2
import numpy as np

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[-2, -2, -2])

def custom_draw_geometry_load_option(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.add_geometry(mesh_frame)
    vis.get_render_option().load_from_json("./renderoption.json")
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    vis.run()
    depth = vis.capture_depth_float_buffer(True)
    depth = np.asarray(depth)
    cv2.imwrite("./test.png", depth)
    vis.destroy_window()

for f in range(1, 2):
  model_path = './Train/Class3_2.ply'
  print(model_path)
  mesh = o3d.io.read_triangle_mesh(model_path)
  mesh.compute_vertex_normals()
  print(mesh)
  smooth_mesh = mesh.filter_smooth_taubin(1, 0.5)
  smooth_mesh.compute_vertex_normals()
  custom_draw_geometry_load_option(smooth_mesh)