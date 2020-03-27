import trimesh
import open3d as o3d
import numpy as np
import cv2
import colorsys

load_path ="./Test/model_1.ply"

pcd = o3d.io.read_triangle_mesh(load_path)
pcd.compute_vertex_normals()
# pcd.estimate_normals()

# # estimate radius for rolling ball
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 1.5 * avg_dist   

# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#            pcd,
#            o3d.utility.DoubleVector([radius, radius * 2]))

mesh = trimesh.Trimesh(np.asarray(pcd.vertices), np.asarray(pcd.triangles),
                          vertex_normals=np.asarray(pcd.vertex_normals))
curv = trimesh.curvature.vertex_defects(mesh)

max_red = curv.max()
max_blue = abs(curv.min())

vertex_colors = np.zeros((curv.shape[0], 3))
print(np.median(vertex_colors))

for idx, val in enumerate(curv):
  hls_color = np.asarray([0, 0, 0])
  if val < 0:
    hls_color = np.asarray([0.4, 60 - ((abs(val) / max_blue) * 50), 100])
  else:
    hls_color = np.asarray([0, 60 - ((val / max_red) * 50), 100])
  hls_color = hls_color / [1, 100, 100]
  # print(hls_color)
  rgb_color = colorsys.hls_to_rgb(hls_color[0], hls_color[1], hls_color[2])
  # print(rgb_color)
  vertex_colors[idx] = rgb_color

pcd.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

o3d.visualization.draw_geometries([pcd])
