import open3d as o3d
import copy
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cal_angle(vector_1, vector_2):
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  angle = np.arccos(dot_product)
  return angle

def projection_model(pcd):
  mesh = copy.deepcopy(pcd)
  pcd.compute_vertex_normals()
  X = np.asarray(pcd.vertices)
  normal_vec = np.asarray(pcd.vertex_normals)

  triangle_normal_vec = np.asarray(pcd.triangle_normals)
  sum_vec = np.sum(triangle_normal_vec, axis = 0)
  sum_vec = sum_vec / np.linalg.norm(sum_vec)



  pca = PCA(n_components=2)
  pca.fit(X)
  X_pca = pca.transform(X)
  X_new = pca.inverse_transform(X_pca)

  dist = np.zeros(X.shape[0])

  X_with_depth = np.zeros((X_pca.shape[0], X_pca.shape[1]+1))
  X_with_depth[:,:-1] = X_pca

  for idx in range(X.shape[0]):
    angle = cal_angle(X_new[idx] - X[idx], sum_vec)
    sign = 1 if (angle > 0.5*np.pi) else -1
    X_with_depth[idx][2] = np.linalg.norm(X[idx] - X_new[idx]) * sign

  mesh.vertices = o3d.utility.Vector3dVector(X_with_depth)
  mesh.compute_vertex_normals()
  return mesh

def projection_model_point_cloud(pcd):
  mesh = copy.deepcopy(pcd)
  # pcd.compute_vertex_normals()
  X = np.asarray(pcd.points)
  # normal_vec = np.asarray(pcd.vertex_normals)

  # triangle_normal_vec = np.asarray(pcd.triangle_normals)
  # sum_vec = np.sum(triangle_normal_vec, axis = 0)
  # sum_vec = sum_vec / np.linalg.norm(sum_vec)



  pca = PCA(n_components=2)
  pca.fit(X)
  X_pca = pca.transform(X)
  X_new = pca.inverse_transform(X_pca)

  dist = np.zeros(X.shape[0])

  X_with_depth = np.zeros((X_pca.shape[0], X_pca.shape[1]+1))
  X_with_depth[:,:-1] = X_pca

  for idx in range(X.shape[0]):
    # angle = cal_angle(X_new[idx] - X[idx], sum_vec)
    # sign = 1 if (angle > 0.5*np.pi) else -1
    X_with_depth[idx][2] = np.linalg.norm(X[idx] - X_new[idx])

  mesh.points = o3d.utility.Vector3dVector(X_with_depth)
  # mesh.compute_vertex_normals()
  return mesh