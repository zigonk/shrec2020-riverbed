import numpy as np
import cv2
import scipy.stats as st
import matplotlib.pyplot as plt
import os

def find_maximum_square(origin):
  mat = origin.copy()
  img = mat.copy()
  size = mat.shape
  mat[mat > 0] = 1
  # mat = 1 - mat
  L = np.zeros((size[0], size[1]))
  R = np.zeros((size[0], size[1]))
  hist = []
  fill_mat = np.zeros((size[0], size[1]))
  stack = [size[1]]
  cnt = 1
  res = []
  for i in range(1, size[0]):
    for j in range(size[1]):
      if (mat[i, j] == 0):
        continue
      else:
        mat[i, j] = mat[i - 1][j] + 1
    stack = []
    check = False
    for j in range(size[1]):
      if (origin[i][j] == 0):
        check = True
      if (not check):
        R[i][j] = -1000
        continue
      while (len(stack) != 0 and mat[stack[-1][0], stack[-1][1]] > mat[i, j]):
        R[stack[-1][0], stack[-1][1]] = j - stack[-1][1]
        stack.pop()
      stack.append((i, j))
    stack = []
    check = False
    for j in range(size[1] - 1, -1, -1):
      if (origin[i][j] == 0):
        check = True
      if (not check):
        L[i][j] = -1000
        continue
      while (len(stack) != 0 and mat[stack[-1][0], stack[-1][1]] > mat[i, j]):
        L[stack[-1][0], stack[-1][1]] = stack[-1][1] - j
        stack.pop()
      stack.append((i, j))
    for j in range(size[1]):
      if (mat[i, j] == 0): continue
      if (L[i, j] + R[i, j] + 1 >= mat[i, j]):
        v_j = L[i, j]
        if (v_j >= mat[i, j] - 1): v_j = j
        else: v_j = j + mat[i, j] - v_j - 1
        res.append((i, v_j, mat[i, j]))
  res.sort(key = lambda tup : tup[2], reverse=True)
  for (i, j, sz) in res:
    i = int(i)
    j = int(j)
    sz = int(sz)
    if (sz < 4): continue
    if (fill_mat[i - sz + 1: i + 1, j - sz + 1: j + 1].max() == 1):
      continue
    for x in range(i - sz + 1, i + 1):
      for y in range(j - sz + 1, j + 1):
        fill_mat[x, y] = 1
    # cv2.rectangle(img,(j - sz + 1, i - sz + 1), (j + 1, i + 1), 0, 1)
    hist.append(sz)
  return hist, img


# for root, _, files in os.walk('Data/TemplateMatching/PointsPoisson/Train/'):
#   for f in files:
#     file_path = os.path.join(root, f)
#     img = cv2.imread(file_path, 0)
#     ret,img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
#     hist, img = find_maximum_square(img)
#     cv2.imwrite('./Data/MaxSquare/Vis_' + f, img)
#     # hist = np.reshape(hist, hist.max())
#     plt.hist(hist, bins=100)
#     plt.savefig('./Data/MaxSquare/' + f)
#     plt.clf()
