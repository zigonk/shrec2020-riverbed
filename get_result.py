import os
import numpy as np

classify_path = './Circle_TopDown/Test'

result = np.zeros(241, dtype=int)
for class_num in os.listdir(classify_path):
  for f in os.listdir(os.path.join(classify_path, class_num)):
    model_num = int((f.split('.')[0]).split('_')[1])
    result[model_num] = class_num
out = open('./result.txt', 'w')
for i in range(1, 241):
  out.write(str(result[i]) + '\n')