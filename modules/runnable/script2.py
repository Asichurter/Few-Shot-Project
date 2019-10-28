# 探索Malimg数据集

import numpy as np

path = 'D:/DL/Malimg-master/Malimg-master/malimg.npz'

a = np.load(path, allow_pickle=True)
label_count = dict().fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
for k in label_count.keys():
    label_count[k] = 0

for item in a['arr']:
    data,label = item
    label_count[label] += 1

print(label_count)
