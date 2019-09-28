import numpy as np
import torch as t
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy.stats import entropy
import PIL.Image as Image
import torch.nn as nn
import math
from sklearn.preprocessing import label_binarize
import random as rd
# from modules.model.RelationNet import EmbeddingNet
import inspect
from sklearn.neighbors import KNeighborsClassifier as KNN
import torch.nn.functional as F
import time
import re


# train_datas = np.array([[0,0],[3,3]])
# train_labels = np.array([2,5])
# test_datas = np.array([[2,2]])
#
# knn = KNN(n_neighbors=1)
# knn.fit(train_datas, train_labels)

# data = t.Tensor([[0.,0.],[1.,1.],[3.,3.],[4.,4.]])
# test = t.Tensor([[0.5,0.5],[2.5,2.5]])
# label = t.Tensor([[0,1,0,0],[0,0,1,0]])
#
# new_test = t.Tensor([])
# for tensor in t.unbind(test):
#     new_one = tensor.view(-1,2).repeat(4,1)
#     new_test = t.cat((new_test, new_one), dim=0)
#
# test = new_test
# data = data.repeat((2,1))
#
# inputs = (data-test).view(-1,2)
#
# x1 = t.ones((2,4), requires_grad=True)
# x2 = t.ones((4,1), requires_grad=True)
#
# result = t.mm(t.mm(inputs,x1),x2)#.view(-1,4)
# label = label.view(-1)
#
# loss = t.nn.MSELoss()
# loss_val = loss(result, label)
#
# loss_val.backward()
#
# query = t.Tensor([[[1,2],[1,2]],[[3,4],[3,4]]])
# support = t.Tensor([[[1,1],[2,2]],[[1,1],[2,2]]])
# out = F.log_softmax(t.sum((query-support)**2, dim=2),dim=1)
# labels = t.LongTensor([0,1])
#
# loss = nn.CrossEntropyLoss()
#
# out = t.Tensor([[1.3,10,0.4],[0.1,0.2,0.9]])
# label=t.LongTensor([1,2])
# loss_val = loss(out, label)
#
# dat = t.Tensor([[1,1],[2,3]])
# dat2 = t.Tensor([[1,3],[4,2]])
#
# dat_length = t.sqrt((dat**2).sum(dim=1))
# dat2_length = t.sqrt((dat2**2).sum(dim=1))
# dot = t.mul(dat,dat2).sum(dim=1)


# a = t.Tensor([[[1,2],[1,1]],[[0,2],[1,3]],[[0,1],[1,2]]])
# b = a.sum(dim=2)
# c = a.sum(dim=2).sum(dim=1).unsqueeze(dim=1).repeat(1,2)

# a = t.randn((5,1,4,14))
# # [qk,n,d]
# # 距离
# dis = t.randn((15,5,14))
#
# conv1 = nn.Conv2d(1,32, kernel_size=(4,1), stride=(1,1), padding=(1,0))
# conv2 = nn.Conv2d(32,32, kernel_size=(4,1), stride=(1,1), padding=(2,0))
# conv3 = nn.Conv2d(32,1, kernel_size=(4,1), stride=(4,1), padding=(0,0))
#
# attention = conv3(conv2(conv1(a))).squeeze().repeat(15,1,1)
#
# attented = (attention*dis).sum(dim=2)

# cs = [i for i in range(1,101)]
# val = []
# f = 10
# for c in cs:
#     a = [-c]*f
#     a[0] = f
#     a = t.Tensor(a)
#     val.append(t.softmax(a, dim=0)[0].item())
#
# plt.plot(cs, val)
# plt.show()

# d = {1:1, 2:4, 3:9, 4:16}
# np.save('D:/Few-Shot-Project/data/dict.npy', d)

# d = np.load('D:/Few-Shot-Project/data/clusters_0.5eps_20minnum.npy', allow_pickle=True)
# d = d.item()
# for key,val in d.items():
#     print(key, 'cluster num: %d'%len(val))
# print(len(d))]

print(os.path.exists('D:/Few-Shot-Project/data/cluster_plot/'))

































