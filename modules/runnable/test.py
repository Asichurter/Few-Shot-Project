import numpy as np
import torch as t
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy.stats import entropy
import PIL.Image as Image
import torch.nn as nn
import math as m
from sklearn.preprocessing import label_binarize
import random as rd
from modules.model.RelationNet import EmbeddingNet
import inspect
from modules.utils.gpu_mem_track import MemTracker
from sklearn.neighbors import KNeighborsClassifier as KNN
import torch.nn.functional as F


# train_datas = np.array([[0,0],[3,3]])
# train_labels = np.array([2,5])
# test_datas = np.array([[2,2]])
#
# knn = KNN(n_neighbors=1)
# knn.fit(train_datas, train_labels)

data = t.Tensor([[0.,0.],[1.,1.],[3.,3.],[4.,4.]])
test = t.Tensor([[0.5,0.5],[2.5,2.5]])
label = t.Tensor([[0,1,0,0],[0,0,1,0]])

new_test = t.Tensor([])
for tensor in t.unbind(test):
    new_one = tensor.view(-1,2).repeat(4,1)
    new_test = t.cat((new_test, new_one), dim=0)

test = new_test
data = data.repeat((2,1))

inputs = (data-test).view(-1,2)

x1 = t.ones((2,4), requires_grad=True)
x2 = t.ones((4,1), requires_grad=True)

result = t.mm(t.mm(inputs,x1),x2)#.view(-1,4)
label = label.view(-1)

loss = t.nn.MSELoss()
loss_val = loss(result, label)

loss_val.backward()

query = t.Tensor([[[1,2],[1,2]],[[3,4],[3,4]]])
support = t.Tensor([[[1,1],[2,2]],[[1,1],[2,2]]])
out = F.log_softmax(t.sum((query-support)**2, dim=2),dim=1)
labels = t.LongTensor([0,1])

loss = nn.NLLLoss()
loss_val = loss(out, labels)




















