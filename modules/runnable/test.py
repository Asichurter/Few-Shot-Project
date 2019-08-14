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

# train_datas = np.array([[0,0],[3,3]])
# train_labels = np.array([2,5])
# test_datas = np.array([[2,2]])
#
# knn = KNN(n_neighbors=1)
# knn.fit(train_datas, train_labels)

a = t.Tensor([[[1],[2]],[[2],[4]],[[3],[5]]])
a = a.sum(dim=1)

aa = [1,23,4]












