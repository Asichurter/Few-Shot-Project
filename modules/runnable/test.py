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
from modules.model.RelationNet import EmbeddingNet, TestModel
import inspect
from modules.utils.gpu_mem_track import MemTracker

a = t.Tensor([0,1,2])
b = t.Tensor([0,1,3])
c = (a==b).sum().item()/a.size(0)











