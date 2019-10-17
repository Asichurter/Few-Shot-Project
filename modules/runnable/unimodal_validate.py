import torch as t
import matplotlib.pyplot as plt
import numpy as np
import os

from modules.model.PrototypicalNet import ProtoNet

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20
CROP_SIZE = 224

model_type = 'ProtoNet'
model_mode = 'best_acc'
model_version = 11

data_folder = 'cluster'
data_type = 'test'

MODEL_PATH = "D:/peimages/New/%s/models/"%data_folder+"%s_%s_model_%dshot_%dway_v%d.0.h5"%(model_type,model_mode,k,n,model_version)
DATA_PATH = "D:/peimages/New/%s/%s.npy"%(data_folder,data_type)
