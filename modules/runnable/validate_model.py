import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from torch.optim.lr_scheduler import StepLR
from torchstat import stat
import inspect
from modules.utils.gpu_mem_track import MemTracker
from modules.utils.dlUtils import RN_baseline_KNN
import torch.nn.functional as F

from modules.model.RelationNet import RN
from modules.utils.dlUtils import RN_weights_init, RN_labelize, net_init
from modules.model.datasets import FewShotRNDataset, get_RN_sampler, get_RN_modified_sampler

PATH = "D:/peimages/New/RN_5shot_5way_exp/train/"
MODEL_SAVE_PATH = "D:/peimages/New/RN_5shot_5way_exp/"
DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_relation_net_exp/"

input_size = 256
hidder_size = 8

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20

# 训练和测试中类的总数
test_classes = 60

TEST_CLASSES = [i for i in range(test_classes)]

rn = RN(input_size, hidder_size, k=k, n=n, qk=qk)
rn.load_state_dict(t.load(MODEL_SAVE_PATH+"best_acc_model_%dshot_%dway.h5",map_location='gpu'))

mse = MSELoss().cuda()
rn.eval()
with no_grad:
    # 每一轮开始的时候先抽取n个实验类
    sample_classes = rd.sample(TEST_CLASSES, n)
    train_dataset = FewShotRNDataset(PATH, N)
    # sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)
    sample_sampler,query_sampler = get_RN_modified_sampler(sample_classes, k, qk, N)

    train_sample_dataloader = DataLoader(train_dataset, batch_size=n*k, sampler=sample_sampler)
    train_query_dataloader = DataLoader(train_dataset, batch_size=qk*n, sampler=query_sampler)


