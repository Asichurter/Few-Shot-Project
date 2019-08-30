# 本实验用于测试Residual-Proto的向量嵌入后的降维分析

import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import NLLLoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import time

from modules.model.ResidualNet import ResidualNet
from modules.utils.dlUtils import RN_weights_init, net_init, RN_labelize
from modules.model.datasets import FewShotRNDataset, get_RN_modified_sampler, get_RN_sampler

VALIDATE_PATH = "D:/peimages/New/Residual_5shot_5way_exp/test/"
# VALIDATE_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/validate/"
# MODEL_LOAD_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/"+"Residual_last_epoch_model_5shot_5way_v9.0.h5"
MODEL_LOAD_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"+"Residual_best_acc_model_5shot_5way_v19.0.h5"

input_size = 256

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20
# 学习率
lr = 1e-3

seed = time.time()%100000

test_classes = 30
TEST_CLASSES = [i for i in range(test_classes)]
VALIDATE_EPISODE = 1

def validate(model, loss, classes, seed=0):
    model.eval()
    # print("test stage at %d episode" % episode)
    with no_grad():
        test_acc = 0.
        test_loss = 0.
        for j in range(VALIDATE_EPISODE):
            #print("test %d" % j)
            support_classes = classes
            # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
            support_sampler, test_sampler = get_RN_sampler(support_classes, k, qk, N, seed)
            # support_sampler, test_sampler = get_RN_modified_sampler(support_classes, k, qk, N)

            test_support_dataloader = DataLoader(dataset, batch_size=n * k,
                                                 sampler=support_sampler)
            test_test_dataloader = DataLoader(dataset, batch_size=qk * n,
                                              sampler=test_sampler)

            supports, support_labels = test_support_dataloader.__iter__().next()
            tests, test_labels = test_test_dataloader.__iter__().next()

            supports = supports.cuda()
            support_labels = support_labels.cuda()
            tests = tests.cuda()
            test_labels = test_labels.cuda()

            # test_labels = RN_labelize(support_labels, test_labels, k, n, type="float", expand=True)
            test_labels = RN_labelize(support_labels, test_labels, k, n, type="long", expand=False)
            test_relations = net(supports, tests)

            test_loss += loss(test_relations, test_labels).item()
            test_acc += (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)
            # test_acc += (t.argmax(test_relations, dim=1) == t.argmax(test_labels,dim=1)).sum().item() / test_labels.size(0)

        return test_acc/VALIDATE_EPISODE,test_loss/VALIDATE_EPISODE


dataset = FewShotRNDataset(VALIDATE_PATH, N)

net = ResidualNet(input_size=input_size,n=n,k=k,qk=qk,metric='Proto',block_num=6)
net.load_state_dict(t.load(MODEL_LOAD_PATH))
net = net.cuda()

# sample_classes = [48, 49, 50, 51, 52]
sample_classes = rd.sample(TEST_CLASSES, n)
sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N, seed=seed)

train_sample_dataloader = DataLoader(dataset, batch_size=n*k, sampler=sample_sampler)
train_query_dataloader = DataLoader(dataset, batch_size=qk*n, sampler=query_sampler)

samples,sample_labels = train_sample_dataloader.__iter__().next()
queries,query_labels = train_query_dataloader.__iter__().next()

samples = samples.cuda()
sample_labels = sample_labels.cuda()
queries = queries.cuda()
query_labels = query_labels.cuda()

sample_labels = sample_labels.cpu().numpy()[::k]
query_labels = query_labels.cpu().numpy()

samples_trans,queries_trans,center_trans = net.proto_embed_reduction(samples,queries, metric="tSNE")

acc,_loss = validate(net, nn.NLLLoss().cuda(), sample_classes, seed)

colors = ["red","blue","orange","green","purple"]
plt.title("Variance not in loss\nAcc = %.4f"%acc)
plt.axis("off")
for i in range(n):
    plt.scatter([center_trans[i][0]], center_trans[i][1], marker="x", color=colors[i])
    plt.scatter([x[0] for x in samples_trans[i*k:(i+1)*k]],[x[1] for x in samples_trans[i*k:(i+1)*k]],
             marker="o", color=colors[i])
    plt.scatter([x[0] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
             [x[1] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
             marker="^", color=colors[i])
plt.show()
