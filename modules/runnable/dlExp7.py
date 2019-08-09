import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
import random as rd
from torch.utils.data import DataLoader

from modules.model.RelationNet import RN
from modules.utils.dlUtils import RN_weights_init, RN_labelize
from modules.model.datasets import FewShotRNDataset, get_RN_sampler

TRAIN_PATH = ""
TEST_PATH = ""

input_size = 256
hidder_size = 8

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 30
# 学习率
lr = 1e-3

MAX_ITER = 100

# 训练和测试中类的总数
train_classes = 10
test_classes = 10

TRAIN_CLASSES = [i for i in range(train_classes)]
TEST_CLASSES = [i for i in range(test_classes)]

rn = RN(input_size, hidder_size, k=k, n=n, qk=qk)
rn = rn.cuda()
rn.Embed.apply(RN_weights_init)
rn.Relation.apply(RN_weights_init)

embed_opt = Adam(rn.Embed.parameters(), lr=lr)
relation_opt = Adam(rn.Relation.parameters(), lr=lr)
mse = MSELoss()

best_acc = 0.
for episode in range(MAX_ITER):
    print("%d th episode"%episode)

    sample_classes = rd.sample(TRAIN_CLASSES, n)
    sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)

    train_sample_dataloader = DataLoader(FewShotRNDataset(TRAIN_PATH+"support/"), batch_size=n*k, sampler=sample_sampler)
    train_query_dataloader = DataLoader(FewShotRNDataset(TRAIN_PATH+"query/"), batch_size=qk*n, sampler=query_sampler)

    samples,sample_labels = train_sample_dataloader.__iter__().next()
    queries,query_labels = train_query_dataloader.__iter__().next()

    labels = RN_labelize(sample_labels, query_labels, k)
    relations = rn(samples, queries).view(-1,)

    loss = mse(relations, labels)

    rn.Embed.zero_grad()
    rn.Relation.zero_grad()

    loss.backward()

    t.nn.utils.clip_grad_norm_(rn.Embed.parameters(), 0.5)
    t.nn.utils.clip_grad_norm_(rn.Relation.parameters(), 0.5)

    embed_opt.step()
    relation_opt.step()