# 本实验验证在RN上预训练的模型的嵌入层部分的输出在L1距离下的类加权距离能否较好的分类

import torch as t
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.nn import MSELoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from modules.utils.dlUtils import RN_baseline_KNN
import torch.nn.functional as F
import PIL.Image as Image

from modules.model.RelationNet import RN, EmbeddingNet
from modules.utils.dlUtils import RN_weights_init, RN_labelize, net_init
from modules.model.datasets import FewShotRNDataset, get_RN_sampler, get_RN_modified_sampler

PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/validate/"
TRAIN_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/train/"
TEST_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/validate/"
MODEL_SAVE_PATH = "D:/peimages/New/RN_5shot_5way_exp/best_acc_model_5shot_5way_v8.0.h5"
MODEL_SAVE_PATH_MY = "D:/peimages/New/RN_5shot_5way_exp/embed_5shot_5way_v8.0.h5"
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
train_classes = 30
test_classes = 30

TRAIN_CLASSES = [i for i in range(train_classes)]
TEST_CLASSES = [i for i in range(test_classes)]

TEST_CLASSES = [i for i in range(test_classes)]

weighs = {'conv':['Embed.layer1.0.weight','Embed.layer2.0.weight','Embed.layer3.0.weight','Embed.layer4.0.weight',
                  'Relation.layer1.0.weight','Relation.layer2.0.weight'],
          'linear':['Relation.fc1.weight','Relation.fc2.weight']}

rn = RN(input_size,hidder_size,k,n,qk)
#
# a = t.randn((25,1,256,256))
# b = t.randn((75,1,256,256))
#
# c = rn(a,b)

rn.load_state_dict(t.load(MODEL_SAVE_PATH))
rn = rn.cuda()

embed = EmbeddingNet()
embed.load_state_dict(rn.Embed.state_dict())
embed = embed.cuda()

t.save(embed.state_dict(), MODEL_SAVE_PATH_MY)

TEST_EPISODE = 10

entro = nn.CrossEntropyLoss().cuda()

with no_grad():
    acc_total = 0.
    loss_total = 0.
    for i in range(TEST_EPISODE):
        print(i)
        sample_classes = rd.sample(TRAIN_CLASSES, n)
        train_dataset = FewShotRNDataset(TRAIN_PATH, N)
        # sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)
        sample_sampler, query_sampler = get_RN_modified_sampler(sample_classes, k, qk, N)

        train_sample_dataloader = DataLoader(train_dataset, batch_size=n * k, sampler=sample_sampler)
        train_query_dataloader = DataLoader(train_dataset, batch_size=qk * n, sampler=query_sampler)

        samples, sample_labels = train_sample_dataloader.__iter__().next()
        queries, query_labels = train_query_dataloader.__iter__().next()

        # tracker.track()
        samples = samples.cuda()
        sample_labels = sample_labels.cuda()
        queries = queries.cuda()
        query_labels = query_labels.cuda()



        # 每一轮开始的时候先抽取n个实验类
        support_classes = rd.sample(TEST_CLASSES, n)
        # support_classes = [0,1,2,3,4]
        # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
        support_sampler, test_sampler = get_RN_modified_sampler(support_classes, k, qk, N)
        # print(list(support_sampler.__iter__()))
        test_dataset = FewShotRNDataset(TEST_PATH, N)

        test_support_dataloader = DataLoader(test_dataset, batch_size=n * k,
                                             sampler=support_sampler)
        test_test_dataloader = DataLoader(test_dataset, batch_size=qk * n,
                                          sampler=test_sampler)

        supports, support_labels = test_support_dataloader.__iter__().next()
        tests, test_labels = test_test_dataloader.__iter__().next()

        supports = supports.cuda()
        support_labels = support_labels.cuda()
        tests = tests.cuda()
        test_labels = test_labels.cuda()

        labels = RN_labelize(support_labels, test_labels, k, type="long", expand=False)
        support_embed = embed(supports)
        test_embed = embed(tests)

        test_size = test_embed.size(0)

        support_embed = support_embed.view(n,k,-1).repeat(test_size,1,1,1)
        test_embed = test_embed.view(test_size,-1).repeat(n*k,1,1).transpose(0,1).view(test_size,n,k,-1)

        out = F.softmax(t.abs(support_embed-test_embed).sum(dim=2).sum(dim=2).neg(),dim=1)
        loss = entro(out, labels).item()

        acc = (t.argmax(out, dim=1)==labels).sum().item()/labels.size(0)

        acc_total += acc
        loss_total += loss
    print("average acc: ", acc_total/TEST_EPISODE)
    print("average loss: ", loss_total/TEST_EPISODE)
