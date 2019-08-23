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

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# frame = inspect.currentframe()
# tracker = MemTracker(frame)

TRAIN_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/train/"
TEST_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/validate/"
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
# 学习率
lr = 1e-3

version = 12

TEST_CYCLE = 50
MAX_ITER = 10000
TEST_EPISODE = 20

# 训练和测试中类的总数
train_classes = 60
test_classes = 30

TRAIN_CLASSES = [i for i in range(train_classes)]
TEST_CLASSES = [i for i in range(test_classes)]

# tracker.track()
rn = RN(input_size, hidder_size, k=k, n=n, qk=qk)
# print(get_parameter_number(rn))
rn = rn.cuda()
# tracker.track()

#print(list(rn.Relation.fc1.named_parameters()))

rn.Embed.apply(RN_weights_init)
rn.Relation.apply(RN_weights_init)
#
# rn.Embed.apply(net_init)
# rn.Relation.apply(net_init)

# embed_opt = Adam(rn.Embed.parameters(), lr=lr, weight_decay=1e-4)
# relation_opt = Adam(rn.Relation.parameters(), lr=lr, weight_decay=1e-4)
opt = Adam(rn.parameters(), lr=lr, weight_decay=1e-4)
# opt = t.optim.SGD(rn.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(opt, step_size=1000, gamma=0.5)
# relation_scheduler = StepLR(relation_opt, step_size=1000, gamma=0.5)
mse = MSELoss().cuda()
# mse = t.nn.NLLLoss().cuda()

train_acc_his = []
train_loss_his = []
test_acc_his = []
test_loss_his = []

best_acc = 0.
for episode in range(MAX_ITER):

    rn.train()
    rn.zero_grad()
    print("%d th episode"%episode)

    # 每一轮开始的时候先抽取n个实验类
    sample_classes = rd.sample(TRAIN_CLASSES, n)
    train_dataset = FewShotRNDataset(TRAIN_PATH, N)
    # sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)
    sample_sampler,query_sampler = get_RN_modified_sampler(sample_classes, k, qk, N)

    train_sample_dataloader = DataLoader(train_dataset, batch_size=n*k, sampler=sample_sampler)
    train_query_dataloader = DataLoader(train_dataset, batch_size=qk*n, sampler=query_sampler)

    samples,sample_labels = train_sample_dataloader.__iter__().next()
    queries,query_labels = train_query_dataloader.__iter__().next()

    # tracker.track()
    samples = samples.cuda()
    sample_labels = sample_labels.cuda()
    queries = queries.cuda()
    query_labels = query_labels.cuda()
    # tracker.track()

    labels = RN_labelize(sample_labels, query_labels, k)

    relations = rn(samples, queries).view(-1,n)

    # relations = F.softmax(relations, dim=1)

    loss = mse(relations, labels)

    loss.backward()

    # 使用了梯度剪裁
    t.nn.utils.clip_grad_norm_(rn.Embed.parameters(), 0.5)
    t.nn.utils.clip_grad_norm_(rn.Relation.parameters(), 0.5)

    # embed_opt.step()
    # relation_opt.step()
    opt.step()

    # print("out:", relations.tolist())
    # print("label:", labels.tolist())

    acc = (t.argmax(relations, dim=1)==t.argmax(labels, dim=1)).sum().item()/labels.size(0)
    loss_val = loss.item()

    print("train acc: ", acc)
    print("train loss: ", loss_val)

    train_acc_his.append(acc)
    train_loss_his.append(loss_val)

    # embed_scheduler.step()
    # relation_scheduler.step()
    scheduler.step()

    if acc > best_acc:
        t.save(rn.state_dict(), MODEL_SAVE_PATH+"best_acc_model_%dshot_%dway_v%d.0.h5"%(k,n,version))
        print("model save at %d episode"%episode)
        best_acc = acc

    if episode % TEST_CYCLE == 0:
        # input("----- Time to test -----")
        rn.eval()
        print("test stage at %d episode"%episode)
        with no_grad():
            test_acc = 0.
            test_loss = 0.
            for j in range(TEST_EPISODE):
                print("%d episode: test %d"%(episode,j))
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

                test_labels = RN_labelize(support_labels, test_labels, k)
                test_relations = rn(supports, tests).view(-1, n).cuda()
                # m_support,m_query = rn(supports, tests, feature_out=True)
                # test_baseline = RN_baseline_KNN(m_support, m_query, support_labels, query_labels, k)

                # 使用softmax将关系输出转化为概率输出
                # test_relations = F.softmax(test_relations, dim=1)

                test_loss += mse(test_relations, test_labels).item()
                test_acc += (t.argmax(test_relations, dim=1)==t.argmax(test_labels, dim=1)).sum().item()/test_labels.size(0)

            test_acc_his.append(test_acc/TEST_EPISODE)
            test_loss_his.append(test_loss/TEST_EPISODE)

            print("****************************************")
            print("val acc: ", test_acc/TEST_EPISODE)
            print("val loss: ", test_loss/TEST_EPISODE)
            # print("knn baseline acc: ", test_baseline)
            print("****************************************")
            # input("----- Test Complete ! -----")

# 根据历史值画出准确率和损失值曲线
train_x = [i for i in range(0,MAX_ITER,TEST_CYCLE)]
test_x = [i for i in range(0,MAX_ITER, TEST_CYCLE)]

plt.title('%d-shot %d-way Relation Net Accuracy'%(k,n))
plt.plot(train_x, [train_acc_his[i] for i in range(0,MAX_ITER,TEST_CYCLE)], linestyle='-', color='blue', label='train')
plt.plot(test_x, test_acc_his, linestyle='-', color='red', label='validate')
plt.plot(train_x, [1/k]*len(train_x), linestyle='--', color="black", label="baseline")
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_acc.png'%version)
plt.show()

plt.title('%d-shot %d-way Relation Net Loss'%(k,n))
plt.plot(train_x, [train_loss_his[i] for i in range(0,MAX_ITER,TEST_CYCLE)], linestyle='-', color='blue', label='train')
plt.plot(test_x, test_loss_his, linestyle='-', color='red', label='validate')
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_loss.png'%version)
plt.show()

np.save(DOC_SAVE_PATH+"%d_acc.npy"%version, np.array(test_acc_his))
np.save(DOC_SAVE_PATH+"%d_loss.npy"%version, np.array(test_loss_his))

