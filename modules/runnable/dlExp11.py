# 本实验用于测试Residual网络

import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import NLLLoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F

from modules.model.ResidualNet import ResidualNet
from modules.utils.dlUtils import RN_weights_init, net_init, RN_labelize
from modules.model.datasets import FewShotRNDataset, get_RN_modified_sampler, get_RN_sampler

import time

TRAIN_PATH = "D:/peimages/New/Residual_5shot_5way_exp/train/"
VALIDATE_PATH = "D:/peimages/New/Residual_5shot_5way_exp/validate/"
# MODEL_SAVE_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/"
# MODEL_LOAD_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/"+"Residual_5000_epoch_model_5shot_5way_v13.0.h5"
MODEL_SAVE_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"
DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_ResidualNet_5shot_5way_exp/"

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

version = 18
metric = "Proto"

TEST_CYCLE = 100
MAX_ITER = 20000
TEST_EPISODE = 50

# 训练和测试中类的总数
total_train_classes = 300
train_classes = 300
test_classes = 60

inner_var_alpha = 1e-2
outer_var_alpha = 1e-2

# init_best_acc = 0.8

TRAIN_CLASSES = rd.sample([i for i in range(total_train_classes)],train_classes)
TEST_CLASSES = [i for i in range(test_classes)]

net = ResidualNet(input_size=input_size,n=n,k=k,qk=qk,metric=metric, block_num=6)
# net = ResidualNet(input_size=input_size,n=n,k=k,qk=qk,metric=metric,block_num=6)
# net.load_state_dict(t.load(MODEL_LOAD_PATH))
net = net.cuda()

# net.Embed.apply(RN_weights_init)
net.apply(RN_weights_init)

# net.apply(net_init)

opt = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
# opt = SGD(net.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
scheduler = StepLR(opt, step_size=1000, gamma=0.5)
# scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100, verbose=True, min_lr=1e-6)
entro = nn.NLLLoss().cuda()
# entro = nn.MSELoss().cuda()
# entro = nn.CrossEntropyLoss().cuda()

train_acc_his = []
train_loss_his = []
test_acc_his = []
test_loss_his = []

# best_acc = init_best_acc
best_acc = 0
best_epoch = -1
print(net)
for episode in range(MAX_ITER):

    if episode%5000 == 0 and episode != 0:
        choice = input("%d episodes have finished, continue?"%(episode-1))
        if choice.find("n") != -1 or choice.find("no")!= -1:
            break

    net.train()
    net.zero_grad()
    print("%d th episode"%episode)

    # 每一轮开始的时候先抽取n个实验类
    sample_classes = rd.sample(TRAIN_CLASSES, n)
    train_dataset = FewShotRNDataset(TRAIN_PATH, N)
    sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)
    # sample_sampler,query_sampler = get_RN_modified_sampler(sample_classes, k, qk, N)

    train_sample_dataloader = DataLoader(train_dataset, batch_size=n*k, sampler=sample_sampler)
    train_query_dataloader = DataLoader(train_dataset, batch_size=qk*n, sampler=query_sampler)

    samples,sample_labels = train_sample_dataloader.__iter__().next()
    queries,query_labels = train_query_dataloader.__iter__().next()

    samples = samples.cuda()
    sample_labels = sample_labels.cuda()
    queries = queries.cuda()
    query_labels = query_labels.cuda()

    labels = RN_labelize(sample_labels, query_labels, k, n, type="long", expand=False)
    # labels = RN_labelize(sample_labels, query_labels, k, n, type="float", expand=True)

    outs = net(samples, queries)

    loss = entro(outs, labels) + inner_var_alpha*net.forward_inner_var - outer_var_alpha*net.forward_outer_var
    # loss = entro(outs, labels)

    loss.backward()

    # 使用了梯度剪裁
    t.nn.utils.clip_grad_norm_(net.parameters(), 0.5)

    opt.step()

    acc = (t.argmax(outs, dim=1)==labels).sum().item()/labels.size(0)
    # acc = (t.argmax(outs, dim=1)==t.argmax(labels,dim=1)).sum().item()/labels.size(0)
    loss_val = loss.item()

    print("train acc: ", acc)
    print("train loss: ", loss_val)

    train_acc_his.append(acc)
    train_loss_his.append(loss_val)

    # scheduler.step()
    if (episode + 1) % 5000 == 0:
        print("save!")
        t.save(net.state_dict(),
               MODEL_SAVE_PATH + "Residual_%d_epoch_model_%dshot_%dway_v%d.0.h5" % (episode + 1, k, n, version))

    if episode % TEST_CYCLE == 0:
        # input("----- Time to test -----")
        net.eval()
        print("test stage at %d episode"%episode)
        with no_grad():
            test_acc = 0.
            test_loss = 0.
            for j in range(TEST_EPISODE):
                print("test %d"%j)
                # 每一轮开始的时候先抽取n个实验类
                support_classes = rd.sample(TEST_CLASSES, n)
                # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
                # support_sampler, test_sampler = get_RN_modified_sampler(support_classes, k, qk, N)
                support_sampler, test_sampler = get_RN_sampler(support_classes, k, qk, N)
                # print(list(support_sampler.__iter__()))
                test_dataset = FewShotRNDataset(VALIDATE_PATH, N)

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

                # test_labels = RN_labelize(support_labels, test_labels, k, n, type="float", expand=True)
                test_labels = RN_labelize(support_labels, test_labels, k, n, type="long", expand=False)
                test_relations = net(supports, tests)

                test_loss += (entro(test_relations, test_labels) + inner_var_alpha*net.forward_inner_var - outer_var_alpha*net.forward_outer_var).item()
                # test_loss += entro(test_relations, test_labels).item()
                test_acc += (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)
                # test_acc += (t.argmax(test_relations, dim=1)==t.argmax(test_labels,dim=1)).sum().item()/test_labels.size(0)

            test_acc_his.append(test_acc/TEST_EPISODE)
            test_loss_his.append(test_loss/TEST_EPISODE)

            print("****************************************")
            print("val acc: ", test_acc/TEST_EPISODE)
            print("val loss: ", test_loss/TEST_EPISODE)
            if test_acc/TEST_EPISODE > best_acc:
                t.save(net.state_dict(),
                       MODEL_SAVE_PATH + "Residual_best_acc_model_%dshot_%dway_v%d.0.h5" % (k, n, version))
                print("model save at %d episode" % episode)
                best_acc = test_acc/TEST_EPISODE
                best_epoch = episode
            print("best val acc: ", best_acc)
            print("best epoch: %d"%best_epoch)
            print("****************************************")
            # input("----- Test Complete ! -----")

# 根据历史值画出准确率和损失值曲线
train_x = [i for i in range(0,len(train_acc_his),TEST_CYCLE)]
test_x = [i for i in range(0,len(train_acc_his), TEST_CYCLE)]

plt.title('%d-shot %d-way Residual-%s Net Accuracy'%(k,n,metric))
plt.plot(train_x, [train_acc_his[i] for i in range(0,len(train_acc_his),TEST_CYCLE)], linestyle='-', color='blue', label='train')
plt.plot(test_x, test_acc_his, linestyle='-', color='red', label='validate')
plt.plot(train_x, [1/k]*len(train_x), linestyle='--', color="black", label="baseline")
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_acc.png'%version)
plt.show()

plt.title('%d-shot %d-way Residual-%s Net Loss'%(k,n,metric))
plt.ylim(0,3)
plt.plot(train_x, [train_loss_his[i] for i in range(0,len(train_acc_his),TEST_CYCLE)], linestyle='-', color='blue', label='train')
plt.plot(test_x, test_loss_his, linestyle='-', color='red', label='validate')
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_loss.png'%version)
plt.show()

np.save(DOC_SAVE_PATH+"%d_acc.npy"%version, np.array(test_acc_his))
np.save(DOC_SAVE_PATH+"%d_loss.npy"%version, np.array(test_loss_his))

