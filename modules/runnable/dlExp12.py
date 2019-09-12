# 本实验用于测试Residual-Relation网络的微调

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

from sklearn.manifold import MDS

from modules.model.ResidualNet import ResidualNet
from modules.model.PrototypicalNet import ProtoNet
from modules.model.RelationNet import RN
from modules.model.SiameseNet import SiameseNet
from modules.utils.dlUtils import RN_weights_init, net_init, RN_labelize
from modules.model.datasets import FewShotRNDataset, get_RN_modified_sampler, get_RN_sampler

def bar_frequency(data, title, bins=10, color="blue", bar_width=0.2, precision=2):
    bin_interval = 1/bins
    x = np.arange(bins)*bar_width
    x_label = (x-bar_width/2)
    x_label = np.append(x_label, x_label[-1]+bar_width)
    x_ticks = [round(i*bin_interval, precision) for i in range(bins+1)]
    print(x, x_label, x_ticks, sep='\n')
    data = np.floor(np.array(data)/bin_interval)
    frequency = [0]*bins
    for i in data:
        frequency[int(i)] += 1/len(data)
    plt.title(title)
    plt.bar(x, frequency, alpha=0.5, width=bar_width, color=color, edgecolor='black', label="frequency", lw=3)
    plt.xticks(x_label, x_ticks)
    plt.legend()
    plt.show()

VALIDATE_PATH = "D:/peimages/New/test/test/"

# VALIDATE_PATH = "D:/peimages/New/Residual_5shot_5way_exp/test/"
# MODEL_LOAD_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/"+"Residual_last_epoch_model_5shot_5way_v9.0.h5"
MODEL_LOAD_PATH = "D:/peimages/New/test/models/"+"ProtoNet_best_acc_model_5shot_5way_v14.0.h5"
# MODEL_LOAD_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"+"Siamese_best_acc_model_5shot_5way_v2.0.h5"
# MODEL_LOAD_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"+"ProtoNet_best_acc_model_5shot_5way_v11.0.h5"
# MODEL_LOAD_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"+"RelationNet_best_acc_model_5shot_5way_v13.0.h5"
# MODEL_LOAD_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"+"Residual_best_acc_model_5shot_5way_v27.0.h5"

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

version = 1

TEST_EPISODE = 600
VALIDATE_EPISODE = 20
FINETUNING_EPISODE = 10
if_finetuning = False

embed_size = 7
hidden_size = 8

test_classes = 30
TEST_CLASSES = [i for i in range(test_classes)]

dataset = FewShotRNDataset(VALIDATE_PATH, N, rd_crop_size=224)

acc_hist = []
loss_hist = []

def validate(model, loss, classes):
    model.eval()
    # print("test stage at %d episode" % episode)
    with no_grad():
        test_acc = 0.
        test_loss = 0.
        for j in range(VALIDATE_EPISODE):
            #print("test %d" % j)
            support_classes = classes
            # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
            support_sampler, test_sampler = get_RN_sampler(support_classes, k, qk, N)
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

            l = loss(test_relations, test_labels).item()
            a = (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)
            # a = (t.argmax(test_relations, dim=1) == t.argmax(test_labels,dim=1)).sum().item() / test_labels.size(0)
            if not if_finetuning:
                acc_hist.append(a)
                loss_hist.append(l)

            test_loss += l
            test_acc += a

        return test_acc/VALIDATE_EPISODE,test_loss/VALIDATE_EPISODE

# net = ResidualNet(input_size=input_size,n=n,k=k,qk=qk,metric='Proto', block_num=5)
# net = ResidualNet(input_size=input_size,n=n,k=k,qk=qk,metric='Relation', block_num=6, hidden_size=64)
# net = RN(input_size, embed_size, hidden_size, k=k, n=n, qk=qk)
net = ProtoNet(k=k, n=n, qk=qk)
# net = SiameseNet(input_size=input_size, k=k, n=n)
states = t.load(MODEL_LOAD_PATH)
net.load_state_dict(states)
net = net.cuda()

# opt = Adam(net.parameters(), lr=lr)
opt = SGD(net.parameters(), lr=lr)
entro = nn.NLLLoss().cuda()
# entro = nn.MSELoss().cuda()
# entro = nn.CrossEntropyLoss().cuda()

# net.Layer1.requires_grad_(False)
# net.Layer2.requires_grad_(False)

# for name,par in net.named_parameters():
#     if name.find("fc") == -1:
#         par.requires_grad_(False)
#         print("---%s---"%name)
#     else:
#         par.requires_grad_(True)
#         print("***%s***" % name)

before_acc_total = 0.
before_loss_total = 0.
after_acc_total = 0.
after_loss_total = 0.

acc_gain_all = 0.
loss_gain_all = 0.
print(net)
rd.seed(time.time()%10000000)
for episode in range(TEST_EPISODE):
    s_time = time.time()
    states = t.load(MODEL_LOAD_PATH)
    net.load_state_dict(states)

    if if_finetuning:
        net.train()
        net.zero_grad()
    print("%d th episode"%episode)

    # 每一轮开始的时候先抽取n个实验类
    sample_classes = rd.sample(TEST_CLASSES, n)

    before_acc,before_loss = validate(net, entro, sample_classes)
    print("before:")
    print("acc: ", before_acc)
    print("loss: ", before_loss)
    print("------------------------------")
    before_acc_total += before_acc
    before_loss_total += before_loss

    if if_finetuning:
        sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N, episode)
        # sample_sampler,query_sampler = get_RN_modified_sampler(sample_classes, k, qk, N)

        train_sample_dataloader = DataLoader(dataset, batch_size=n*k, sampler=sample_sampler)
        # train_query_dataloader = DataLoader(dataset, batch_size=qk*n, sampler=query_sampler)

        samples,sample_labels = train_sample_dataloader.__iter__().next()
        # queries,query_labels = train_query_dataloader.__iter__().next()

        samples = samples.cuda()
        sample_labels = sample_labels.cuda()
        # queries = queries.cuda()
        # query_labels = query_labels.cuda()

        labels = RN_labelize(sample_labels, sample_labels, k, n, type="long", expand=False)
        # labels = RN_labelize(sample_labels, sample_labels, k, n, type="float", expand=True)



        for i in range(FINETUNING_EPISODE):
            # fine-tuning
            net.train()
            net.zero_grad()
            outs = net(samples, samples)

            loss = entro(outs, labels)

            loss.backward()

            opt.step()

            after_acc,after_loss = validate(net, entro, sample_classes, episode)
            after_acc_total += after_acc
            after_loss_total += after_loss

            acc_gain = after_acc-before_acc
            loss_gain = after_loss-before_loss



            print("after %d fine-tuning:"%i)
            print("acc: ", after_acc)
            print("loss: ", after_loss)
            print("------------------------------------")
            print("acc gain:", acc_gain)
            print("loss gain:", loss_gain)
            print("*************************************")

            acc_gain_all += acc_gain
            loss_gain_all += loss_gain
    print("time:%.2f"%(time.time()-s_time))

print("***********************************")
print("average acc:", np.mean(acc_hist))
print("average loss:", np.mean(loss_hist))
print("acc std var:", np.std(acc_hist))
print("loss std var:", np.std(loss_hist))
if if_finetuning:
    print("average acc after:", after_acc_total/TEST_EPISODE)
    print("average loss after:", after_loss_total/TEST_EPISODE)
    print("average acc gain: ", acc_gain_all/TEST_EPISODE)
    print("average loss gain: ", loss_gain_all/TEST_EPISODE)
    print("average acc gain ratio: ", (acc_gain_all/TEST_EPISODE)/(before_acc_total/TEST_EPISODE))
    print("average loss gain ratio: ", -1*(loss_gain_all/TEST_EPISODE)/(before_loss_total/TEST_EPISODE))

bar_frequency(acc_hist, "Test Accuracy Distribution\nAcc=%.3f"%np.mean(acc_hist))
