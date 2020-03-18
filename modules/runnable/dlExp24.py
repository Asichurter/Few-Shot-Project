# MetaSGD 测试程序

import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
import os

import time

from modules.model.MetaSGD import MetaSGD
from modules.utils.dlUtils import RN_labelize
from modules.utils.datasets import FewShotFileDataset, get_RN_sampler
from modules.utils.dlUtils import cal_beliefe_interval, labels_normalize

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
CROP_SIZE = 224
TEST_EPISODE = 1000

version = 2
type = "MetaSGD"
folder = 'cluster'
draw_confusion_matrix = False
conf_mat = []

VALIDATE_PATH = "D:/peimages/New/%s/test.npy"%folder
VALIDATE_LENGTH_PATH = "D:/peimages/New/%s/test/"%folder
mode = 'best_acc'
if_finetuning = False

MODEL_LOAD_PATH = "D:/peimages/New/%s/models/"%folder+\
                  "%s_%s_model_%dshot_%dway_v%d.0.h5"%(type,mode,k,n,version)

VALIDATE_EPISODE = 20

test_classes = len(os.listdir(VALIDATE_LENGTH_PATH))#50
TEST_CLASSES = [i for i in range(test_classes)]

dataset = FewShotFileDataset(VALIDATE_PATH, N, test_classes, rd_crop_size=CROP_SIZE, rotate=False)
# dataset = FewShotRNDataset(VALIDATE_PATH, N, rd_crop_size=224)

acc_hist = []
loss_hist = []

def bar_frequency(data, title, bins=10, color="blue", bar_width=0.2, precision=2):
    bin_interval = 1/bins
    x = np.arange(bins)*bar_width
    x_label = (x-bar_width/2)
    x_label = np.append(x_label, x_label[-1]+bar_width)
    x_ticks = [round(i*bin_interval, precision) for i in range(bins+1)]
    # print(x, x_label, x_ticks, sep='\n')
    data = np.floor(np.array(data)/bin_interval)
    frequency = [0]*bins
    for i in data:
        try:
            frequency[int(i)] += 1/len(data)
        except IndexError:
            if int(i)==10:
                frequency[9] += 1/len(data)
    plt.title(title)
    plt.bar(x, frequency, alpha=0.5, width=bar_width, color=color, edgecolor='black', label="frequency", lw=3)
    plt.xticks(x_label, x_ticks)
    plt.legend()
    plt.show()

def drawHeatmapWithGrid(data, title, col_labels, row_labels, cbar_label, formatter="%s", **kwargs):
    fig, ax = plt.subplots()
    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, formatter%data[i][j],
                           ha="center", va="center", color="k")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def validate(model, loss, classes, seed=None):
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

            support_labels = labels_normalize(support_labels, n=n, batch_length=k*n)
            test_labels = labels_normalize(test_labels, n=n, batch_length=qk*n)
            test_relations = net(supports, tests, support_labels)

            l = loss(test_relations, test_labels).item()
            a = (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)
            if not if_finetuning:
                acc_hist.append(a)
                loss_hist.append(l)

            test_loss += l
            test_acc += a

        return test_acc/VALIDATE_EPISODE,test_loss/VALIDATE_EPISODE

loss_fn = nn.NLLLoss().cuda()
if type == 'MetaSGD':
    net = MetaSGD(input_size=input_size, n=n, loss_fn=loss_fn)
else:
    assert False, "不支持的网络类型：%s"%type

states = t.load(MODEL_LOAD_PATH)
net.load_state_dict(states)
net = net.cuda()

before_acc_total = 0.
before_loss_total = 0.
after_acc_total = 0.
after_loss_total = 0.

acc_gain_all = 0.
loss_gain_all = 0.
print(net)
rd.seed(time.time()%10000000)
for episode in range(TEST_EPISODE):
    seed = time.time()%10000000
    s_time = time.time()
    states = t.load(MODEL_LOAD_PATH)
    net.load_state_dict(states)

    with no_grad:
        print("%d th episode"%episode)

        # 每一轮开始的时候先抽取n个实验类
        sample_classes = rd.sample(TEST_CLASSES, n)

        before_acc,before_loss = validate(net, loss_fn, sample_classes, seed=seed)
        print("acc: ", before_acc)
        print("loss: ", before_loss)
        print("------------------------------")
        before_acc_total += before_acc
        before_loss_total += before_loss
        print("time:%.2f"%(time.time()-s_time))

print("***********************************")
print("average acc:", np.mean(acc_hist) if not if_finetuning else before_acc_total/TEST_EPISODE)
print("average loss:", np.mean(loss_hist) if not if_finetuning else before_loss_total/TEST_EPISODE)
print('acc\'s 95%% belief interval: %f' % cal_beliefe_interval(acc_hist))
if if_finetuning:
    print("average acc after:", after_acc_total/TEST_EPISODE)
    print("average loss after:", after_loss_total/TEST_EPISODE)
    print('---------------------------------------')
    print("average acc gain: ", acc_gain_all/TEST_EPISODE)
    print("average loss gain: ", loss_gain_all/TEST_EPISODE)
    print("average acc gain ratio: ", (acc_gain_all/TEST_EPISODE)/(before_acc_total/TEST_EPISODE))
    print("average loss gain ratio: ", -1*(loss_gain_all/TEST_EPISODE)/(before_loss_total/TEST_EPISODE))
else:
    print("acc std var:", np.std(acc_hist))
    print("loss std var:", np.std(loss_hist))
    print('------------------------------------')

bar_frequency(acc_hist, "Test Accuracy Distribution\nAcc=%.3f"%np.mean(acc_hist))
