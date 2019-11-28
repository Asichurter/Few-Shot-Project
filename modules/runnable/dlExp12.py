# 本实验用于测试Residual-Relation网络的微调

import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
import os

import time

from modules.model.PrototypicalNet import ProtoNet
from modules.model.ChannelNet import ChannelNet
from modules.model.RelationNet import RN
from modules.model.HybridAttentionProtoNet import HAPNet
from modules.utils.dlUtils import RN_labelize
from modules.utils.datasets import FewShotFileDataset, get_RN_sampler
from modules.utils.dlUtils import cal_beliefe_interval

input_size = 256

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 10
# 测试时每个类多少个样本
qk = 5
# 一个类总共多少个样本
N = 10
# 学习率

lr = 1e-3
CROP_SIZE = 256
TEST_EPISODE = 500

version = 42
type = "ChannelNet"
draw_confusion_matrix = False
conf_mat = []

folder = 'drebin_10'
VALIDATE_PATH = "D:/peimages/New/%s/test.npy"%folder
VALIDATE_LENGTH_PATH = "D:/peimages/New/%s/test/"%folder
mode = 'best_acc'
if_finetuning = False

MODEL_LOAD_PATH = "D:/peimages/New/%s/models/"%folder+"%s_%s_model_%dshot_%dway_v%d.0.h5"%(type,mode,k,n,version)

def freeze_weight_func(net):
    for name,par in net.named_parameters():
        if name.find('ProtoNet') == -1:
            par.requires_grad_(False)
        else:
            par.requires_grad_(True)

inner_var_alpha = 1e-2
outer_var_alpha = 1e-2*(k-1)*n
margin = 1

VALIDATE_EPISODE = 20
FINETUNING_EPISODE = 10


embed_size = 7
hidden_size = 8

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

            # test_labels = RN_labelize(support_labels, test_labels, k, n, type="float", expand=True)
            test_labels = RN_labelize(support_labels, test_labels, k, n, type="long", expand=False)
            test_relations = net(supports.view(n,k,1,CROP_SIZE,CROP_SIZE), tests.view(qk*n,1,CROP_SIZE,CROP_SIZE))
            # test_relations = net(supports, tests)

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
# net = ResProtoNet()
if type == 'ProtoNet':
    net = ProtoNet()
elif type == 'ChannelNet':
    net = ChannelNet(k=k)
elif type == 'RelationNet':
    net  = RN(linear_hidden_size=8)
elif type == 'HybridAttentionNet':
    net = HAPNet(n=n, k=k, qk=qk)
else:
    assert False, "不支持的网络类型：%s"%type
# net = ProtoNet(k=k, n=n, qk=qk)
# net = SiameseNet(input_size=input_size, k=k, n=n)
states = t.load(MODEL_LOAD_PATH)
net.load_state_dict(states)
net = net.cuda()

# ---------------------------
# 冻结网络中部分权重
# freeze_weight_func(net)
# ---------------------------

# opt = Adam(net.parameters(), lr=lr)
opt = SGD(net.parameters(), lr=lr)
entro = nn.CrossEntropyLoss().cuda()
# entro = nn.NLLLoss().cuda()
# entro = nn.MSELoss().cuda()

# net.Layer1.requires_grad_(False)
# net.Layer2.requires_grad_(False)

# for name,par in net.named_parameters():
#     # print(name)
#     if name.find("Transformer") != -1:
#         par.requires_grad_(True)
#         print("---%s---"%name)
#     else:
#         par.requires_grad_(False)
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
    seed = time.time()%10000000
    s_time = time.time()
    states = t.load(MODEL_LOAD_PATH)
    net.load_state_dict(states)

    if if_finetuning:
        net.train()
        net.zero_grad()
    print("%d th episode"%episode)

    # 每一轮开始的时候先抽取n个实验类
    sample_classes = rd.sample(TEST_CLASSES, n)

    before_acc,before_loss = validate(net, entro, sample_classes, seed=seed)
    print("before:")
    print("acc: ", before_acc)
    print("loss: ", before_loss)
    print("------------------------------")
    before_acc_total += before_acc
    before_loss_total += before_loss

    if if_finetuning:
        sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N, seed)
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
            outs = net(samples.view(n,k,1,CROP_SIZE,CROP_SIZE), samples.view(n*k,1,CROP_SIZE,CROP_SIZE))

            # loss = entro(outs, labels)
            # inner_var_loss = inner_var_alpha * net.forward_inner_var
            # outer_var_loss = -outer_var_alpha * net.forward_outer_var

            # loss  = margin + inner_var_loss + outer_var_loss
            f_loss = entro(outs, labels)

            f_loss.backward()

            opt.step()

        after_acc,after_loss = validate(net, entro, sample_classes, seed)
        after_acc_total += after_acc
        after_loss_total += after_loss

        acc_gain = after_acc-before_acc
        loss_gain = after_loss-before_loss



        print("after %d fine-tuning:"%FINETUNING_EPISODE)
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
