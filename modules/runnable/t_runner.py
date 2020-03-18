# SNAIL 测试程序

import torch as t
import torch.nn as nn
import numpy as np
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
import os

from modules.model.SNAIL import SNAIL
from modules.model.MetaSGD import MetaSGD
from modules.model.ChannelNet import ChannelNet
from modules.model.RoutingNet import RoutingNet
from modules.model.PrototypicalNet import ProtoNet
from modules.model.HybridAttentionProtoNet import HAPNet
from modules.utils.dlUtils import RN_labelize
from modules.utils.datasets import FewShotFileDataset, get_RN_sampler
from modules.utils.dlUtils import cal_beliefe_interval, labels_normalize, labels_one_hot

folders = ['cluster','test','virushare_20','drebin_10', 'miniImageNet']
Ns = {'cluster':20, 'test':20, 'virushare_20':20, 'drebin_10':10, 'miniImageNet':600}
in_channels = {'cluster':1, 'test':1, 'virushare_20':1, 'drebin_10':1, 'miniImageNet':3}

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15

# 学习率
lr = 1e-3
CROP_SIZE = 84
TEST_EPISODE = 5000
version = 60
type = "ProtoNet"
folder = 'miniImageNet'
draw_confusion_matrix = False
conf_mat = []
# 一个类总共多少个样本
N = Ns[folder]

metric_based_models = ['RoutingNet', 'ProtoNet', 'ChannelNet', 'HybridAttentionNet']

# 对于不同的模型，调整本方法来进行调用
def data_input(model, **datas):
    if type == 'SNAIL':
        return model(datas['support'],
                     datas['test'],
                     datas['s_label'])[:,-1,:].squeeze()
    elif type == 'MetaSGD':
        return model(datas['support'],
                     datas['test'],
                     datas['s_label'])

    elif type in metric_based_models :
        return model(datas['support'].view(n,k,in_channels[folder],CROP_SIZE,CROP_SIZE),
                     datas['test'].view(qk*n,in_channels[folder],CROP_SIZE,CROP_SIZE))

def label_processing(s_label, q_label):
    if type=='SNAIL':
        test_labels = RN_labelize(s_label, q_label, k, n, type="long", expand=False)
        support_labels = labels_normalize(s_label, k, n, batch_length=k * n)
        support_labels = labels_one_hot(support_labels, n=n)

    elif type == 'MetaSGD':
        test_labels =  RN_labelize(s_label, q_label, k, n, type="long", expand=False)
        support_labels = labels_normalize(s_label, k, n, batch_length=k * n)

    elif type in metric_based_models :
    # ChannelNet的标签对位置有依赖，必须使用labelize方法
        test_labels = RN_labelize(s_label, q_label, k, n, type="long", expand=False)
        support_labels = s_label

    return support_labels, test_labels

VALIDATE_PATH = "D:/peimages/New/%s/test.npy"%folder
VALIDATE_LENGTH_PATH = "D:/peimages/New/%s/test/"%folder
mode = 'best_acc'

MODEL_LOAD_PATH = "D:/peimages/New/%s/models/"%folder+\
                  "%s_%s_model_%dshot_%dway_v%d.0.h5"%(type,mode,k,n,version)

test_classes = len(os.listdir(VALIDATE_LENGTH_PATH))#50
TEST_CLASSES = [i for i in range(test_classes)]

dataset = FewShotFileDataset(VALIDATE_PATH, N, test_classes, rd_crop_size=CROP_SIZE, rotate=False)

acc_hist = []
loss_hist = []

nll = nn.NLLLoss().cuda()

if type == 'SNAIL':
    net = SNAIL(n,k)
elif type == 'MetaSGD':
    net = MetaSGD(input_size=CROP_SIZE,n=n,loss_fn=nll)
elif type == 'RoutingNet':
    net = RoutingNet()
elif type == 'ProtoNet':
    net = ProtoNet(in_channels=in_channels[folder])
elif type == 'HybridAttentionNet':
    net = HAPNet(n, k, qk)
elif type == 'ChannelNet':
    net = ChannelNet(k, in_channels=in_channels[folder])

net.load_state_dict(t.load(MODEL_LOAD_PATH))
net = net.cuda()

net.eval()
test_acc_his = []
test_loss_his = []
for j in range(TEST_EPISODE):
    print("-------------episode %d-------------" % j)
    # 每一轮开始的时候先抽取n个实验类
    support_classes = rd.sample(TEST_CLASSES, n)
    # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
    support_sampler, test_sampler = get_RN_sampler(support_classes, k, qk, N)
    test_support_dataloader = DataLoader(dataset, batch_size=n * k,
                                         sampler=support_sampler)
    test_test_dataloader = DataLoader(dataset, batch_size=qk * n,
                                      sampler=test_sampler)

    supports, support_labels = test_support_dataloader.__iter__().next()
    tests, test_labels = test_test_dataloader.__iter__().next()

    support_labels, test_labels = label_processing(support_labels, test_labels)

    supports = supports.cuda()
    support_labels = support_labels.cuda()
    tests = tests.cuda()
    test_labels = test_labels.cuda()

    test_relations = data_input(net,
                                support=supports,
                                test=tests,
                                s_label=support_labels,
                                t_label=test_labels)#net(supports, tests, support_labels)[:, -1, :].squeeze()

    val_nll_loss = nll(test_relations, test_labels)

    test_loss = val_nll_loss.item()
    test_acc = (t.argmax(test_relations, dim=1) == test_labels).sum().item() / test_labels.size(0)

    test_acc_his.append(test_acc)
    test_loss_his.append(test_loss)

    print('acc:', test_acc)
    print('loss:', test_loss)

print('**************************************************')
print('average acc:', np.mean(test_acc_his))
print('average loss:', np.mean(test_loss_his))
print('acc 95%% belief interval:', cal_beliefe_interval(test_acc_his))