# 本实验用于测试SNAIL

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import NLLLoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from torch.optim.lr_scheduler import StepLR
import visdom
import os

import time

from modules.model.SNAIL_ import SNAIL
from modules.utils.dlUtils import net_init, RN_labelize, labels_one_hot, labels_normalize_
from modules.utils.datasets import FewShotFileDataset, SNAIL_sampler, get_RN_sampler

folders = ['cluster','test','virushare_20','drebin_10']
Ns = {'cluster':20, 'test':20, 'virushare_20':20, 'drebin_10':10}
data_folder = 'cluster'

PATH = "D:/peimages/New/%s/"%data_folder
TRAIN_FILE_PATH =  PATH+'train.npy'
TEST_FILE_PATH = PATH+'validate.npy'
MODEL_SAVE_PATH = "D:/peimages/New/%s/models/"%data_folder
DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_SNAIL_exp/"

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
qk = 15
batch_size = 4

# 一个类总共多少个样本
N = Ns[data_folder]
# 学习率
lr = 1e-3

version = 3

TEST_CYCLE = 100
MAX_ITER = 50000
TEST_EPISODE = 100
ASK_CYCLE = 60000
ASK_THRESHOLD = 20000
CROP_SIZE = 224
FRESH_CYCLE = 1000

# 训练和测试中类的总数
train_classes = len(os.listdir(PATH+'train/'))
test_classes = len(os.listdir(PATH+'validate/'))

TRAIN_CLASSES = [i for i in range(train_classes)]
TEST_CLASSES = [i for i in range(test_classes)]

IF_LOAD_MODEL = False

vis = visdom.Visdom(env="train monitoring")
acc_names = ["train acc", "validate acc"]
loss_names = ["train loss", "validate loss"]

train_dataset = FewShotFileDataset(TRAIN_FILE_PATH, N, class_num=train_classes, rd_crop_size=CROP_SIZE)
test_dataset = FewShotFileDataset(TEST_FILE_PATH, N, class_num=test_classes, rd_crop_size=CROP_SIZE)

net = SNAIL(N=n, K=k)
# net.load_state_dict(t.load(MODEL_SAVE_PATH+"SNAIL_best_acc_model_%dshot_%dway_v%d.0.h5"%(k,n,2)))
net = net.cuda()

num_of_params = 0
for par in net.parameters():
    num_of_params += par.numel()
print('params:', num_of_params)

# net.apply(net_init)

opt = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
# opt = SGD(net.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)
scheduler = StepLR(opt, step_size=15000 , gamma=0.1)
nll = NLLLoss().cuda()

train_acc_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_acc_train.npy"%version).tolist()
train_loss_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_loss_train.npy"%version).tolist()
test_acc_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_acc.npy"%version).tolist()
test_loss_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_loss.npy"%version).tolist()
proto_norms = []
time_consuming = []

best_acc = 0.
best_epoch = 0
previous_stamp = time.time()
rd.seed(time.time()%10000000)

global_step = 0

for episode in range(MAX_ITER):
    if episode % ASK_CYCLE == 0 and episode != 0 and episode <= ASK_THRESHOLD:
        choice = input("%d episode, continue?" % episode)
        if choice.find("no") != -1 or choice.find("n") != -1:
            break
    net.train()
    net.zero_grad()
    print("%d th episode" % episode)

    train_sampler = SNAIL_sampler(k, n, train_classes, N, batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size * (k * n + 1), sampler=train_sampler)

    samples, sample_labels = train_dataloader.__iter__().next()
    sample_labels = labels_normalize_(sample_labels, n, batch_length=k * n + 1)
    samples = samples.cuda()
    targets = sample_labels[k * n::(k * n + 1)]  # 获取测试值的标签，因为每k*n后面会有一个额外的样本作
    # 为测试样本，形成一个batch
    sample_labels = labels_one_hot(sample_labels, n=n).cuda()
    targets = targets.cuda()

    # 每一轮开始的时候先抽取n个实验类
    # sample_classes = rd.sample(TRAIN_CLASSES, n)

    # # sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)
    # sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)

    # train_sample_dataloader = DataLoader(train_dataset, batch_size=n*k, sampler=sample_sampler)
    # train_query_dataloader = DataLoader(train_dataset, batch_size=qk*n, sampler=query_sampler)

    # samples,sample_labels = train_sample_dataloader.__iter__().next()
    # queries,query_labels = train_query_dataloader.__iter__().next()

    # labels = RN_labelize(sample_labels, query_labels, k, n, type="long",expand=False).cuda()

    # sample_labels = labels_normalize(sample_labels, k, n, batch_length=k*n)
    # sample_labels = labels_one_hot(sample_labels, n=n)

    # samples = samples.cuda()
    # sample_labels = sample_labels.cuda()
    # queries = queries.cuda()
    # query_labels = query_labels.cuda()

    outs = net(samples, sample_labels)[:, -1, :].squeeze()  # 取一个batch的最后一个输出
    # 因为batch的最后一个是测试样本
    loss = nll(outs, targets)

    loss.backward()
    opt.step()

    acc = (t.argmax(outs, dim=1) == targets).sum().item() / targets.size(0)
    loss_val = loss.item()

    train_acc_his.append(acc)
    train_loss_his.append(loss_val)

    scheduler.step()

    if episode % TEST_CYCLE == -1:
        # input("----- Time to test -----")
        net.eval()
        print("test stage at %d episode" % episode)
        with no_grad():
            test_acc = 0.
            test_loss = 0.
            test_inner = 0.
            test_outer = 0.
            test_nll = 0.
            for j in range(TEST_EPISODE):
                print("episode %d: test %d"%(j,episode))
                # 每一轮开始的时候先抽取n个实验类
                # support_classes = rd.sample(TEST_CLASSES, n)
                # # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
                # support_sampler, test_sampler = get_RN_sampler(support_classes, k, qk, N)
                # # support_sampler, test_sampler = get_RN_modified_sampler(support_classes, k, qk, N)
                # test_support_dataloader = DataLoader(test_dataset, batch_size=n * k,
                #                                      sampler=support_sampler)
                # test_test_dataloader = DataLoader(test_dataset, batch_size=qk * n,
                #                                     sampler=test_sampler)

                # supports, support_labels = test_support_dataloader.__iter__().next()
                # tests, test_labels = test_test_dataloader.__iter__().next()

                # test_labels = RN_labelize(support_labels, test_labels, k, n, type="long", expand=False)
                # support_labels = labels_normalize(support_labels, k, n, batch_length= k * n)
                # support_labels = labels_one_hot(support_labels, n=n)

                # supports = supports.cuda()
                # support_labels = support_labels.cuda()
                # tests = tests.cuda()
                # test_labels = test_labels.cuda()

                test_sampler = SNAIL_sampler(k, n, test_classes, N, batch_size)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size * (k * n + 1), sampler=test_sampler)

                supports, support_labels = test_dataloader.__iter__().next()
                support_labels = labels_normalize_(support_labels, n, batch_length=k * n + 1)
                supports = supports.cuda()
                test_labels = support_labels[k * n::(k * n + 1)]  # 获取测试值的标签，因为每k*n后面会有一个额外的样本作
                # 为测试样本，形成一个batch
                support_labels = labels_one_hot(support_labels, n=n).cuda()
                test_labels = test_labels.cuda()

                test_relations = net(supports, support_labels)[:, -1, :].squeeze()

                val_nll_loss = nll(test_relations, test_labels)

                test_loss += val_nll_loss.item()
                test_acc += (t.argmax(test_relations, dim=1) == test_labels).sum().item() / test_labels.size(0)

                test_nll += val_nll_loss.item()

            test_acc_his.append(test_acc / TEST_EPISODE)
            test_loss_his.append(test_loss / TEST_EPISODE)
            # proto_norms.append(net.ProtoNorm)

            current_length = TEST_CYCLE if len(train_acc_his) >= TEST_CYCLE else 1
            current_train_acc = np.mean(train_acc_his[-1 * current_length:])
            current_train_loss = np.mean(train_loss_his[-1 * current_length:])
            # current_norm = np.mean(proto_norms[-1*current_length:])

            now_stamp = time.time()
            time_consuming.append(now_stamp - previous_stamp)
            plot_time = np.array([time_consuming[-1]])

            global_step += TEST_CYCLE

            print("****************************************")
            print("train acc: ", current_train_acc)
            print("train acc: ", current_train_loss)
            print("----------------------------------------")
            print("val acc: ", test_acc / TEST_EPISODE)
            print("val loss: ", test_loss / TEST_EPISODE)
            # print("val nll loss: ", test_nll/TEST_EPISODE)
            # print("val inner loss: ", test_inner/TEST_EPISODE)
            # print("val outer loss: ", test_outer/TEST_EPISODE)
            if test_acc / TEST_EPISODE > best_acc:
                # t.save(net.state_dict(),
                #        MODEL_SAVE_PATH + "SNAIL_best_acc_model_%dshot_%dway_v%d.0.h5" % (k, n, version))
                print("model save at %d episode" % episode)
                best_acc = test_acc / TEST_EPISODE
                best_epoch = episode
            print("best val acc: ", best_acc)
            print("best epoch: %d" % best_epoch)
            print(TEST_CYCLE, "episode time consume:", now_stamp - previous_stamp)
            print("****************************************")
            previous_stamp = now_stamp
            print('%d -> %d epoches...' % (episode, episode + TEST_CYCLE))

