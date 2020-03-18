# 本实验用于启动深度学习实验

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
from torch.nn import NLLLoss, MSELoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from torch.optim.lr_scheduler import StepLR
import visdom
import os

import time

from modules.model.ChannelNet import ChannelNet
from modules.model.RoutingNet import RoutingNet
from modules.model.PrototypicalNet import ProtoNet
from modules.model.ResChannelNet import ResChannelNet
from modules.model.ReMalAttNet import ReMalAttNet
from modules.utils.dlUtils import net_init, RN_labelize
from modules.utils.datasets import FewShotFileDataset, get_RN_sampler

folders = ['cluster','test','virushare_20','drebin_10', 'miniImageNet']
Ns = {'cluster':20, 'test':20, 'virushare_20':20, 'drebin_10':10, 'miniImageNet':600}
in_channels = {'cluster':1, 'test':1, 'virushare_20':1, 'drebin_10':1, 'miniImageNet':3}
loss_func_dict = {'nll':NLLLoss(), 'mse':MSELoss()}

#-----------------------------------------------------------------
model = 'ReMalAttNet'
data_folder = 'virushare_20'
# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 2
# 一个类总共多少个样本
N = Ns[data_folder]
# 学习率
lr = 1e-3

version = 1
CROP_SIZE = 256
MAX_ITER = 50000
loss_func = 'nll'
optimizer = 'adam'
initilized = False
#-----------------------------------------------------------------

PATH = "D:/peimages/New/%s/"%data_folder
TRAIN_FILE_PATH =  PATH+'train.npy'
TEST_FILE_PATH = PATH+'validate.npy'
MODEL_SAVE_PATH = "D:/peimages/New/%s/models/"%data_folder
DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_%s_exp/" % model

TEST_CYCLE = 100
TEST_EPISODE = 100
ASK_CYCLE = 60000
ASK_THRESHOLD = 20000
FRESH_CYCLE = 1000
lr_decay_iterations = 15000
magic = 7355605

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

#-----------------------------------------------------------------
if model == 'RoutingNet':
    net = RoutingNet(iters=3)
elif model == 'ProtoNet':
    net = ProtoNet(in_channels=in_channels[data_folder])
elif model == 'ResChannelNet':
    net = ResChannelNet(k=k, in_channel=in_channels[data_folder])
elif model == 'ReMalAttNet':
    net = ReMalAttNet(in_channel=in_channels[data_folder],
                      in_size=CROP_SIZE,
                      hidden_size=128)
# net = ChannelNet(k=k)
#-----------------------------------------------------------------

# net.load_state_dict(t.load(MODEL_SAVE_PATH+"ProtoNet_best_acc_model_%dshot_%dway_v%d.0.h5"%(k,n,26)))
net = net.cuda()

num_of_params = 0
for par in net.parameters():
    num_of_params += par.numel()
print('params:', num_of_params)

if initilized:
    net.apply(net_init)

#-----------------------------------------------------------------
if optimizer == 'adam':
    opt = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
elif optimizer == 'rmsprop':
    opt = RMSprop(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
elif optimizer == 'adagrad':
    opt = Adagrad(net.parameters(), lr=lr, weight_decay=1e-4)
#-----------------------------------------------------------------

# opt = SGD(net.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)
scheduler = StepLR(opt, step_size=lr_decay_iterations, gamma=0.1)

nll = loss_func_dict[loss_func].cuda()

train_acc_his = []
train_loss_his = []
test_acc_his = []
test_loss_his = []
proto_norms = []
time_consuming = []

best_acc = 0.
best_epoch = 0
previous_stamp = time.time()
rd.seed(time.time()%magic)

global_step = 0

for episode in range(MAX_ITER):
    if episode%ASK_CYCLE == 0 and episode!=0 and episode <= ASK_THRESHOLD:
        choice = input("%d episode, continue?"%episode)
        if choice.find("no") != -1 or choice.find("n") != -1:
            break
    net.train()
    net.zero_grad()
    print("%d th episode"%episode)

    # 每一轮开始的时候先抽取n个实验类
    sample_classes = rd.sample(TRAIN_CLASSES, n)

    # sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)
    sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N)

    train_sample_dataloader = DataLoader(train_dataset, batch_size=n*k, sampler=sample_sampler)
    train_query_dataloader = DataLoader(train_dataset, batch_size=qk*n, sampler=query_sampler)

    samples,sample_labels = train_sample_dataloader.__iter__().next()
    queries,query_labels = train_query_dataloader.__iter__().next()

    samples = samples.cuda()
    sample_labels = sample_labels.cuda()
    queries = queries.cuda()
    query_labels = query_labels.cuda()

    labels = RN_labelize(sample_labels, query_labels, k, n,
                            type="long" if loss_func=='nll' else 'float',
                            expand=(loss_func=='mse'))

    outs = net(samples.view(n,k,in_channels[data_folder],CROP_SIZE,CROP_SIZE),
               queries.view(n*qk,in_channels[data_folder],CROP_SIZE,CROP_SIZE))

    loss = nll(outs, labels)

    loss.backward()

    # 使用了梯度剪裁
    opt.step()

    if loss_func == 'mse':
        acc = (t.argmax(outs.view(-1,n), dim=1)==labels.view(-1,n).argmax(dim=1)).sum().item() / labels.size(0)
    else:
        acc = (t.argmax(outs, dim=1)==labels).sum().item() / labels.size(0)

    loss_val = loss.item()

    print("train acc: ", acc)
    print("train loss: ", loss_val)
    # print('----------------------------------------------')

    train_acc_his.append(acc)
    train_loss_his.append(loss_val)

    scheduler.step()

    # if (episode + 1) % 5000 == 0:
    #     print("save!")

    if episode % TEST_CYCLE == 0:
        net.eval()
        print("test stage at %d episode"%episode)
        with no_grad():
            test_acc = 0.
            test_loss = 0.
            test_inner = 0.
            test_outer = 0.
            test_nll = 0.
            for j in range(TEST_EPISODE):
                # print("episode %d: test %d"%(j,episode))
                # 每一轮开始的时候先抽取n个实验类
                support_classes = rd.sample(TEST_CLASSES, n)
                # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
                support_sampler, test_sampler = get_RN_sampler(support_classes, k, qk, N)
                # support_sampler, test_sampler = get_RN_modified_sampler(support_classes, k, qk, N)
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

                test_labels = RN_labelize(support_labels, test_labels, k, n,
                                          type="long" if loss_func=='nll' else 'float',
                                          expand=(loss_func=='mse'))
                test_relations = net(supports.view(n,k,in_channels[data_folder],CROP_SIZE,CROP_SIZE),
                                     tests.view(n*qk,in_channels[data_folder],CROP_SIZE,CROP_SIZE))

                val_nll_loss = nll(test_relations, test_labels)

                test_loss += val_nll_loss.item()

                if loss_func == 'mse':
                    test_acc += (t.argmax(test_relations.view(-1, n), dim=1) == test_labels.view(-1, n).argmax(dim=1)).sum().item() / test_labels.size(0)
                else:
                    test_acc += (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)


                test_nll += val_nll_loss.item()

            test_acc_his.append(test_acc/TEST_EPISODE)
            test_loss_his.append(test_loss/TEST_EPISODE)
            # proto_norms.append(net.ProtoNorm)

            current_length = TEST_CYCLE if len(train_acc_his) >= TEST_CYCLE else 1
            current_train_acc = np.mean(train_acc_his[-1*current_length:])
            current_train_loss = np.mean(train_loss_his[-1*current_length:])
            # current_norm = np.mean(proto_norms[-1*current_length:])

            plot_x = np.ones((1,2))*global_step
            plot_acc = np.array([current_train_acc, test_acc/TEST_EPISODE]).reshape((1,2))
            plot_loss = np.array([current_train_loss, test_loss/TEST_EPISODE]).reshape((1,2))
            acc_line = vis.line(X=plot_x,
                                Y=plot_acc,
                                win="acc",
                                opts=dict(
                                    legend=acc_names,
                                    title="Accuracy",
                                    xlabel="Iterations",
                                    ylabel="Accuracy"
                                ),
                                update=None if episode==0 else "append")
            loss_line = vis.line(X=plot_x,
                                Y=plot_loss,
                                 win="loss",
                                opts=dict(
                                    legend=loss_names,
                                    title="Loss",
                                    xlabel="Iterations",
                                    ylabel="Loss"
                                ),
                                update=None if episode==0 else "append")
            # plot_norm = np.array([current_norm])
            # plot_norm_x = np.array([1]) * episode
            # norm_line = vis.line(X=plot_norm_x,
            #                      Y=plot_norm,
            #                      win="Proto Norm",
            #                      opts=dict(
            #                          title="ProtoNorm",
            #                          xlabel="Iterations",
            #                          ylabel="Norm"
            #                      ),
            #                      # update=None if episode == 0 else "append")
            #                      update=None if episode == 0 else "append")
            now_stamp = time.time()
            time_consuming.append(now_stamp - previous_stamp)
            plot_time = np.array([time_consuming[-1]])

            global_step += TEST_CYCLE

            print("****************************************")
            print("train acc: ", current_train_acc)
            print("train acc: ", current_train_loss)
            print("----------------------------------------")
            print("val acc: ", test_acc/TEST_EPISODE)
            print("val loss: ", test_loss/TEST_EPISODE)

            if test_acc/TEST_EPISODE > best_acc:
                t.save(net.state_dict(),
                       MODEL_SAVE_PATH + "%s_best_acc_model_%dshot_%dway_v%d.0.h5" % (model, k, n, version))
                print("model save at %d episode" % episode)
                best_acc = test_acc/TEST_EPISODE
                best_epoch = episode
            print("best val acc: ", best_acc)
            print("best epoch: %d"%best_epoch)
            print(TEST_CYCLE,"episode time consume:",now_stamp-previous_stamp)
            print("****************************************")
            previous_stamp = now_stamp
            print('%d -> %d epoches...'%(episode, episode+TEST_CYCLE))

# 根据历史值画出准确率和损失值曲线
train_x = [i*TEST_EPISODE for i in range(len(test_acc_his))]
test_x = [i*TEST_EPISODE for i in range(len(test_acc_his))]

# 修改为计算TEST_CYCLE轮内的正确率和损失值的平均值
train_acc_plot = np.array(train_acc_his).reshape(-1,TEST_CYCLE).mean(axis=1).reshape(-1).tolist()
train_loss_plot = np.array(train_loss_his).reshape(-1,TEST_CYCLE).mean(axis=1).reshape(-1).tolist()

plt.title('%d-shot %d-way %s Accuracy'%(k,n,model))
plt.plot(train_x, train_acc_plot, linestyle='-', color='blue', label='train')
plt.plot(test_x, test_acc_his, linestyle='-', color='red', label='validate')
plt.plot(train_x, [1/n]*len(train_x), linestyle='-', color="green", label="baseline")
plt.grid(True, axis='y', color='black' ,linestyle='--')
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_acc.png'%version)
plt.show()

plt.title('%d-shot %d-way %s Loss'%(k,n,model))
plt.plot(train_x, train_loss_plot, linestyle='-', color='blue', label='train')
plt.plot(test_x, test_loss_his, linestyle='-', color='red', label='validate')
plt.grid(True, axis='y', color='black' ,linestyle='--')
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_loss.png'%version)
plt.show()

np.save(DOC_SAVE_PATH+"%d_acc.npy"%version, np.array(test_acc_his))
np.save(DOC_SAVE_PATH+"%d_loss.npy"%version, np.array(test_loss_his))
