# 测试ResProtoNet

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import NLLLoss
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
from torch.optim.lr_scheduler import StepLR
import visdom

import time

from modules.model.ResProtoNet import ResProtoNet
from modules.utils.dlUtils import RN_labelize
from modules.utils.datasets import FewShotRNDataset, get_RN_sampler

# TRAIN_PATH = "D:/peimages/New/Residual_5shot_5way_exp/train/"
# TEST_PATH = "D:/peimages/New/Residual_5shot_5way_exp/validate/"
# MODEL_SAVE_PATH = "D:/peimages/New/Residual_5shot_5way_exp/models/"
# DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_ProtoNet_5shot_5way_exp/"

folder = 'cluster'
TRAIN_PATH = "D:/peimages/New/%s/train/" %folder
TEST_PATH = "D:/peimages/New/%s/validate/"%folder
MODEL_SAVE_PATH = "D:/peimages/New/%s/models/"%folder
DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_ResProto_exp/"

input_size = 256

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 10
# 一个类总共多少个样本
N = 20
# 学习率
lr = 1e-3

version = 2

TEST_CYCLE = 100
MAX_ITER = 60000
TEST_EPISODE = 100
ASK_CYCLE = 60000
ASK_THRESHOLD = 20000
REST_CYCLE = 2000
REST_TIME = 120
CROP_SIZE = 224

# 训练和测试中类的总数
train_classes = 100#len(os.listdir(TRAIN_PATH))
test_classes = 58#len(os.listdir(TEST_PATH))

TRAIN_CLASSES = [i for i in range(train_classes)]
TEST_CLASSES = [i for i in range(test_classes)]

IF_LOAD_MODEL = False

vis = visdom.Visdom(env="train monitoring")
acc_names = ["train acc", "validate acc"]
loss_names = ["train loss", "validate loss"]


train_dataset = FewShotRNDataset(TRAIN_PATH, N, rd_crop_size=CROP_SIZE)
test_dataset = FewShotRNDataset(TEST_PATH, N, rd_crop_size=CROP_SIZE)

net = ResProtoNet()
# net.load_state_dict(t.load(MODEL_SAVE_PATH+"ProtoNet_best_acc_model_%dshot_%dway_v%d.0.h5"%(k,n,26)))
net = net.cuda()

num_of_params = 0
for par in net.parameters():
    num_of_params += par.numel()
print('params:', num_of_params)

# net.apply(RN_weights_init)

opt = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
# opt = SGD(net.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
scheduler = StepLR(opt, step_size=20000 , gamma=0.1)
nll = NLLLoss().cuda()

train_acc_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_acc_train.npy"%version).tolist()
train_loss_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_loss_train.npy"%version).tolist()
test_acc_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_acc.npy"%version).tolist()
test_loss_his = [] if not IF_LOAD_MODEL else np.load(DOC_SAVE_PATH+"%d_loss.npy"%version).tolist()

best_acc = 0.
best_epoch = 0
previous_stamp = time.time()
rd.seed(time.time()%10000000)

global_step = 0

for i in range(10):
    time.sleep(1)
    print(10-i)
for episode in range(MAX_ITER):
    if episode%ASK_CYCLE == 0 and episode!=0 and episode <= ASK_THRESHOLD:
        choice = input("%d episode, continue?"%episode)
        if choice.find("no") != -1 or choice.find("n") != -1:
            break
    if episode%REST_CYCLE == 0 and episode!=0:
        print("episode %d, sleeping..."%episode)
        time.sleep(REST_TIME)
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

    labels = RN_labelize(sample_labels, query_labels, k, n, type="long", expand=False)

    outs = net(samples.view(n,k,1,CROP_SIZE,CROP_SIZE), queries.view(n*qk,1,CROP_SIZE,CROP_SIZE))

    loss = nll(outs, labels)
    # inner_var_loss = inner_var_alpha*net.forward_inner_var
    # outer_var_loss = -outer_var_alpha*net.forward_outer_var
    # total_loss = inner_var_loss + outer_var_loss + margin
    # total_loss = loss + inner_var_loss + outer_var_loss + margin

    loss.backward()

    # 使用了梯度剪裁
    # t.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()

    acc = (t.argmax(outs, dim=1)==labels).sum().item()/labels.size(0)
    loss_val = loss.item()

    print("train acc: ", acc)
    print("train loss: ", loss_val)
    # print("loss component:、\nnll: %f\ninner:%f\nouter:%f"%
    #       (loss.item(), inner_var_loss.item(),outer_var_loss.item()))
    print('----------------------------------------------')

    train_acc_his.append(acc)
    train_loss_his.append(loss_val)

    scheduler.step()

    if (episode + 1) % 5000 == 0:
        print("save!")
        t.save(net.state_dict(),
               MODEL_SAVE_PATH + "ResProtoNet_%d_epoch_model_%dshot_%dway_v%d.0.h5" % (episode + 1, k, n, version))

    if episode % TEST_CYCLE == 0:
        # input("----- Time to test -----")
        net.eval()
        print("test stage at %d episode"%episode)
        with no_grad():
            test_acc = 0.
            test_loss = 0.
            test_inner = 0.
            test_outer = 0.
            test_nll = 0.
            for j in range(TEST_EPISODE):
                print("episode %d: test %d"%(j,episode))
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

                test_labels = RN_labelize(support_labels, test_labels, k, n, type="long", expand=False)
                test_relations = net(supports.view(n,k,1,CROP_SIZE,CROP_SIZE), tests.view(n*qk,1,CROP_SIZE,CROP_SIZE))

                val_nll_loss = nll(test_relations, test_labels)
                # val_inner_var_loss = inner_var_alpha * net.forward_inner_var
                # val_outer_var_loss = -outer_var_alpha * net.forward_outer_var
                # val_total_loss = val_inner_var_loss + val_outer_var_loss + margin

                test_loss += val_nll_loss.item()
                test_acc += (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)

                # test_inner += val_inner_var_loss.item()
                # test_outer += val_outer_var_loss.item()
                test_nll += val_nll_loss.item()

            test_acc_his.append(test_acc/TEST_EPISODE)
            test_loss_his.append(test_loss/TEST_EPISODE)

            current_length = TEST_CYCLE if len(train_acc_his) >= TEST_CYCLE else 1
            current_train_acc = np.mean(train_acc_his[-1*current_length:])
            current_train_loss = np.mean(train_loss_his[-1*current_length:])

            # writer.add_scalars("Accuracy",
            #                    {"train":current_train_acc,"validate":test_acc/TEST_EPISODE},
            #                    global_step)
            # writer.add_scalars("Loss",
            #                    {"train":current_train_loss,"validate":test_loss/TEST_EPISODE},
            #                    global_step)

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

            global_step += TEST_CYCLE

            print("****************************************")
            print("train acc: ", current_train_acc)
            print("train loss: ", current_train_loss)
            print("----------------------------------------")
            print("val acc: ", test_acc/TEST_EPISODE)
            print("val loss: ", test_loss/TEST_EPISODE)
            # print("val nll loss: ", test_nll/TEST_EPISODE)
            # print("val inner loss: ", test_inner/TEST_EPISODE)
            # print("val outer loss: ", test_outer/TEST_EPISODE)
            if test_acc/TEST_EPISODE > best_acc:
                t.save(net.state_dict(),
                       MODEL_SAVE_PATH + "ResProtoNet_best_acc_model_%dshot_%dway_v%d.0.h5" % (k, n, version))
                print("model save at %d episode" % episode)
                best_acc = test_acc/TEST_EPISODE
                best_epoch = episode
            print("best val acc: ", best_acc)
            print("best epoch: %d"%best_epoch)
            now_stamp = time.time()
            print(TEST_CYCLE,"episode time consume:",now_stamp-previous_stamp)
            print("****************************************")
            previous_stamp = now_stamp
            # input("----- Test Complete ! -----")

# 根据历史值画出准确率和损失值曲线
train_x = [i*TEST_EPISODE for i in range(len(test_acc_his))]
test_x = [i*TEST_EPISODE for i in range(len(test_acc_his))]

# 修改为计算TEST_CYCLE轮内的正确率和损失值的平均值
train_acc_plot = np.array(train_acc_his).reshape(-1,TEST_CYCLE).mean(axis=1).reshape(-1).tolist()
train_loss_plot = np.array(train_loss_his).reshape(-1,TEST_CYCLE).mean(axis=1).reshape(-1).tolist()

plt.title('%d-shot %d-way ResProtoNet Accuracy'%(k,n))
plt.plot(train_x, train_acc_plot, linestyle='-', color='blue', label='train')
plt.plot(test_x, test_acc_his, linestyle='-', color='red', label='validate')
plt.plot(train_x, [1/n]*len(train_x), linestyle='--', color="black", label="baseline")
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_acc.png'%version)
plt.show()

plt.title('%d-shot %d-way ResProto Net Loss'%(k,n))
plt.ylim(0,2)
plt.plot(train_x, train_loss_plot, linestyle='-', color='blue', label='train')
plt.plot(test_x, test_loss_his, linestyle='-', color='red', label='validate')
plt.legend()
plt.savefig(DOC_SAVE_PATH + '%d_loss.png'%version)
plt.show()

np.save(DOC_SAVE_PATH+"%d_acc.npy"%version, np.array(test_acc_his))
np.save(DOC_SAVE_PATH+"%d_loss.npy"%version, np.array(test_loss_his))

