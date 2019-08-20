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
DOC_SAVE_PATH = "D:/Few-Shot-Project/doc/dl_relation_net_exp/"
MODEL_SAVE_PATH_MY = "D:/peimages/New/RN_5shot_5way_exp/embed_5shot_5way_v8.0.h5"

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

weighs = {'conv': ['Embed.layer1.0.weight', 'Embed.layer2.0.weight', 'Embed.layer3.0.weight', 'Embed.layer4.0.weight',
                   'Relation.layer1.0.weight', 'Relation.layer2.0.weight'],
          'linear': ['Relation.fc1.weight', 'Relation.fc2.weight']}


def visualize_filters(tensors, shape):
    # fig2 = plt.figure(constrained_layout=True, figsize=(shape[0],shape[1]))
    # spec2 = gridspec.GridSpec(ncols=shape[0], nrows=shape[1], figure=fig2)
    a = np.random.randint(0, 256, (3, 3))
    fig = plt.figure(figsize=(8, 8))
    for i, tensor in enumerate(tensors):
        # ax = fig2.add_subplot(spec2[int(i/shape[1]), i%shape[1]])
        plt.subplot(8, 8, i + 1)
        plt.title("%d" % i)
        plt.axis("off")
        plt.imshow(tensor.detach().numpy(), cmap="gray")
    plt.show()


def inner_class_divergence(tensors, metric="Man"):
    num = len(tensors)
    total_div = 0.
    count = 0
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            count += 1
            if metric == "Man":
                div = t.abs(tensors[i] - tensors[j]).sum()
            if metric == "Euc":
                div = t.sqrt(((tensors[i] - tensors[j]) ** 2).sum())
            total_div += div
    return total_div / count


def outer_class_divergence(tensors_x, tensors_y, metric="Man"):
    total_div = 0
    count = 0
    for x in tensors_x:
        for y in tensors_y:
            count += 1
            if metric == "Man":
                div = t.abs(x - y).sum()
            if metric == "Euc":
                div = t.sqrt(((x - y) ** 2).sum())
            total_div += div
    return total_div / count


def div_hist(inputs, labels, class_labels, ylabel, title, format='%s', ylim=None, xlabel=None):
    num = len(inputs)  # 多少种类输入
    x = np.arange(len(labels))  # the label locations
    width = 0.2 * num  # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    for i in range(num):
        rects.append(ax.bar(x - (num * width / 2) + (i + 0.5) * width, inputs[i], width, label=class_labels[i]))
    # rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
    # rects2 = ax.bar(x + width / 2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if ylim is not None:
        plt.ylim(*ylim)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(format % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)
    for rect in rects:
        autolabel(rect)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    embed = EmbeddingNet()
    embed.load_state_dict(t.load(MODEL_SAVE_PATH_MY))
    embed = embed.cuda()

    # rn = RN(input_size, hidder_size, k=k, n=n, qk=qk)
    # rn.load_state_dict(t.load(MODEL_SAVE_PATH))
    # rn = rn.cuda()

    # mse = MSELoss().cuda()
    # rn.eval()

    # 可视化中间层权重
    # states = rn.state_dict()
    # visualize_filters(states['Embed.layer2.0.weight'][:,],shape=(8,8))
    # a = np.random.randint(0,256,(3,3))
    # plt.figure(figsize=(1,1))
    # plt.axis("off")
    # plt.imshow(a, cmap="gray")
    # plt.show()

    TEST_EPISODE = 20
    entro = nn.CrossEntropyLoss().cuda()
    with no_grad():
        embed.eval()
        same_class_train = []
        same_class_test = []
        dif_class_train = []
        dif_class_test = []
        acc_total = 0.
        loss_total = 0.
        for i in range(TEST_EPISODE):
            print(i)
            sample_classes = rd.sample(TRAIN_CLASSES, n)
            train_dataset = FewShotRNDataset(PATH, N)
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

            support_embed = support_embed.view(n, k, -1).repeat(test_size, 1, 1, 1)
            test_embed = test_embed.view(test_size, -1).repeat(n * k, 1, 1).transpose(0, 1).view(test_size, n, k, -1)

            out = F.softmax(t.abs(support_embed - test_embed).sum(dim=2).sum(dim=2).neg(), dim=1)
            loss = entro(out, labels).item()

            acc = (t.argmax(out, dim=1) == labels).sum().item() / labels.size(0)

            acc_total += acc
            loss_total += loss
            print("acc:", acc)
            print("loss:", loss)

            same_train_dis = 0.
            for i in range(0,n):
                same_train_support = samples[i*k:(i+1)*k]
                same_train_query = queries[i*qk:(i+1)*qk]
                same_train_dis += outer_class_divergence(same_train_support,same_train_query)

            dif_train_dis = 0.
            count = 0
            for i in range(0,n-1):
                for j in range(i+1,n):
                    count += 1
                    dif_train_support = samples[i*k:(i+1)*k]
                    dif_train_query = queries[j*qk:(j+1)*qk]
                    dif_train_dis += outer_class_divergence(dif_train_support,dif_train_query)

            same_test_dis = 0.
            for i in range(0,n):
                same_test_support = supports[i*k:(i+1)*k]
                same_test_query = tests[i*qk:(i+1)*qk]
                same_test_dis += outer_class_divergence(same_test_support,same_test_query)

            dif_test_dis = 0.
            for i in range(0,n-1):
                for j in range(i+1,n):
                    dif_test_support = supports[i*k:(i+1)*k]
                    dif_test_query = tests[j*qk:(j+1)*qk]
                    dif_test_dis += outer_class_divergence(dif_test_support,dif_test_query)

            same_class_train.append(same_train_dis.item()/n)
            dif_class_train.append(dif_train_dis.item()/count)
            same_class_test.append(same_test_dis.item()/n)
            dif_class_test.append(dif_test_dis.item()/count)


        # div_hist((train_inners,train_outer,test_inners,test_outers,),[str(i+1) for i in range(TEST_EPISODE)],
        #          ("train inner divergence",'train outer divergence',"test inner divergence",'test outer divergence'),
        #          'divergence', 'Embed Module Output element-wise divergence', format="%d", ylim=(0,3000), xlabel="Stage")

        div_hist((np.mean(same_class_train), np.mean(dif_class_train), np.mean(same_class_test), np.mean(dif_class_test)),
                 [''],
                 ("same class in training", 'different class in training', "same class in testing", 'different class in testing'),
                 'mean divergence', 'Embed Module Output element-wise mean divergence for %d stages\nacc=%.3f' % (TEST_EPISODE,acc_total/TEST_EPISODE),
                 format="%d", ylim=(0,60000))
        print("avg acc:",acc_total/TEST_EPISODE)
        print("avg loss:",loss_total/TEST_EPISODE)

        #


