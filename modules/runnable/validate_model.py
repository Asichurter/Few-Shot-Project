import torch as t
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad

from modules.model.RelationNet import EmbeddingNet
from modules.utils.datasets import FewShotRNDataset, get_RN_modified_sampler

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

weighs = {'conv':['Embed.layer1.0.weight','Embed.layer2.0.weight','Embed.layer3.0.weight','Embed.layer4.0.weight',
                  'Relation.layer1.0.weight','Relation.layer2.0.weight'],
          'linear':['Relation.fc1.weight','Relation.fc2.weight']}

def visualize_filters(tensors,shape):
    # fig2 = plt.figure(constrained_layout=True, figsize=(shape[0],shape[1]))
    # spec2 = gridspec.GridSpec(ncols=shape[0], nrows=shape[1], figure=fig2)
    a = np.random.randint(0,256,(3,3))
    fig = plt.figure(figsize=(8,8))
    for i,tensor in enumerate(tensors):
        # ax = fig2.add_subplot(spec2[int(i/shape[1]), i%shape[1]])
        plt.subplot(8,8,i+1)
        plt.title("%d"%i)
        plt.axis("off")
        plt.imshow(tensor.detach().numpy(), cmap="gray")
    plt.show()

def inner_class_divergence(tensors, metric="Man"):
    num = len(tensors)
    total_div = 0.
    count = 0
    for i in range(0,num-1):
        for j in range(i+1,num):
            count += 1
            if metric == "Man":
                div = t.abs(tensors[i]-tensors[j]).sum()
            if metric == "Euc":
                div = t.sqrt(((tensors[i] - tensors[j])**2).sum())
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
    width = 0.2*num  # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    for i in range(num):
        rects.append(ax.bar(x - (num*width/2) + (i+0.5)*width, inputs[i], width, label=class_labels[i]))
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
            ax.annotate(format%height,
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

    TEST_EPISODE = 10

    with no_grad():
        embed.eval()
        train_inners = []
        train_outers = []
        test_inners = []
        test_outers = []
        same_class_train = []
        same_class_test = []
        dif_class_train = []
        dif_class_test = []
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


            test_class_1_inputs = supports[10:15]
            test_class_2_inputs = supports[20:25]

            test_class_1 = embed(test_class_1_inputs)
            test_class_2 = embed(test_class_2_inputs)

            test_inner_1 = inner_class_divergence(test_class_1).item()
            test_inner_2 = inner_class_divergence(test_class_2).item()
            test_outer = outer_class_divergence(test_class_1,test_class_2).item()

            train_class_1_inputs = samples[10:15]
            train_class_2_inputs = samples[20:25]

            train_class_1 = embed(train_class_1_inputs)
            train_class_2 = embed(train_class_2_inputs)

            train_inner_1 = inner_class_divergence(train_class_1).item()
            train_inner_2 = inner_class_divergence(train_class_2).item()
            train_outer = outer_class_divergence(train_class_1,train_class_2).item()

            same_class_train_support = samples[:n]
            same_class_train_query = queries[:qk]

            test_inners.append((test_inner_1+test_inner_2)/2)
            test_outers.append(test_outer)
            train_inners.append((train_inner_1+train_inner_2)/2)
            train_outers.append(train_outer)

        # div_hist((train_inners,train_outer,test_inners,test_outers,),[str(i+1) for i in range(TEST_EPISODE)],
        #          ("train inner divergence",'train outer divergence',"test inner divergence",'test outer divergence'),
        #          'divergence', 'Embed Module Output element-wise divergence', format="%d", ylim=(0,3000), xlabel="Stage")

        div_hist((np.mean(train_inners), np.mean(train_outers), np.mean(test_inners), np.mean(test_outers)),
                 [''],
                 ("train inner divergence",'train outer divergence',"test inner divergence",'test outer divergence'),
                 'mean divergence', 'Embed Module Output element-wise mean divergence for %d stages'%TEST_EPISODE,
                 format="%d", ylim=(0,3000))

        #


