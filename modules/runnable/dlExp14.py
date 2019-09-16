# 本实验利用训练好的模型将验证集可视化

import matplotlib.pyplot as plt
import torch as t
import numpy as np
from sklearn.manifold import t_sne
import  torchvision.transforms as T
import PIL.Image as Image
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from modules.model.PrototypicalNet import ProtoNet
from modules.model.datasets import FewShotRNDataset,RNSamlper,get_RN_sampler

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20

MODEL_PATH = ""
DATA_PATH = ""
BATCH_LENGTH = int(len(os.listdir(DATA_PATH))/n)

dataset = FewShotRNDataset(DATA_PATH, N, rotate=False)
datas = []
reducer = t_sne.TSNE(n_components=2)

# def transform(path, transforms=T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])):
#     img = Image.open(path)
#     return transforms(img)
#
#
# datas = []
#
# for c in os.listdir(DATA_PATH):
#     class_datas = []
#     class_path = DATA_PATH+c+"/"
#     for item in os.listdir(class_path):
#         item = transform(class_path+item)
#         class_datas.append(item)
#     datas.append(class_datas)

net = ProtoNet(k=k, n=n, qk=qk).cuda()
for i in range(BATCH_LENGTH):
    classes = [i*n+j for j in range(n)]
    support_sampler, test_sampler = get_RN_sampler(classes, k, qk, N)
    test_support_dataloader = DataLoader(dataset, batch_size=n * k,
                                         sampler=support_sampler)
    test_test_dataloader = DataLoader(dataset, batch_size=qk * n,
                                      sampler=test_sampler)

    supports, support_labels = test_support_dataloader.__iter__().next()
    tests, test_labels = test_test_dataloader.__iter__().next()

    supports = supports.cuda()
    tests = tests.cuda()

    supports,tests = net(supports, tests, save_embed=True)
    batch_datas = t.cat((supports, tests), dim=1).view(n, qk+k, -1).cpu().numpy().tolist()

    datas += batch_datas

reduced_datas = reducer.fit_transform(np.array(datas))

colors = ['']

plt.title("Variance not in loss\nAcc = %.4f"%acc)
plt.axis("off")
for i in range(n):
    plt.scatter([center_trans[i][0]], center_trans[i][1], marker="x", color=colors[i])
    plt.scatter([x[0] for x in samples_trans[i*k:(i+1)*k]],[x[1] for x in samples_trans[i*k:(i+1)*k]],
             marker="o", color=colors[i])
    plt.scatter([x[0] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
             [x[1] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
             marker="^", color=colors[i])
plt.show()








