import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import random as rd
import torchvision.transforms as T
import torch as t
import numpy as np
import math
import matplotlib.pyplot as plt
import collections
from operator import itemgetter
import json

from modules.utils.datasets import FewShotPreloadDataset
from modules.utils.datasets import get_RN_sampler


# train_datas = np.array([[0,0],[3,3]])
# train_labels = np.array([2,5])
# test_datas = np.array([[2,2]])
#
# knn = KNN(n_neighbors=1)
# knn.fit(train_datas, train_labels)

# data = t.Tensor([[0.,0.],[1.,1.],[3.,3.],[4.,4.]])
# test = t.Tensor([[0.5,0.5],[2.5,2.5]])
# label = t.Tensor([[0,1,0,0],[0,0,1,0]])
#
# new_test = t.Tensor([])
# for tensor in t.unbind(test):
#     new_one = tensor.view(-1,2).repeat(4,1)
#     new_test = t.cat((new_test, new_one), dim=0)
#
# test = new_test
# data = data.repeat((2,1))
#
# inputs = (data-test).view(-1,2)
#
# x1 = t.ones((2,4), requires_grad=True)
# x2 = t.ones((4,1), requires_grad=True)
#
# result = t.mm(t.mm(inputs,x1),x2)#.view(-1,4)
# label = label.view(-1)
#
# loss = t.nn.MSELoss()
# loss_val = loss(result, label)
#
# loss_val.backward()
#
# query = t.Tensor([[[1,2],[1,2]],[[3,4],[3,4]]])
# support = t.Tensor([[[1,1],[2,2]],[[1,1],[2,2]]])
# out = F.log_softmax(t.sum((query-support)**2, dim=2),dim=1)
# labels = t.LongTensor([0,1])
#
# loss = nn.CrossEntropyLoss()
#
# out = t.Tensor([[1.3,10,0.4],[0.1,0.2,0.9]])
# label=t.LongTensor([1,2])
# loss_val = loss(out, label)
#
# dat = t.Tensor([[1,1],[2,3]])
# dat2 = t.Tensor([[1,3],[4,2]])
#
# dat_length = t.sqrt((dat**2).sum(dim=1))
# dat2_length = t.sqrt((dat2**2).sum(dim=1))
# dot = t.mul(dat,dat2).sum(dim=1)


# a = t.Tensor([[[1,2],[1,1]],[[0,2],[1,3]],[[0,1],[1,2]]])
# b = a.sum(dim=2)
# c = a.sum(dim=2).sum(dim=1).unsqueeze(dim=1).repeat(1,2)

# a = t.randn((5,1,4,14))
# # [qk,n,d]
# # 距离
# dis = t.randn((15,5,14))
#
# conv1 = nn.Conv2d(1,32, kernel_size=(4,1), stride=(1,1), padding=(1,0))
# conv2 = nn.Conv2d(32,32, kernel_size=(4,1), stride=(1,1), padding=(2,0))
# conv3 = nn.Conv2d(32,1, kernel_size=(4,1), stride=(4,1), padding=(0,0))
#
# attention = conv3(conv2(conv1(a))).squeeze().repeat(15,1,1)
#
# attented = (attention*dis).sum(dim=2)

# cs = [i for i in range(1,101)]
# val = []
# f = 10
# for c in cs:
#     a = [-c]*f
#     a[0] = f
#     a = t.Tensor(a)
#     val.append(t.softmax(a, dim=0)[0].item())
#
# plt.plot(cs, val)
# plt.show()

# d = {1:1, 2:4, 3:9, 4:16}
# np.save('D:/Few-Shot-Project/data/dict.npy', d)

# d = np.load('D:/Few-Shot-Project/data/clusters_0.5eps_20minnum.npy', allow_pickle=True)
# d = d.item()
# for key,val in d.items():
#     print(key, 'cluster num: %d'%len(val))
# print(len(d))]

# a = [1, 'a', ['1', 2], t.Tensor([[1,2],[3,4]])]
# np.save('test.npy', a)
# a = np.load('test.npy', allow_pickle=True)

# def variant_length_collect_fn(datas):
#     pass
#
# classes = [i for i in range(100)]
# sample_classes = rd.sample(classes, 5)
# dataset = FewShotPreloadDataset('D:/peimages/New/cluster_fix_width/train/', square=False,
#                                 transform=T.Compose([T.ToTensor(), T.Normalize([0.4077583], [0.09825569])]))
#
# sample_sampler, query_sampler = get_RN_sampler(sample_classes, 5, 15, 20)
#
# train_sample_dataloader = DataLoader(dataset, batch_size=5*5, sampler=sample_sampler)
# train_query_dataloader = DataLoader(dataset, batch_size=15 * 5, sampler=query_sampler)
#
# samples, sample_labels = train_sample_dataloader.__iter__().next()
# queries, query_labels = train_query_dataloader.__iter__().next()
#
# samples = samples.cuda()
# sample_labels = sample_labels.cuda()
# queries = queries.cuda()
# query_labels = query_labels.cuda()
#
# net = nn.Sequential(
#     nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0, bias=False),
#     nn.BatchNorm2d(16),
#     nn.ReLU(inplace=True),
#     )
# net = net.cuda()
# support_outs = net(samples)
# query_outs = net(queries)

# a = np.load('D:/peimages/New/cluster/test.npy', allow_pickle=True)

# def normalize(tensors):
#     dim = len(tensors.size())-1
#     length = tensors.size(dim)
#     repeat_index = [1 for i in range(dim)]
#     repeat_index.append(length)
#     norm = tensors.norm(dim=dim).unsqueeze(dim=dim).repeat(repeat_index)
#
#     return tensors/norm
#

# def fx(x):
#     return math.exp(1)/(math.exp(1)+(x-1)*math.exp(-1))
#
# x = [i+2 for i in range(99)]
# plt.title()
# plt.xlabel('|C|')
# plt.ylabel('probability upper bound')
# plt.plot(x, [fx(xx) for xx in x])
# plt.show()

# dic = collections.OrderedDict()#{}#
#
# dic['k1'] = 2
# dic['k2'] = 3
# dic['k3'] = 1
# dic = sorted(dic.items(), key=itemgetter(1), reverse=True)

# a = {'1':'A', '2':'B', '3':'C'}
# with open('D:/peimages/New/cluster/gist/temp/1.json', 'w') as fp:
#     json.dump(a, fp)
#
# with open('D:/peimages/New/cluster/gist/temp/1.json', 'r') as fp:
#     a = json.load(fp)
#     print(a)

# import random
# import time
# import multiprocessing
#
#
# def worker(name, q, num):
#     t = 0
#     for i in range(10):
#         # print(name + " " + str(i))
#         x = random.randint(1, 3)
#         t += x
#         time.sleep(x * 0.1)
#     q.put(t)
#     print(num, 'done')
#
# def main():
#     q = multiprocessing.Queue()
#     jobs = []
#     for i in range(10):
#         print('create', i)
#         p = multiprocessing.Process(target=worker, args=(str(i), q, i))
#         jobs.append(p)
#         p.start()
#
#     for i,p in enumerate(jobs):
#         print('waiting for', i)
#         p.join()
#         print('process',i,'exit')
#
#     results = [q.get() for j in jobs]
#     print(results)
#
# if __name__ == '__main__':
#     main()

N = 20
k = 32
d = 64

c = t.randn((1,d)).repeat((N,1))
e = t.randn((N,d))
b = nn.Bilinear(d, d, k, bias=False)
layer = nn.Sequential(
    nn.ReLU(inplace=True),
    nn.Linear(k, 1, bias=True),
    nn.Sigmoid()
)

r = layer(b(c,e))





































