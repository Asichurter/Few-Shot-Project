################################################################
# 本脚本是2019年大学生创新创业项目：小样本机器学习在恶意代码检测上的应用研究，
# 的演示程序，用于表现本项目的主要成果：ConvProtoNet，在恶意代码分类任务
# 上的作用。
#
# 本脚本将会使用一个已经经过元学习训练好的模型，接受若干个在训练中没有见过的
# 恶意代码类并输入少量样本到模型中，然后接受同样来自这几个类的样本并进行分类
# 任务。最终每个样本的分类的结果都会以控制台的形式打印出来，还会统计正确率等
# 数值。
################################################################

import torch as t
from torch.utils.data import DataLoader
import os
import time

from modules.model.ChannelNet import ChannelNet
from modules.utils.datasets import FewShotRNDataset, get_RN_sampler

k = 10
qk = 10
n = 5
N = 20
crop = 256

v = {5: {
        5: 37,
        20: 23
    },
    10:{
        5: 25,
        20: 26
    }}

model_load_path = 'D:/peimages/New/cluster/models/ChannelNet_best_acc_model_{k}shot_{n}way_v{version}.0.h5'.format(
    k=k, n=n, version=v[k][n]
)

data_path = 'D:/Few-Shot-Project/demo/data/'
dataset = FewShotRNDataset(base=data_path, n=N)

names = []
labels = []
for cls in os.listdir(data_path):
    labels.append(cls)
    for item in os.listdir(data_path + cls + '/'):
        names.append(item)

start_stamp = time.time()
net = ChannelNet(k=k, in_channels=1)
net.load_state_dict(t.load(model_load_path))
net = net.cuda()

support_sampler, query_sampler = get_RN_sampler([i for i in range(5)], k, qk, N)
support_dataloader = DataLoader(dataset, batch_size=n * k,
                                     sampler=support_sampler)
query_dataloader = DataLoader(dataset, batch_size=qk * n,
                                  sampler=query_sampler)

supports, support_labels = support_dataloader.__iter__().next()
queries, query_labels = query_dataloader.__iter__().next()

supports = supports.cuda()
support_labels = support_labels.cuda()
queries = queries.cuda()
test_labels = query_labels.cuda()

query_labels = query_labels.cuda()

outs = net(supports.view(n, k, 1, crop, crop),
           queries.view(n * qk, 1, crop, crop))

predict_labels = t.argmax(outs, dim=1).squeeze()
acc = (predict_labels==query_labels).sum().item() / queries.size(0)
end_stamp = time.time()

print('----------------------------------')
print('实验情况:')
print('本次运行为{n}分类问题，每个类仅有{k}个样本'.format(n=n, k=k))
print('测试样本共有{total}个，总耗时{t}'.format(total=qk*n, t=end_stamp-start_stamp))
print('---------------------------------')

print('分类结果:')
for i,idx in enumerate(query_sampler):
    print('#{index} 名称:{name} 预测分类: {label}'.format(
        index=i,
        name=names[idx],
        label=labels[idx//N])
    )
print('------------------------------')
print('分类正确率: {acc}'.format(acc=acc))


