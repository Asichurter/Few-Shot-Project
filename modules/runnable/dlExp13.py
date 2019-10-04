# 本实验用于测试Residual-Proto的向量嵌入后的降维分析

import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
import numpy as np

import time
from sklearn.manifold import MDS
from sklearn.manifold.t_sne import TSNE

from modules.model.ResidualNet import ResidualNet
from modules.model.PrototypicalNet import ProtoNet
from modules.utils.dlUtils import RN_labelize
from modules.utils.datasets import FewShotRNDataset, FewShotFileDataset, get_RN_sampler

folder = 'cluster'
model = 'ProtoNet'
version = 40

VALIDATE_PATH = "D:/peimages/New/%s/test.npy"%folder
# VALIDATE_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/validate/"
# MODEL_LOAD_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/"+"Residual_last_epoch_model_5shot_5way_v9.0.h5"
MODEL_LOAD_PATH = "D:/peimages/New/%s/models/"%folder+"%s_best_acc_model_5shot_5way_v%d.0.h5"%(model, version)

input_size = 256

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20
# 学习率
lr = 1e-3

seed = time.time()%100000

test_classes = 50
TEST_CLASSES = [i for i in range(test_classes)]
VALIDATE_EPISODE = 1


def proto_embed_reduction(support, query, metric="MDS", proto=None):
    assert support.size(2)==query.size(2), 'suppor和query的向量维度不一致！'
    d = support.size(2)
    support = support.view(-1,d)
    query = query.view(-1,d)

    support_size = support.size(0)
    proto_size = proto.size(0) if proto is not None else 0

    if proto is not None:
        merge = t.cat((support, query, proto), dim=0).cpu().detach().numpy()
    else:
        merge = t.cat((support, query), dim=0).cpu().detach().numpy()
    # support_np = support.cpu().detach().numpy()
    # support_labels = np.array([[i for j in range(k)] for i in range(n)]).reshape(-1)

    if metric == "MDS":
        reducer = MDS(n_components=2, verbose=True)
    elif metric == "tSNE":
        reducer = TSNE(n_components=2)
    else:
        assert False, "无效的metric"
    merge_transformed = reducer.fit_transform(merge)

    support = merge_transformed[:support_size]
    query = merge_transformed[support_size:] if proto_size==0 else merge_transformed[support_size:-proto_size]
    if proto_size != 0:
        proto = merge_transformed[-proto_size:]

    support_center = support.reshape((n, k, -1)).mean(axis=1)
    support_center = support_center.reshape((n, -1))

    if proto_size == 0:
        return support, query, support_center
    else:
        return support, query, support_center,proto

def validate(model, loss, classes, seed=0):
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
            test_relations = net(supports.view(n,k,1,256,256), tests.view(n*qk,1,256,256))

            test_loss += loss(test_relations, test_labels).item()
            test_acc += (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)
            # test_acc += (t.argmax(test_relations, dim=1) == t.argmax(test_labels,dim=1)).sum().item() / test_labels.size(0)

        return test_acc/VALIDATE_EPISODE,test_loss/VALIDATE_EPISODE


dataset = FewShotFileDataset(VALIDATE_PATH, N, test_classes)

# net = ResidualNet(input_size=input_size,n=n,k=k,qk=qk,metric='Proto',block_num=5)
net = ProtoNet()
net.load_state_dict(t.load(MODEL_LOAD_PATH))
net = net.cuda()

# sample_classes = [48, 49, 50, 51, 52]
sample_classes = rd.sample(TEST_CLASSES, n)
sample_sampler,query_sampler = get_RN_sampler(sample_classes, k, qk, N, seed=seed)

train_sample_dataloader = DataLoader(dataset, batch_size=n*k, sampler=sample_sampler)
train_query_dataloader = DataLoader(dataset, batch_size=qk*n, sampler=query_sampler)

samples,sample_labels = train_sample_dataloader.__iter__().next()
queries,query_labels = train_query_dataloader.__iter__().next()

samples = samples.cuda()
sample_labels = sample_labels.cuda()
queries = queries.cuda()
query_labels = query_labels.cuda()

sample_labels = sample_labels.cpu().numpy()[::k]
query_labels = query_labels.cpu().numpy()

samples,queries = net(samples.view(n,k,1,256,256), queries.view(n*qk,1,256,256), save_embed=True, save_proto=False)

samples_trans,queries_trans,center_trans = proto_embed_reduction(samples,queries,metric="MDS")

acc,_loss = validate(net, nn.NLLLoss().cuda(), sample_classes, seed)

colors = ["red","blue","orange","green","purple"]
plt.title("Acc = %.4f"%acc)
plt.axis("off")
for i in range(n):
    plt.scatter([center_trans[i][0]], center_trans[i][1], marker="x", color=colors[i])
    # plt.scatter([proto[i][0]], proto[i][1], marker="*", color=colors[i])
    plt.scatter([x[0] for x in samples_trans[i*k:(i+1)*k]],[x[1] for x in samples_trans[i*k:(i+1)*k]],
             marker="o", color=colors[i])
    plt.scatter([x[0] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
             [x[1] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
             marker="^", color=colors[i])
plt.show()
