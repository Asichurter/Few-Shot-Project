# 本实验为了验证ProtoNet的非单峰分布

import os
from time import time
import random as rd
import torch as t
import numpy as np
import matplotlib.pyplot as plt

from modules.model.PrototypicalNet import ProtoNet
from modules.model.ChannelNet import ChannelNet
from modules.utils.datasets import FewShotFileDataset, SingleClassSampler
from torch.utils.data import DataLoader

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

def cal_KL_div(fx, gx):
    return np.sum(np.exp(fx)*(fx-gx))

k = 10
n = 5
N = 20

version_dict = {'cluster':{5:{'ChannelNet':22, 'ProtoNet':35}, 10:{'ChannelNet':25, 'ProtoNet':46}},
                'test':{5:{'ChannelNet':33, 'ProtoNet':53}, 10:{'ChannelNet':36, 'ProtoNet': 55}},
                'virushare_20':{5:{'ChannelNet':27,'ProtoNet':48}, 10:{'ChannelNet':30, 'ProtoNet':52}}}

folder = 'virushare_20'
cropSize = 256
mode = 'best_acc'
modelType = 'ChannelNet'
version = version_dict[folder][k][modelType]

print('version',version)

exp_epoches = 100
mean_iterations = 1000
plot = False

magicNum = 7355608

MODEL_LOAD_PATH = "D:/peimages/New/%s/models/"%folder\
                  +"%s_%s_model_%dshot_%dway_v%d.0.h5"%(modelType,mode,k,n,version)
DATASET_PATH = "D:/peimages/New/%s/test.npy"%folder
CLASS_NUM = len(os.listdir("D:/peimages/New/%s/test/"%folder))

dataset = FewShotFileDataset(DATASET_PATH, N, CLASS_NUM, cropSize)

rd.seed(time()%magicNum)
clsIndex = rd.randint(0,CLASS_NUM-1)

if modelType == 'ProtoNet':
    net = ProtoNet()
if modelType == 'ChannelNet':
    net = ChannelNet(k)

states = t.load(MODEL_LOAD_PATH)
net.load_state_dict(states)
net = net.cuda()
net.eval()

kls = []

for epoch in range(exp_epoches):
    print('epoch',epoch, end=' ')
    # -------------------- 数据嵌入原分布 ------------------------
    # print('geting original embeddings...')
    originalSampler = SingleClassSampler(clsIndex, N, N)
    originalLoader = DataLoader(dataset, batch_size=N, sampler=originalSampler)

    originalData,_l = originalLoader.__iter__().next()
    originalData = originalData.cuda()

    originalEmbed = net.embed_data(originalData, return_mean=False).squeeze().tolist()
    # originalPCA = PCA(n_components=1)
    # originalEmbed = originalPCA.fit_transform(originalEmbed).reshape(-1,1)
    # originalEmbed = [x[0] for x in originalEmbed_]   # 取第0维度上的值
    # ---------------------------------------------------------

    # -------------------- 嵌入后的均值分布 ------------------------
    meanProtos = []
    for i in range(mean_iterations):
        # print('iter',i)
        protoSampler = SingleClassSampler(clsIndex, N, k)
        protoLoader = DataLoader(dataset, batch_size=N, sampler=protoSampler)

        protoData,_l = protoLoader.__iter__().next()
        protoData = protoData.cuda()

        proto = net.embed_data(protoData, return_mean=True).tolist()
        meanProtos.append(proto)

    # protoPCA = PCA(n_components=1)
    # meanProtos = protoPCA.fit_transform(meanProtos)
    # ---------------------------------------------------------

    pca = PCA(n_components=1)
    pca.fit(originalEmbed+meanProtos)

    originalEmbed = pca.fit_transform(originalEmbed)
    meanProtos = pca.fit_transform(meanProtos)

    # print('fitting original embeddings...')
    oriKD = KernelDensity()
    oriKD.fit(originalEmbed)

    protoKD = KernelDensity()
    protoKD.fit(meanProtos)

    # x = np.linspace(int(meanProtos.min())-3, int(meanProtos.max())+3, 1000).reshape(-1,1)
    x = np.linspace(int(originalEmbed.min())-3, int(originalEmbed.max())+3, 1000).reshape(-1,1)
    oriP = oriKD.score_samples(x)
    protoP = protoKD.score_samples(x)
    kl_div = cal_KL_div(oriP, protoP)
    kls.append(kl_div)
    print(kl_div)

    if plot:
        plt.figure(dpi=600)
        # plt.title('KL divergence:%.3f'%kl_div)
        plt.plot(x,oriP, color='red', label='real embedded data')
        plt.plot(x,protoP, color='blue', label='mean prototype')
        plt.ylabel('log-probability')
        plt.xlabel('embedded value')
        plt.xticks([])
        plt.legend()
        plt.show()

if not plot:
    print('mean KL div:',np.mean(kls))







