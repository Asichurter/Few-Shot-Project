# pixel-knn

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from modules.utils.dlUtils import cal_beliefe_interval

import random as rd
import os
from time import time
import gc
from PIL import Image

folder = 'virushare_20'
PATH = 'D:/peimages/New/%s/train/' % folder

k = 10
n = 5
qk = 10
N = 20

iterations = 500

all_classes_names = os.listdir(PATH)
all_class_num = len(all_classes_names)

train_samples = []
train_labels = []

test_samples = []
test_labels = []

acc_his = []

for i in range(iterations):
    # 每一轮开始前先清空数据仓
    train_samples = []
    train_labels = []

    test_samples = []
    test_labels = []
    gc.collect()

    rd.seed(time()%7355650)

    # 先抽取n个实验类
    sample_classes = rd.sample(all_classes_names, n)

    for l in range(len(sample_classes)):
        rd.seed(time()%7355650)

        all_insts_name = os.listdir(PATH+sample_classes[l]+'/')
        all_insts = set([ii for ii in range(N)])
        train_insts = set(rd.sample(all_insts, k))

        # 测试集的样本下标与训练集的样本下标不应该重复
        all_insts = all_insts.difference(train_insts)
        test_insts = set(rd.sample(all_insts, qk))

        # 分配数据到训练集和测试集中
        for j in range(N):
            extract_sample = np.array(Image.open(PATH+sample_classes[l]+'/'+all_insts_name[j])).reshape(-1,)
            if extract_sample is None:
                print('iter',i,'class',sample_classes[l],'sample',j,'extracting failed!')
                continue
            if j in train_insts:
                train_samples.append(extract_sample)
                train_labels.append(l)
            elif j in test_insts:
                test_samples.append(extract_sample)
                test_labels.append(l)
            else:
                continue

    knn = KNeighborsClassifier(n_neighbors=1)

    # 标准化各维度
    # 利用训练集计算出来的均值和方差来处理训练集和测试集的数据
    scaler = StandardScaler().fit(train_samples)
    train_samples = scaler.transform(train_samples)
    test_samples = scaler.transform(test_samples)

    knn.fit(train_samples, train_labels)

    predicts = knn.predict(np.array(test_samples))
    crt_cnt = (np.array(test_labels)==predicts).sum()

    acc = crt_cnt/len(test_samples)
    acc_his.append(acc)

    print(i, 'acc', acc)

print('average acc', np.mean(acc_his))
print('95%% belief interval: %f' % cal_beliefe_interval(acc_his))











