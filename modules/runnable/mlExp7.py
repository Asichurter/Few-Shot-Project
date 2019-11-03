# 测试Gist特征+kNN的小样本性能

from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random as rd
import os
import matplotlib.pyplot as plt

from modules.utils.gist import getGists

path = 'D:/peimages/New/test/train/'

k = 5
n = 10
N = 20
qk = N-k

iterations = 1000

train_samples = []
train_labels = []

test_samples = []
test_labels = []

def bar_frequency(data, title, bins=10, color="blue", bar_width=0.2, precision=2):
    bin_interval = 1/bins
    x = np.arange(bins)*bar_width
    x_label = (x-bar_width/2)
    x_label = np.append(x_label, x_label[-1]+bar_width)
    x_ticks = [round(i*bin_interval, precision) for i in range(bins+1)]
    # print(x, x_label, x_ticks, sep='\n')
    data = np.floor(np.array(data)/bin_interval)
    frequency = [0]*bins
    for i in data:
        try:
            frequency[int(i)] += 1/len(data)
        except IndexError:
            if int(i)==10:
                frequency[9] += 1/len(data)
    plt.title(title)
    plt.bar(x, frequency, alpha=0.5, width=bar_width, color=color, edgecolor='black', label="frequency", lw=3)
    plt.xticks(x_label, x_ticks)
    plt.legend()
    plt.show()

def get_random_indexes(total, sample):
    return rd.sample([i for i in range(total)], sample)

def Gist(imgs):
    # 8*8的分块，45°方向，3、5、7、9的卷积核尺寸
    return getGists(imgs, blocks=8, direction=4, scale=[3,5,7,9])

def clear_all():
    train_samples.clear()
    train_labels.clear()
    test_samples.clear()
    test_labels.clear()

class_names = os.listdir(path)
class_num = len(class_names)

acc_his = []
for i in range(iterations):
    print(i)
    # 清空数据仓
    clear_all()

    # 选中的N个类
    class_indexes = get_random_indexes(class_num, n)

    for l,c_index_each in enumerate(class_indexes):
        # 选中K个样本
        train_item_indexes = get_random_indexes(N, k)
        class_items = os.listdir(path+class_names[c_index_each]+'/')
        train_imgs = []
        test_imgs = []
        for j in range(N):
            img = Image.open(path+class_names[c_index_each]+'/'+class_items[j])
            if j in train_item_indexes:
                train_imgs.append(img)
            else:
                test_imgs.append(img)

        train_samples += Gist(train_imgs)
        train_labels += [l]*k

        test_samples += Gist(test_imgs)
        test_labels += [l]*qk

        assert len(train_samples)==len(train_labels) and len(test_samples)==len(test_labels)

    # 根据Gist特征分类论文中的描述，k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(np.array(train_samples), np.array(train_labels))

    acc = (knn.predict(test_samples)==test_labels).sum()/len(test_labels)
    print('acc:',acc)
    acc_his.append(acc)

print('average acc: ', np.mean(acc_his))
bar_frequency(acc_his, title='%d-shot %d-way with Gist Feature'%(k,n))



