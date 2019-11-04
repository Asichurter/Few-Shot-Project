# 测试Gist特征+kNN的小样本性能

from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random as rd
import os
import matplotlib.pyplot as plt
from time import time

from multiprocessing import Process, Queue
import warnings

from modules.utils.gist import getGists

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




def cal_gist_acc(iters, q, num):
    path = 'D:/peimages/New/test/train/'

    k = 10
    n = 20
    N = 20
    qk = N - k

    train_samples = []
    train_labels = []

    test_samples = []
    test_labels = []

    def clear_all():
        train_samples.clear()
        train_labels.clear()
        test_samples.clear()
        test_labels.clear()

    class_names = os.listdir(path)
    class_num = len(class_names)

    last_stamp = time()

    for i in range(iters):
        # print(i, num)


        # 清空数据仓
        clear_all()

        # 选中的N个类
        class_indexes = get_random_indexes(class_num, n)

        for l, c_index_each in enumerate(class_indexes):
            # 选中K个样本
            train_test_mix_indexes = set(get_random_indexes(N, k+qk))
            train_item_indexes = set(rd.sample(train_test_mix_indexes, k))
            test_item_indexes = train_test_mix_indexes.difference(train_item_indexes)
            class_items = os.listdir(path + class_names[c_index_each] + '/')
            train_imgs = []
            test_imgs = []
            for j in range(N):
                img = Image.open(path + class_names[c_index_each] + '/' + class_items[j])
                if j in train_item_indexes:
                    train_imgs.append(img)
                elif j in test_item_indexes:
                    test_imgs.append(img)

            train_samples += Gist(train_imgs)
            train_labels += [l] * k

            test_samples += Gist(test_imgs)
            test_labels += [l] * qk


            assert len(train_samples) == len(train_labels) and len(test_samples) == len(test_labels)

        # 根据Gist特征分类论文中的描述，k=3
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(np.array(train_samples), np.array(train_labels))

        acc = (knn.predict(test_samples) == test_labels).sum() / len(test_labels)

        consume_time = time() - last_stamp
        last_stamp = time()

        print('process',num,',',i,'th iter, time:',consume_time,'acc:', acc)
        q.put(acc)

def consume_acc(q, num, timeout=60):
    all_acc = []
    while len(all_acc) < num:
        try:
            acc = q.get(timeout=timeout)
            # print('acc got,', acc)
        except:
            assert False, '获取acc的进程的get超时！'
        all_acc.append(acc)

    if q.qsize() != 0:
        warnings.warn('取到足够数量的acc时，队列中还有')

    print('average acc:', np.mean(all_acc))
    bar_frequency(all_acc, title='Gist+kNN Accuracy')


if __name__ == '__main__':

    iterations = 200
    process_amount = 5
    iter_per_process = int(iterations/process_amount)

    acc_queue = Queue()
    process_pool = []

    for i in range(process_amount):
        p = Process(target=cal_gist_acc, args=(iter_per_process, acc_queue, i))
        process_pool.append(p)
        p.start()

    csm_p = Process(target=consume_acc, args=(acc_queue,iterations))
    csm_p.start()

    csm_p.join()

    for p in process_pool:
        p.join()

    # if acc_queue.qsize() != iterations:
    #     warnings.warn('acc实际总数%d小于预期总数%d'%(acc_queue.qsize(), iterations))
    #
    # acc_his = []
    # while acc_queue.qsize() != 0:
    #     acc_his.append(acc_queue.get())
    #
    # print('average acc:', np.mean(acc_his))



