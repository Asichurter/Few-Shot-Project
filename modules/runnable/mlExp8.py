# n-gram实验

import numpy as np
import random as rd
import os
from multiprocessing import Process, Queue, Value, Lock
import time
import warnings

from modules.utils.nGram import FrqNGram, KNN

path = 'D:/peimages/PEs/cluster/train/'

k = 5
n = 5
N = 20
qk = N-k

NG = 3
L = 1024

iterations = 200

train_samples = []
test_samples = []

def debug():
    while True:
        command = input('command >>')
        if command != 'quit':
            exec(command)
        else:
            return

def get_random_indexes(total, sample):
    return rd.sample([i for i in range(total)], sample)

def clear_all():
    train_samples.clear()
    test_samples.clear()

def extract_ngram_multiprocess(path, N, L, queue, num):
    # print('thread',num,'begin')
    # time.sleep(5)
    # ngram = FrqNGram(path, N, L)
    #
    # tar_lock.acquire()
    # try:
    #     tar_list.append(ngram)
    # finally:
    #     tar_lock.release()
    queue.put(FrqNGram(path,N,L))
    # print('thread',num,'done')

def predict(knn, datas, crt, num):
    for i,data in enumerate(datas):
        # print(num,i)
        ngram,label = data
        prd_label = knn.predict(ngram)
        if prd_label==label:
            crt.value += 1
    print(num, 'exit', end=' | ')


class_names = os.listdir(path)
class_num = len(class_names)

acc_his = []
time_stamp = time.time()
for i in range(iterations):
    print(i, 'th iteration')
    # 清空数据仓
    clear_all()

    # 选中的N个类
    class_indexes = get_random_indexes(class_num, n)
    print('extracting...')
    for l, c_index_each in enumerate(class_indexes):
        # 选中K个样本
        train_item_indexes = get_random_indexes(N, k)
        class_items = os.listdir(path + class_names[c_index_each] + '/')

        process_pool = []
        ngram_train_queue = Queue()#[]
        ngram_test_queue = Queue()#[]
        # train_lock = Lock()
        # test_lock = Lock()

        # N个样本使用N个进程
        for j in range(N):
            # print(i, l, j, 'process')
            using_queue = ngram_train_queue if j in train_item_indexes else ngram_test_queue
            p = Process(target=extract_ngram_multiprocess,
                        args=(path + class_names[c_index_each] + '/' + class_items[j], NG, L, using_queue, j))
            process_pool.append(p)
            p.start()

        time.sleep(7)       # 处理间隔
        for c in range(k):
            train_samples.append([ngram_train_queue.get(), l])
        for c in range(qk):
            test_samples.append([ngram_test_queue.get(), l])
        for ii,p in enumerate(process_pool):
            # print('waiting for',i)
            p.join()

    if len(train_samples) != n*k:
        warnings.warn('训练样本数量不足！预期数量：%d 实际数量：%d'%(n*k, len(train_samples)))
    if len(test_samples) != n*qk:
        warnings.warn('测试样本数量不足！预期数量：%d 实际数量：%d'%(n*qk, len(test_samples)))

    knn = KNN(train_samples, k=1)
    crt_count = Value('i', 0, lock=Lock())
    pool = []

    process_amount = 5
    each_process_amount = int(len(test_samples)/process_amount)
    print('predicting...')
    for ii in range(process_amount):
        p = Process(target=predict, args=(knn, test_samples[ii*each_process_amount:(ii+1)*each_process_amount],crt_count,ii))
        pool.append(p)
        p.start()

    for p in pool:
        p.join()

    print()
    # for i,test_sp in enumerate(test_samples):
    #     print(i)
    #     prd_label = knn.predict(test_sp[0], dis_metric=FrqNGram.dist_func)
    #     if prd_label == test_sp[1]:
    #         crt_count += 1
    # while True:
    #     command = input('command >>')
    #     if command != 'quit':
    #         exec(command)
    #     else:
    #         break
    acc = crt_count.value / len(test_samples)
    print(i, 'acc:', acc)
    acc_his.append(acc)

    print('time: %.2f'%(time.time()-time_stamp))
    time_stamp = time.time()

print('average acc: ', np.mean(acc_his))
