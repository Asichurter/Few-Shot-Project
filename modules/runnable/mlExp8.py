# n-gram实验

import numpy as np
import random as rd
import os
from multiprocessing import Process, Queue, Value, Lock
import time
import warnings

from modules.utils.nGram import FrqNGram, KNN

path = 'D:/peimages/PEs/cluster/train/'

k = 10
n = 5
N = 20
qk = 5

NG = 3
L = 1024

iterations = 2

def debug():
    while True:
        command = input('command >>')
        if command != 'quit':
            exec(command)
        else:
            return

def get_random_indexes(total, sample):
    return rd.sample([i for i in range(total)], sample)

def extract_ngram_multiprocess(path, N, L, queue, num):
    queue.put(FrqNGram(path,N,L))

def predict(knn, datas, crt, num):
    for i,data in enumerate(datas):
        # print(num,i)
        ngram,label = data
        prd_label = knn.predict(ngram)
        if prd_label==label:
            crt.value += 1
    print(num, 'exit', end=' | ')

def main():
    train_samples = []
    test_samples = []

    def clear_all():
        train_samples.clear()
        test_samples.clear()

    class_names = os.listdir(path)
    class_num = len(class_names)

    acc_his = [0.87, 0.83, 0.77, 0.46, 0.72, 0.46, 0.7, 0.74, 0.71, 0.65, 0.65, 0.61, 0.89, 0.63, 0.71, 0.6, 0.8, 0.75, 0.79, 0.56, 0.47, 0.82]#[0.49, 0.34, 0.3, 0.42, 0.28, 0.27, 0.29, 0.36, 0.24, 0.34, 0.44, 0.33, 0.46, 0.31, 0.43, 0.36, 0.36]
    time_stamp = time.time()
    for i in range(iterations):
        print(i, 'th iteration')
        # 清空数据仓
        clear_all()

        # 选中的N个类
        class_indexes = get_random_indexes(class_num, n)
        print('extracting...', end=' ')
        section_time_stamp = time.time()
        for l, c_index_each in enumerate(class_indexes):
            print(l, end=' ')
            # 选中K个样本
            train_test_mix_indexes = set(get_random_indexes(N, k + qk))
            train_item_indexes = set(rd.sample(train_test_mix_indexes, k))
            test_item_indexes = train_test_mix_indexes.difference(train_item_indexes)

            class_items = os.listdir(path + class_names[c_index_each] + '/')

            process_pool = []
            ngram_train_queue = Queue()  # []
            ngram_test_queue = Queue()  # []
            # train_lock = Lock()
            # test_lock = Lock()

            # N个样本使用N个进程
            for j in range(N):
                # print(i, l, j, 'process')
                if j in train_item_indexes:
                    using_queue = ngram_train_queue
                elif j in test_item_indexes:
                    using_queue = ngram_test_queue
                else:
                    continue
                p = Process(target=extract_ngram_multiprocess,
                            args=(path + class_names[c_index_each] + '/' + class_items[j], NG, L, using_queue, j))
                process_pool.append(p)
                p.start()

            for c in range(k):
                train_samples.append([ngram_train_queue.get(), l])
            for c in range(qk):
                test_samples.append([ngram_test_queue.get(), l])
            for ii, p in enumerate(process_pool):
                # print('waiting for',i)
                p.join()

        if len(train_samples) != n * k:
            warnings.warn('训练样本数量不足！预期数量：%d 实际数量：%d' % (n * k, len(train_samples)))
        if len(test_samples) != n * qk:
            warnings.warn('测试样本数量不足！预期数量：%d 实际数量：%d' % (n * qk, len(test_samples)))

        knn = KNN(train_samples, k=1)
        # while True:
        #     command = input('Command >>')
        #     if command != 'quit':
        #         exec(command)
        #     else:
        #         break
        crt_count = Value('i', 0, lock=Lock())
        pool = []

        process_amount = 5
        each_process_amount = int(len(test_samples) / process_amount)

        print('time: %.2f'%(time.time()-section_time_stamp))
        section_time_stamp = time.time()

        print('predicting...', end=' ')
        for ii in range(process_amount):
            p = Process(target=predict, args=(
            knn, test_samples[ii * each_process_amount:(ii + 1) * each_process_amount], crt_count, ii))
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

        print('time: %.2f'%(time.time()-section_time_stamp))

        print('time: %.2f' % (time.time() - time_stamp))
        time_stamp = time.time()

    print('average acc: ', np.mean(acc_his))

if __name__ == '__main__':
    main()
    # acc_his = [0.87, 0.83, 0.77, 0.46, 0.72, 0.46, 0.7, 0.74, 0.71, 0.65, 0.65, 0.61, 0.89, 0.63, 0.71, 0.6, 0.8, 0.75,
    #            0.79, 0.56, 0.47, 0.82]
    # print(np.mean(acc_his))
