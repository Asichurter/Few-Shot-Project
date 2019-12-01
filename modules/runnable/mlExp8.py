# n-gram实验

import numpy as np
import random as rd
import os
from multiprocessing import Process, Queue, Value, Lock
import time
import warnings

from modules.utils.nGram import FrqNGram, KNN
from modules.utils.dlUtils import cal_beliefe_interval

path = 'D:/peimages/PEs/drebin_10/train/'

k = 5
qk = 5
n = 10
N = 10

NG = 3
L = 65536

iterations = 50

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
        datas[i] = None     # 试图回收内存
    print(num, 'exit', end=' | ')

def main():
    train_samples = []
    test_samples = []

    def clear_all():
        train_samples.clear()
        test_samples.clear()

    class_names = os.listdir(path)
    class_num = len(class_names)

    acc_his = []
    time_stamp = time.time()
    for i in range(iterations):
        print(i, 'th iteration')
        # 清空数据仓
        clear_all()

        rd.seed(time.time()%7355065)

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
                try:
                    p = Process(target=extract_ngram_multiprocess,
                                args=(path + class_names[c_index_each] + '/' + class_items[j], NG, L, using_queue, j))
                except IndexError:
                    print('class names length',len(class_names),'index:',c_index_each)
                    print('class items length',len(class_items),'index:',j)
                    while True:
                        command = input('command > ')
                        if command != 'quit':
                            print_template = 'print(%s)'
                            exec(print_template%command)
                        else:
                            assert False
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

main()
    # acc_his = [0.4666666667,
    #            0.3333333333,
    #            0.2666666666,
    #            0.4,
    #            0.4166666667,
    #            0.5,
    #            0.45,
    #            0.575,
    #            0.175, 0.4, 0.325, 0.475, 0.5, 0.25, 0.275, 0.475, 0.4, 0.3, 0.45, 0.425, 0.3, 0.375,
    #            0.4, 0.425, 0.3, 0.4, 0.375, 0.4, 0.375, 0.4, 0.325, 0.425, 0.475, 0.375, 0.275, 0.375, 0.35, 0.325,
    #            ]
    # print('average acc:', np.mean(acc_his))
