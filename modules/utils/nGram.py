from PIL import Image
import numpy as np
import warnings
from multiprocessing import Process, Value

class FrqNGram:
    def __init__(self, path, N, L):
        self.N = N
        self.L = L
        self.Counter = None

        self.extract_NGram(path)

    def extract_NGram(self, path):
        n = self.N
        counter = {}
        with open(path, "rb") as f:
            image_byte_seq = np.fromfile(f, dtype=np.uint8).tolist()
            for i in range(len(image_byte_seq)-n+1):
                self.add_window_feature(image_byte_seq, i, counter)

        self.Counter = counter

    def add_window_feature(self, seq, start, counter):
        n = self.N
        assert start < len(seq)-n+1, '起始下标超过了n-gram边界！start_index:%d seq_length:%d'%(start, seq.shape[0])

        n_gram_slice = seq[start:start+n]
        counter.setdefault(str(n_gram_slice), 0)
        counter[str(n_gram_slice)] += 1/(len(seq)-n+1)   # 归一化概率

    def fetch_topL_frequent(self):
        new_NGram = {}
        if len(self.Counter) <= self.L:
            warnings.warn("NGram总长度 %d 小于提取长度 %d !" % (len(self.Counter)), self.L)
            return
        for i in range(self.L):
            most_frqt_key = max(self.Counter, key=self.Counter.get)
            new_NGram[most_frqt_key] = self.Counter.pop(most_frqt_key)

        self.Counter = new_NGram

    def get_NGram_item_frqc(self, ngram_item):
        return self.Counter.get(ngram_item, 0)

    @staticmethod
    def dist_func(NG1, NG2):
        assert NG1.L == NG2.L and NG1.N == NG2.N, "两个NGram的N和L不匹配！"

        ng1_set = set(NG1.Counter.keys())
        ng2_set = set(NG2.Counter.keys())
        ngram_set = set.union(ng1_set, ng2_set)   # 取两者ngram的并集

        distance = 0
        for ng_item in ngram_set:
            f1 = NG1.get_NGram_item_frqc(ng_item)
            f2 = NG2.get_NGram_item_frqc(ng_item)

            assert f1+f2 > 0, "错误！频率之和为0将导致分母为0！"

            distance += ((f1-f2)/(f1+f2))**2*4

        return distance

class KNN:
    def __init__(self, datas, k=1):
        self.Datas = datas
        self.K = k

    # def multiprocess_predict(self, xs):
    #     index = Value('i', 0)
    #     process_amount = 5
    #     process_pool = []
    #     for i in range(process_amount):
    #         p = Process(target=self.predict, args=())
    #         process_pool.

    def predict(self, x, dis_metric=FrqNGram.dist_func):
        dis = []
        # 计算距离
        for item in self.Datas:
            data, label = item
            dis.append(dis_metric(data,x))

        assert len(dis) >= self.K, '样本过少，数量小于K'
        dis = np.array(dis)
        replace_token = dis.max()   # 用于标识已经取过的样本的替换符
        candidate_indexes = []      # top k 样本下标
        for i in range(self.K):
            min_index = dis.argmin()
            candidate_indexes.append(min_index)
            dis[min_index] = replace_token     # 从最小距离值开始，每找到一个距离值就将其替换为最大距离值以避免再次选中

        labels_count = {}
        # 计数
        for can_index in  candidate_indexes:
            label = self.Datas[can_index][1]
            labels_count.setdefault(label, 0)
            labels_count[label] += 1

        return max(labels_count, key=labels_count.get)





if __name__ == '__main__':
    p = 'D:/peimages/New/cluster/train/Backdoor.Win32.Agobot/Backdoor.Win32.Agobot.015.e.jpg'
    a = {'a':2, 'b':1, 'c':3}

