import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import os

from modules.utils.extract import extract_infos

def extract_class_name(name):
    return '.'.join(name.split('.')[:3])

def filter_by_sub_classname(base, class_name, pca=None, normal=False):
    datas = []
    candidates = os.listdir(base)
    candidates = list(filter(lambda x: extract_class_name(x)==class_name, candidates))
    for i,can in enumerate(candidates):
        print('%d/%d'%(i+1,len(candidates)))
        data = extract_infos(base+can)
        if data is not None:
            datas.append(data)

    datas = np.array(datas)
    if normal:
        datas = normalize(datas)

    if pca is not None:
        pca_reducer = PCA(n_components=pca)
        datas = pca_reducer.fit_transform(datas)

    return datas,candidates

def DBscan_cluster(datas, base_num, name=None, plot=False, fig_save=None):
    name = name if name is not None else ''
    dbscan = DBSCAN(eps=0.5, min_samples=base_num)
    labels = dbscan.fit_predict(datas)

    unique_labels = set(labels)

    if plot:
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        if datas.shape[1] != 2:
            pca = PCA(n_components=2)
            datas = pca.fit_transform(datas)
        plt.figure(figsize=(15, 12))
        for k, col in zip(unique_labels, colors):
            marker_size = 4
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                marker_size = 1

            class_member_mask = (labels == k)

            xy = datas[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', color=tuple(col),
                     label='cluster%d'%(k+1) if k!=-1 else 'noise', markersize=marker_size)
        plt.title('%s clsuter num: %d'%(name, len(unique_labels)))
        plt.axis('off')
        plt.legend()
        if fig_save is not None:
            plt.savefig(fig_save)
        plt.show()

    return labels

def normalize(datas):
    d = datas.shape[1]
    l = datas.shape[0]
    mean = np.mean(datas, axis=0).reshape(1,d).repeat(l, axis=0)
    std = np.std(datas, axis=0).reshape(1,d).repeat(l, axis=0)
    return np.nan_to_num((datas-mean)/std)

def count_class_num(base):
    files = os.listdir(base)
    files_prefix = list(map(extract_class_name, files))
    files_prefix = set(files_prefix)

    # 初始化计数字典
    count = {}
    for prefix in files_prefix:
        count[prefix] = 0

    # 遍历数据集计数
    for each in files:
        count[extract_class_name(each)] += 1

    return count

def get_data_clusters(base, base_num, cluster_func=DBscan_cluster):
    clusters = {}
    sub_index = 0
    for c in os.listdir(base):
        count = count_class_num(base+c+'/')
        print(count)
        for sub_c,num in count.items():
            print(sub_index,sub_c)
            sub_index += 1
            sub_cluster = {}
            if num < base_num:
                continue
            try:
                c_datas,file_names = filter_by_sub_classname(base+c+'/', sub_c, pca=2, normal=True)
            except:
                continue
            labels = cluster_func(c_datas, base_num,
                                  fig_save='D:/Few-Shot-Project/data/cluster_plot/'+sub_c+'.png',
                                  name=sub_c, plot=True)

            # 去掉噪声只有n-1个类
            label_size = len(set(labels))-1

            if label_size == 0:
                continue

            # 初始化类簇字典
            for i in range(label_size):
                sub_cluster[i] = []

            for n,l in zip(file_names, labels):
                # 过滤噪声
                if l == -1:
                    continue

                # 按类簇将名称放入
                sub_cluster[l].append(base+c+'/'+n)
            clusters[sub_c] = sub_cluster
            print("dict length: %d"%len(clusters))
            del c_datas,file_names
    return clusters



if __name__ == '__main__':
    # base = 'D:/pe/trojan0/'
    # class_name = 'Trojan-PSW.Win32.LdPinch'
    # datas,names = filter_by_sub_classname(base, class_name, normal=True, pca=2)#np.load('D:/Few-Shot-Project/data/backdoor.Win32.PcClient.npy')#
    # cluster_labels = DBscan_cluster(datas, plot=True)
    # np.save('D:/Few-Shot-Project/data/%s.npy'%class_name, datas)

    d = get_data_clusters('D:/pe/', 20)
    np.save('D:/Few-Shot-Project/data/clusters_0.5eps_20minnum.npy', d)
    # d = np.load('D:/Few-Shot-Project/data/clusters_0.5eps_20minnum.npy', allow_pickle=True).item()


