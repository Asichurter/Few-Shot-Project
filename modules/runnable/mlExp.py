#传统机器学习实验1
#本实验随机抽取所有大类恶意样本各100个和相同数量的良性样本作为实验数据，良性样本来自windows系统文件夹中

#传统机器学习方法实验1.5
#本实验以大类作为实验的基础，以某些大类的数据作为训练集来测试其他大类的分类效率，即单类缺省状态
#缺省的类分别为backdoor,trojan,net-worm,virus,email

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
# from sklearn.manifold import
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix

# 实验1所使用的数据路径
folder = "ExtClassOneDefault/virus" #ExtClassEach200
TRAIN_DATA_SAVE = "D:/Few-Shot-Project/data/%s/train_data_raw.npy" % folder
TRAIN_LABEL_SAVE = "D:/Few-Shot-Project/data/%s/train_label_raw.npy"% folder
TEST_DATA_SAVE = "D:/Few-Shot-Project/data/%s/test_data_raw.npy"% folder
TEST_LABEL_SAVE = "D:/Few-Shot-Project/data/%s/test_label_raw.npy"% folder
# -----------------------------------------------------------------------------------------

# 实验2使用的数据路径
# class_path = "email"
# TRAIN_DATA_SAVE = "D:/Few-Shot-Project/data/ExtClassOneDefault/%s/train_data.npy" % class_path
# TRAIN_LABEL_SAVE = "D:/Few-Shot-Project/data/ExtClassOneDefault/%s/train_label.npy"  % class_path
# TEST_DATA_SAVE = "D:/Few-Shot-Project/data/ExtClassOneDefault/%s/test_data.npy"  % class_path
# TEST_LABEL_SAVE = "D:/Few-Shot-Project/data/ExtClassOneDefault/%s/test_label.npy"  % class_path

def drawHeatmap(data, title, col_labels, row_labels, cbar_label, formatter="%s", **kwargs):
    fig, ax = plt.subplots()
    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, formatter%data[i][j],
                           ha="center", va="center", color="k")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_data_ = np.load(TRAIN_DATA_SAVE)
    train_label = np.load(TRAIN_LABEL_SAVE)
    test_data_ = np.load(TEST_DATA_SAVE)
    test_label = np.load(TEST_LABEL_SAVE)

    pca = PCA(n_components=2)
    pca.fit(train_data_)
    train_data = pca.transform(train_data_)
    test_data = pca.transform(test_data_)

    # tsne = TSNE(n_components=2)
    # data = np.concatenate((train_data_,test_data_))
    # data = tsne.fit_transform(data)

    # mds = MDS(n_components=2)
    # data = np.concatenate((train_data_,test_data_))
    # data = mds.fit_transform(data)

    # isomap = Isomap(n_components=2)
    # data = np.concatenate((train_data_,test_data_))
    # data = isomap.fit_transform(data)

    # lle = LocallyLinearEmbedding(n_components=2,eigen_solver="dense")
    # data = np.concatenate((train_data_,test_data_))
    # data = lle.fit_transform(data)

    # train_data = data[:len(train_data_)]
    # test_data = data[len(train_data):]

    knn = KNN(n_neighbors=1)
    svm = SVC(gamma='auto')

    knn.fit(train_data_, train_label)
    svm.fit(train_data_, train_label)

    knn_predict = knn.predict(test_data_)
    svm_predict = svm.predict(test_data_)

    knn_acc = np.sum(knn_predict == test_label) / len(test_label)
    svm_acc = np.sum(svm_predict == test_label) / len(test_label)

    plt.figure(figsize=(15, 15))
    plt.title('Using KNN with k=1, acc=%.2f' % knn_acc, fontsize=25)
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 1],
             [x[1] for x, l in zip(train_data, train_label) if l == 1], 'ro', label='train_malware')
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 0],
             [x[1] for x, l in zip(train_data, train_label) if l == 0], 'bo', label='train_benign')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 1], 'rx',
             label='malware_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 0], 'bx',
             label='benign_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 0], 'kx',
             label='malware_wrong')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 1], 'gx',
             label='benign_wrong')
    plt.legend(fontsize=20)
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.title('Using SVM, acc=%.2f' % svm_acc, fontsize=25)
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 1],
             [x[1] for x, l in zip(train_data, train_label) if l == 1], 'ro', label='train_malware')
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 0],
             [x[1] for x, l in zip(train_data, train_label) if l == 0], 'bo', label='train_benign')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 1], 'rx',
             label='malware_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 0], 'bx',
             label='benign_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 0], 'kx',
             label='malware_wrong')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 1], 'gx',
             label='benign_wrong')
    plt.legend(fontsize=20)
    plt.show()
    #
    # print(pca.explained_variance_ratio_)
    svm_mat = confusion_matrix(test_label, svm_predict)
    knn_mat = confusion_matrix(test_label, knn_predict)

    #混淆矩阵归一化
    svm_sum = np.sum(svm_mat, axis=1, keepdims=True)
    knn_sum = np.sum(knn_mat, axis=1, keepdims=True)
    svm_mat = svm_mat/svm_sum
    knn_mat = knn_mat/knn_sum

    print(svm_mat)
    print("-------------------")
    print(knn_mat)

    labels = ["benign", "malware"]
    drawHeatmap(svm_mat, "svm's normalized confusion matrix", labels, labels, "acc", formatter="%.4f", cmap="YlOrRd")
    drawHeatmap(knn_mat, "knn's normalized confusion matrix", labels, labels, "acc", formatter="%.4f", cmap="YlOrRd")