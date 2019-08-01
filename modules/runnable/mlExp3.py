# 机器学习方法实验3
# 本实验用于测试无其他基础样本，只有单类的支撑样本的情况下，小样本学习：1-shot，5-shot，10-shot，20-shot 的效果
# 使用的数据都是没有经过标准化的数据，由于不同维度之间数据幅度差异大，不宜使用降维方法可视化，因此没有使用可视化方法
# 同样使用训练的良性样本和测试样本分开的策略，训练样本中良性样本数量设定为等同于恶意样本的数量

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix

class_path = "virus"
sub_class = "VB"
TRAIN_DATA_SAVE = "D:/Few-Shot-Project/data/ExtClassFewShot/%s/%s_train_data.npy" % (class_path, sub_class)
TRAIN_LABEL_SAVE = "D:/Few-Shot-Project/data/ExtClassFewShot/%s/%s_train_label.npy" % (class_path, sub_class)
TEST_DATA_SAVE = "D:/Few-Shot-Project/data/ExtClassFewShot/%s/%s_test_data.npy" % (class_path, sub_class)
TEST_LABEL_SAVE = "D:/Few-Shot-Project/data/ExtClassFewShot/%s/%s_test_label.npy" % (class_path, sub_class)

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

def drawHeatmapWithGrid(data, title, col_labels, row_labels, cbar_label, formatter="%s", **kwargs):
    fig, ax = plt.subplots()
    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

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

    knn = KNN(n_neighbors=1)
    svm = SVC(gamma='auto')

    knn.fit(train_data_, train_label)
    svm.fit(train_data_, train_label)

    knn_predict = knn.predict(test_data_)
    svm_predict = svm.predict(test_data_)

    knn_acc = np.sum(knn_predict == test_label) / len(test_label)
    svm_acc = np.sum(svm_predict == test_label) / len(test_label)
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
    drawHeatmap(svm_mat, "%s: svm's normalized confusion matrix, acc=%.3f"%(class_path+"-"+sub_class,svm_acc), labels, labels, "acc", formatter="%.4f", cmap="YlOrRd")
    drawHeatmap(knn_mat, "%s: knn's normalized confusion matrix, acc=%.3f"%(class_path+"-"+sub_class,knn_acc), labels, labels, "acc", formatter="%.4f", cmap="YlOrRd")
