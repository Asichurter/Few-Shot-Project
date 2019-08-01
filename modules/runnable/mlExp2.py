# 机器学习方法实验2
# 本实验用于测试标准化对分类性能的提升
# 使用的数据都是没有经过标准化的数据，由于不同维度之间数据幅度差异大，不宜使用降维方法可视化，因此没有使用可视化方法

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix

class_path = "virus"
super_class_path = "ExtClassOneDefaultTestSplit"
TRAIN_DATA_SAVE = "D:/Few-Shot-Project/data/%s/%s/train_data_raw.npy" % (super_class_path,class_path)
TRAIN_LABEL_SAVE = "D:/Few-Shot-Project/data/%s/%s/train_label_raw.npy" % (super_class_path,class_path)
TEST_DATA_SAVE = "D:/Few-Shot-Project/data/%s/%s/test_data_raw.npy" % (super_class_path,class_path)
TEST_LABEL_SAVE = "D:/Few-Shot-Project/data/%s/%s/test_label_raw.npy" % (super_class_path,class_path)

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
    drawHeatmap(svm_mat, "%s: svm's normalized confusion matrix, acc=%.3f"%(class_path,svm_acc), labels, labels, "acc", formatter="%.4f", cmap="YlOrRd")
    drawHeatmap(knn_mat, "%s: knn's normalized confusion matrix, acc=%.3f"%(class_path,knn_acc), labels, labels, "acc", formatter="%.4f", cmap="YlOrRd")
