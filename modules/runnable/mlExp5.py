# 本实验是验证传统机器学习在小样本情况下，将恶意代码分为小类的实验
# 5-shot


import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

TRAIN_DATA_PATHS = ["D:/Few-Shot-Project/data/ExtClass5Shot/trojan/OnLineGames_train_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/PcClient_train_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/trojan/LdPinch_train_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/Agent_train_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/aworm/AutoRun_train_data.npy"]
TRAIN_LABEL_PATHS = ["D:/Few-Shot-Project/data/ExtClass5Shot/trojan/OnLineGames_train_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/PcClient_train_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/trojan/LdPinch_train_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/Agent_train_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/aworm/AutoRun_train_label.npy"]

TEST_DATA_PATHS = ["D:/Few-Shot-Project/data/ExtClass5Shot/trojan/OnLineGames_test_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/PcClient_test_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/trojan/LdPinch_test_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/Agent_test_data.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/aworm/AutoRun_test_data.npy"]
TEST_LABEL_PATHS = ["D:/Few-Shot-Project/data/ExtClass5Shot/trojan/OnLineGames_test_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/PcClient_test_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/trojan/LdPinch_test_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/backdoor_default/Agent_test_label.npy",
                    "D:/Few-Shot-Project/data/ExtClass5Shot/aworm/AutoRun_test_label.npy"]

seed = 22

def drawHeatmap(data, title, col_labels, row_labels, cbar_label, formatter="%s", **kwargs):
    fig, ax = plt.subplots(figsize=(10,8))
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
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
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

#清洗旧的数据以满足当前实验的需要
def make(data_path, label_path):
    datas = np.load(data_path)
    labels = np.load(label_path)

    return_datas = []

    for data,label in zip(datas, labels):
        if label == 1:
            return_datas.append(data)

    return np.array(return_datas)

train_datas = make(TRAIN_DATA_PATHS[0],TRAIN_LABEL_PATHS[0]),\
              make(TRAIN_DATA_PATHS[1],TRAIN_LABEL_PATHS[1]),\
              make(TRAIN_DATA_PATHS[2],TRAIN_LABEL_PATHS[2]),\
              make(TRAIN_DATA_PATHS[3],TRAIN_LABEL_PATHS[3]),\
              make(TRAIN_DATA_PATHS[4],TRAIN_LABEL_PATHS[4])
train_labels = np.array([0]*len(train_datas[0])),\
               np.array([1]*len(train_datas[1])),\
               np.array([2]*len(train_datas[2])),\
               np.array([3]*len(train_datas[3])),\
               np.array([4]*len(train_datas[4]))

test_datas = make(TEST_DATA_PATHS[0],TEST_LABEL_PATHS[0]),\
              make(TEST_DATA_PATHS[1],TEST_LABEL_PATHS[1]),\
              make(TEST_DATA_PATHS[2],TEST_LABEL_PATHS[2]),\
              make(TEST_DATA_PATHS[3],TEST_LABEL_PATHS[3]),\
              make(TEST_DATA_PATHS[4],TEST_LABEL_PATHS[4])
test_labels = np.array([0]*len(test_datas[0])),np.array([1]*len(test_datas[1])),\
                np.array([2]*len(test_datas[2])),np.array([3]*len(test_datas[3])),\
                np.array([4]*len(test_datas[4]))

train_data = np.concatenate(train_datas, axis=0)
train_label = np.concatenate(train_labels, axis=0)
test_data = np.concatenate(test_datas, axis=0)
test_label = np.concatenate(test_labels, axis=0)

np.random.seed(seed)
train_data = np.random.permutation(train_data)
train_label = np.random.permutation(train_label)
test_data = np.random.permutation(test_data)
test_label = np.random.permutation(test_label)

svm_train_label = label_binarize(train_label, classes=[0,1,2,3,4])
svm_test_label = label_binarize(test_label, classes=[0,1,2,3,4])

knn = KNN(n_neighbors=1)
svm = OneVsRestClassifier(SVC(gamma="auto", probability=True, decision_function_shape="ovr"))

knn.fit(train_data, train_label)
svm.fit(train_data, svm_train_label)

knn_predict = knn.predict(test_data)
svm_predict = np.argmax(svm.predict_proba(test_data), axis=1)

knn_acc = np.sum(knn_predict==test_label)/test_data.shape[0]
svm_acc = np.sum(svm_predict==test_label)/test_data.shape[0]

knn_cm = confusion_matrix(test_label, knn_predict)
svm_cm = confusion_matrix(test_label, svm_predict)

svm_sum = np.sum(svm_cm, axis=1, keepdims=True)
knn_sum = np.sum(knn_cm, axis=1, keepdims=True)

knn_cm = knn_cm/knn_sum
svm_cm = svm_cm/svm_sum

tags = ["trojan.PSW.OnLineGames","backdoor_default.PcClient","trojan.PSW.LdPinch","backdoor_default.Agent","worm.AutoRun"]

drawHeatmap(svm_cm, "svm classify confusion matrix, acc=%.3f" % svm_acc, tags, tags, "relative acc", formatter="%.4f", cmap="GnBu")
drawHeatmap(knn_cm, "knn classify confusion matrix, acc=%.3f" % knn_acc, tags, tags, "relative acc", formatter="%.4f", cmap="YlOrRd")
