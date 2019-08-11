# 本实验是为了验证Resnet对恶意代码识别具有可行性
# 采用的数据中，训练和测试的良性数据同样会被分离

import numpy as np
from modules.utils.imageUtils import validate
import torch as t
from modules.model.MalResnet import ResNet
from modules.model.datasets import DirDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchstat import stat

early_stop = False
early_stop_window = 3
model_save_path = 'D:/peimages/New/whole_exp/'
save_path = 'D:/Few-Shot-Project/doc/dl_whole_exp/'
train_set_path = 'D:/peimages/New/whole_exp/train/'
val_set_path = 'D:/peimages/New/whole_exp/validate/'

# 最大迭代次数
MAX_ITER = 30

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

# 记录历史的训练和验证数据
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

# 训练集数据集
dataset = DirDataset(train_set_path)
# 验证集数据集
val_set = DirDataset(val_set_path)

# 训练集数据加载器
train_loader = DataLoader(dataset, batch_size=48, shuffle=True)
# 验证集数据加载器
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

resnet = ResNet(1, 2)

# resnet,pars = get_pretrained_resnet()
resnet = resnet.cuda()

# opt = t.optim.SGD(pars, lr=1e-2, momentum=0.9, weight_decay=0.2, nesterov=True)
# 根据resnet的论文，使用1e-4的权重衰竭
opt = t.optim.Adam(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
# 使用二元交叉熵为损失函数（可以替换为交叉熵损失函数）
criteria = t.nn.BCELoss()
# 学习率调整器，使用的是按照指标的变化进行调整的调整器
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(resnet))

# num = 0
# best_val_loss = 0.
# print('training...')
# for i in range(MAX_ITER):
#     print(i, ' th')
#     a = 0
#     c = 0
#     Loss = 0.
#
#     # 将模型调整为学习状态
#     resnet.train()
#     for datas, l in train_loader:
#         opt.zero_grad()
#         datas = datas.cuda()
#
#         # 创建可以输入到损失函数的float类型标签batch
#         labels = [[1, 0] if L == 0 else [0, 1] for L in l]
#         labels = t.FloatTensor(labels).cuda()
#
#         out = resnet(datas).squeeze()
#         loss = criteria(out, labels).cuda()
#         loss.backward()
#         opt.step()
#
#         # 计算损失和准确率
#         Loss += loss.data.item()
#         # 进行与实际标签的比较时，由于标签是LongTensor类型，因此转化
#         # 选用值高的一个作为预测结果
#         predict = t.LongTensor([0 if x[0] >= x[1] else 1 for x in out])
#         a += predict.shape[0]
#         c += (predict == l).sum().item()
#     print('train loss: ', Loss)
#     train_loss_history.append(Loss)
#     print('train acc: ', c / a)
#     train_acc_history.append(c / a)
#
#     val_acc, val_loss = validate(resnet, val_loader, criteria)
#     print('val loss: ', val_loss)
#     val_loss_history.append(val_loss)
#     print('val accL: ', val_acc)
#     val_acc_history.append(val_acc)
#
#     if len(val_loss_history) == 1 or val_loss < best_val_loss:
#         best_val_loss = val_loss
#         t.save(resnet, model_save_path + 'best_loss_model.h5')
#         print('save model at epoch %d' % i)
#
#     num += 1
#     # 使用学习率调节器来随验证损失来调整学习率
#     scheduler.step(val_loss)
#
#
#
# # 根据历史值画出准确率和损失值曲线
# x = [i for i in range(num)]
#
# plt.title('Whole Experiment Accuracy')
# plt.plot(x, val_acc_history, linestyle='-', color='green', label='validate')
# plt.plot(x, train_acc_history, linestyle='-', color='red', label='train')
# plt.legend()
# plt.savefig(save_path + 'acc.png')
# plt.show()
#
# plt.title('Whole Experiment Loss')
# plt.plot(x, val_loss_history, linestyle='--', color='green', label='validate')
# plt.plot(x, train_loss_history, linestyle='--', color='red', label='train')
# plt.legend()
# plt.savefig(save_path + 'loss.png')
# plt.show()
#
# acc_np = np.array(val_acc_history)
# los_np = np.array(val_loss_history)
#
# np.save(save_path + 'acc.npy', acc_np)
# np.save(save_path + 'loss.npy', los_np)
#
# Acc,Loss,real,pred = validate(resnet, val_loader, criteria, return_predict=True)
# conf_mat = confusion_matrix(real, pred)
#
# sum_up = np.sum(conf_mat, axis=1, keepdims=True)
# conf_mat = conf_mat/sum_up
#
# tags = ["benign", "malware"]
#
# drawHeatmapWithGrid(conf_mat, "Resnet's confusion matrix acc=%.3f loss=%.3f"%(Acc,Loss),
#                     tags, tags, "relative acc", formatter="%.4f", cmap="YlOrRd")

