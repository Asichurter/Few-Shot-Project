# 本实验是为了测试Resnet恶意代码分为子类的性能
# 训练集和测试集中的良性样本是分开的

import numpy as np
from modules.utils.imageUtils import classfy_validate
import torch as t
from modules.model.MalResnet import ResNet
from modules.utils.datasets import ClassifyDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.nn import CrossEntropyLoss

train = True
model_save_path = 'D:/peimages/New/sub_classify_exp/'
save_path = 'D:/Few-Shot-Project/doc/dl_sub_classify_exp/'
train_set_path = 'D:/peimages/New/sub_classify_exp/train/'
val_set_path = 'D:/peimages/New/sub_classify_exp/validate/'


# 最大迭代次数
MAX_ITER = 100

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
train_acc_history = [] if train else np.load(save_path+"train_acc.npy").tolist()
val_acc_history = [] if train else np.load(save_path+"train_loss.npy").tolist()
train_loss_history = [] if train else np.load(save_path+"acc.npy").tolist()
val_loss_history = [] if train else np.load(save_path+"loss.npy").tolist()

# 训练集数据集
dataset = ClassifyDataset(train_set_path, 5)
# 验证集数据集
val_set = ClassifyDataset(val_set_path, 5)

# 训练集数据加载器
train_loader = DataLoader(dataset, batch_size=48, shuffle=True)
# 验证集数据加载器
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

resnet = ResNet(1, 5) if train else t.load(model_save_path + 'best_loss_model.h5')
# resnet,pars = get_pretrained_resnet()
resnet = resnet.cuda()

# opt = t.optim.SGD(pars, lr=1e-2, momentum=0.9, weight_decay=0.2, nesterov=True)
# 根据resnet的论文，使用1e-4的权重衰竭
opt = t.optim.Adam(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
# 使用二元交叉熵为损失函数（可以替换为交叉熵损失函数）
criteria = CrossEntropyLoss()
# 学习率调整器，使用的是按照指标的变化进行调整的调整器
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5)

num = len(train_acc_history)
best_val_loss = 0. if train else min(val_loss_history)
print('training...')
for i in range(len(train_acc_history), len(train_acc_history)+MAX_ITER):
    print(i, ' th')
    a = 0
    c = 0
    Loss = 0.

    # 将模型调整为学习状态
    resnet.train()
    for datas, l in train_loader:
        opt.zero_grad()
        datas = datas.cuda()

        # 创建可以输入到损失函数的float类型标签batch
        # labels = label_binarize(l, [i for i in range(6)])
        labels = t.LongTensor(l).cuda()
        l = l.cuda()

        out = resnet(datas).squeeze()
        loss = criteria(out, labels).cuda()
        loss.backward()
        opt.step()

        # 计算损失和准确率
        Loss += loss.data.item()
        # 进行与实际标签的比较时，由于标签是LongTensor类型，因此转化
        # 选用值高的一个作为预测结果
        predict = t.argmax(out, dim=1)
        a += predict.shape[0]
        c += (predict == l).sum().item()
    print('train loss: ', Loss)
    train_loss_history.append(Loss)
    print('train acc: ', c / a)
    train_acc_history.append(c / a)

    val_acc, val_loss = classfy_validate(resnet, val_loader, criteria, 5)
    print('val loss: ', val_loss)
    val_loss_history.append(val_loss)
    print('val acc: ', val_acc)
    val_acc_history.append(val_acc)

    if len(val_loss_history) == 1 or val_loss < best_val_loss:
        best_val_loss = val_loss
        t.save(resnet, model_save_path + 'best_loss_model.h5')
        print('save model at epoch %d' % i)

    num += 1
    # 使用学习率调节器来随验证损失来调整学习率
    scheduler.step(val_loss)
    if i%20==0:
        choice = input("%d epoches have done, continue?"%num)
        if choice == "n" or choice=="no":
            break



# 根据历史值画出准确率和损失值曲线
x = [i for i in range(num)]

plt.title('Classifying Accuracy')
plt.plot(x, val_acc_history, linestyle='-', color='green', label='validate')
plt.plot(x, train_acc_history, linestyle='-', color='red', label='train')
plt.legend()
plt.savefig(save_path + 'acc.png')
plt.show()

plt.title('Classifying Loss')
plt.plot(x, val_loss_history, linestyle='--', color='green', label='validate')
plt.plot(x, train_loss_history, linestyle='--', color='red', label='train')
plt.legend()
plt.savefig(save_path + 'loss.png')
plt.show()

acc_np = np.array(val_acc_history)
los_np = np.array(val_loss_history)

np.save(save_path + 'acc.npy', acc_np)
np.save(save_path + 'loss.npy', los_np)
np.save(save_path + "train_acc.npy", np.array(train_acc_history))
np.save(save_path + "train_loss.npy", np.array(train_loss_history))

Acc,Loss,real,pred = classfy_validate(resnet, val_loader, criteria, 5, return_predict=True)
conf_mat = confusion_matrix(real, pred)

sum_up = np.sum(conf_mat, axis=1, keepdims=True)
conf_mat = conf_mat/sum_up

tags = ["backdoor_default.Agent", "backdoor_default.PcClient", "trojan.PSW.LdPinch", "trojan.PSW.OnLineGames", "worm.AutoRun"]

drawHeatmapWithGrid(conf_mat, "Subclass Classfying: Resnet's confusion matrix acc=%.3f loss=%.3f"%(Acc,Loss),
                    tags, tags, "relative acc", formatter="%.4f", cmap="YlGn")

