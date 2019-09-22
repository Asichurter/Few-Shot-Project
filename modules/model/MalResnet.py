# 用于恶意代码识别和分类的Resnet

import torch as t
import torch.nn.functional as F
from torch.nn.modules import Conv2d, BatchNorm2d, ReLU, Sequential, MaxPool2d, Linear, Dropout2d, Dropout
from torch.nn.init import kaiming_normal_


class ResBlock(t.nn.Module):
    def __init__(self, in_size, out_size, stride=1, kernel_size=3, shortcut=None, dropout=None):
        super(ResBlock, self).__init__()
        # 残差块的左侧是两层普通的卷积网络
        # 需要注意的是，一个层的第一个残差块牵涉到尺寸增大，因此需要特别的shortcut来处理F(x)和x尺寸不同的情况
        self.Left = t.nn.Sequential()
        self.Left.add_module('conv1',
                             Conv2d(in_size, out_size, kernel_size, stride=stride, padding=int((kernel_size - 1) / 2),
                                    bias=False))
        self.Left.add_module('bn1', BatchNorm2d(out_size))
        self.Left.add_module('relu1', ReLU(inplace=True))
        if dropout is not None:
            self.Left.add_module('dropout1', Dropout2d(p=dropout))
        self.Left.add_module('conv2',
                             Conv2d(out_size, out_size, kernel_size, 1, padding=int((kernel_size - 1) / 2), bias=False))
        self.Left.add_module('bn2', BatchNorm2d(out_size))
        if dropout is not None:
            self.Left.add_module('dropout2', Dropout2d(p=dropout))
        # 右侧是捷径
        # 一般情况下，捷径就是x自己，除非是层的第一个残差块
        self.Right = shortcut

    def forward(self, x):
        left = self.Left(x)
        # 右侧为None时代表捷径就是x自己
        right = self.Right(x) if self.Right is not None else x
        # 由于left在整个过程中没有经过池化而且padding均为same，因此形状理应该一样
        assert left.shape == right.shape, '残差块内左右两部分的形状不一样无法相加!%s,%s' % (left.shape, right.shape)
        return F.relu(left + right)


class ResLayer(t.nn.Module):
    def __init__(self, block_num, in_size, out_size, kernel_size=3, stride=1, dropout=None):
        super(ResLayer, self).__init__()
        self.Layer = Sequential()
        # 层的第一个残差块因为尺寸变化，因此需要使用一个投影矩阵来使得x变为Wx
        # 使用一个1x1的卷积网络来代替投影矩阵，使得F(x)和Wx的尺寸相同
        shortcut = Sequential()
        shortcut.add_module('conv', Conv2d(in_size, out_size, 1, stride=stride, bias=False))
        shortcut.add_module('bn', BatchNorm2d(out_size))
        self.Layer.add_module('increaseBlock',
                              ResBlock(in_size, out_size, stride=stride, shortcut=shortcut, dropout=dropout))
        # 后续的残差块的捷径直接使用x
        for i in range(block_num - 1):
            self.Layer.add_module('block%d' % i, ResBlock(out_size, out_size, dropout=dropout))

    def forward(self, x):
        return self.Layer(x)


class ResNet(t.nn.Module):
    '''自己实现的ResNet18'''

    def __init__(self, channel, out=2, dropout=None, channels=[64,128,256,512,512]):
        super(ResNet, self).__init__()
        # 前置的卷积网络不是残差块，卷积网络的stride让尺寸减半，最大池化也让尺寸减半
        # 因此前置网络将尺寸减半两次
        self.Ahead = Sequential()
        self.Ahead.add_module('con1v', Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.Ahead.add_module('bn1', BatchNorm2d(64))
        self.Ahead.add_module('relu1', ReLU(inplace=True))
        self.Ahead.add_module('maxpool', MaxPool2d(3, 2, 1))
        if dropout is not None:
            self.Ahead.add_module('dropout0', Dropout2d(dropout))
        self.Layer1 = ResLayer(2, channels[0], channels[1], dropout=dropout)
        self.Layer2 = ResLayer(2, channels[1], channels[2], stride=2, dropout=dropout)
        self.Layer3 = ResLayer(2, channels[2], channels[3], stride=2, dropout=dropout)
        self.Layer4 = ResLayer(2, channels[3], channels[4], stride=2, dropout=dropout)
        self.Dense = Linear(512, out)
        self.Dropout = Dropout('dropout1', dropout) if dropout is not None else None
        self.initialize()

    def forward(self, x):
        print("x: ", x.size())
        x = self.Ahead(x)
        print("Ahead out: ", x.size())
        x = self.Layer1(x)
        print("layer1 out: ", x.size())
        x = self.Layer2(x)
        print("layer2 out: ", x.size())
        x = self.Layer3(x)
        print("layer3 out: ", x.size())
        x = self.Layer4(x)
        print("layer4 out: ", x.size())
        # 256/4/2/2/2=8,变成1的话需要长度为8的平均池化
        x = F.avg_pool2d(x, 8)
        # 将样本整理为(批大小，1)的形状
        x = x.view(x.shape[0], -1)
        # assert x.shape[1]==1, '在输入到全连接层之前，数据维度不为1维！维度:%d'%x.shape[1]
        x = self.Dense(x)
        if self.Dropout is not None:
            x = self.Dropout(x)
        return F.softmax(x, dim=1)

    def initialize(self):
        for name, par in self.named_parameters():
            # print(name, par.shape)
            if 'bn' not in name and 'bias' not in name:
                kaiming_normal_(par, mode='fan_in', nonlinearity='relu')


if __name__ == '__main__':
    x = ResNet(1).cuda()
    a = t.randn((48,1,256,256)).cuda()
    a = x(a)
    # for name,par in x.named_modules():
    #     print(name, par)