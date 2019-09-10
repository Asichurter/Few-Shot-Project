import torch as t
import torch.nn as nn
import torch.nn.functional as F

from modules.utils.dlUtils import RN_repeat_query_instance


# 基于卷积神经网络的图像嵌入网络
class ProtoNet(nn.Module):
    def __init__(self, n, k, qk, metric="SqEuc", **kwargs):
        super(ProtoNet, self).__init__()

        self.n = n
        self.k = k
        self.qk = qk
        self.metric = metric

        # 修改：考虑参考Resnet的实现，使用大卷积核和大的步伐
        # 第一层是一个1输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 经过这层以后，尺寸除以4
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第二层是一个64输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 卷积核的宽度为3,13变为10，再经过宽度为2的pool变为5
        # 经过这层以后，尺寸除以4
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第三层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数,的卷积层
        # 经过这层以后，尺寸除以2
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第四层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数的卷积层
        # 经过这层以后，尺寸除以2
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.Transformer = nn.Sequential(
            nn.Linear(k*kwargs['trans_size'], kwargs['hidden_size']),
            nn.ReLU(kwargs['hidden_size']),
            nn.Dropout(0.1),
            nn.Linear(kwargs['hidden_size'], kwargs['feature_size'])
        )

    def forward(self, *x):
        k = self.k
        qk = self.qk
        n = self.n

        # 每一层都是以上一层的输出为输入，得到新的输出、
        # 支持集输入是N个类，每个类有K个实例
        support = self.layer1(x[0])
        support = self.layer2(support)
        support = self.layer3(support)
        support = self.layer4(support)

        # 查询集的输入是N个类，每个类有qk个实例
        # 但是，测试的时候也需要支持单样本的查询
        query = self.layer1(x[1])
        query = self.layer2(query)
        query = self.layer3(query)
        query = self.layer4(query)

        query_size = query.size(0)

        # 计算类的原型向量
        # shape: [n, k, d]
        support = support.view(self.n, self.k, -1)
        d = support.size(2)

        # support = t.sum(support, dim=1).div(self.k).squeeze(1)

        # 利用类内向量的均值向量作为键使用注意力机制生成类向量
        # 类均值向量
        # centers shape: [n,k,d]->[n,d]->[n,k,d]
        support_center = support.sum(dim=1).div(k).repeat(1, k).reshape(n, k, -1)
        # 支持集与均值向量的欧式平方距离
        # dis shape: [n,k,d]->[n]->[n,k]
        support_dis_sum = ((support - support_center) ** 2).sum(dim=2).sum(dim=1).unsqueeze(dim=1).repeat(1, k)
        # attention shape: [n,k,d]
        # 类内对每个向量的注意力映射，由负距离输入到softmax生成
        attention_map = t.softmax((((support - support_center) ** 2).sum(dim=2) / support_dis_sum).neg(), dim=1)
        # [n,k]->[n,k,d]
        # 将向量的注意力系数重复到每个位置
        attention_map = attention_map.unsqueeze(dim=2).repeat(1, 1, d)
        # support: [n,k,d]->[n,d]
        # 注意力映射后的支持集中心向量
        support = t.mul(support, attention_map).sum(dim=1).squeeze()

        # 将原型向量与查询集打包
        support = support.repeat((query_size,1,1)).view(query_size,self.n,-1)

        query = query.repeat(self.n,1,1,1,1).transpose(0,1).contiguous().view(query_size,self.n, -1)

        # query = RN_repeat_query_instance(query, self.n).view(-1,self.n,support.size(1),support.size(2),support.size(3))

        if self.metric == "SqEuc":
            # 由于pytorch中的NLLLoss需要接受对数概率，根据官网上的提示最后一层改为log_softmax
            # 已修正：以负距离输入到softmax中,而不是距离
            posterior = F.log_softmax(t.sum((query-support)**2, dim=2).neg(),dim=1)

        return posterior
