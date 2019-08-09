import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math as m


# 基于卷积神经网络的图像嵌入网络
class EmbeddingNet(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # 第一层是一个1输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 由于卷积核的宽度是3，因此28x28变为64x25x25,经过了pool后变为64x13x13
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 第二层是一个64输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 卷积核的宽度为3,13变为10，再经过宽度为2的pool变为5
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 第三层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数,的卷积层
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        # 第四层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数的卷积层
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    # 前馈函数，利用图像输入得到图像嵌入后的输出
    def forward(self, x):
        # 每一层都是以上一层的输出为输入，得到新的输出
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        # 输出的矩阵深度是64
        return out  # 64

#关系神经网络，用于在得到图像嵌入向量后计算关系的神经网络
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
#第一层是128输入（因为两个深度为64的矩阵相加），64个3x3过滤器，周围补0，批正则化，relu为激活函数，2x2maxpool的卷积层
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
#第二层是64输入，64个3x3过滤器，周围补0，批正则化，relu为激活函数，2x2maxpool的卷积层
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        #第三层是一个将矩阵展平的线性全连接层，输入64维度，输出隐藏层维度10维度
        self.fc1 = nn.Linear(input_size,hidden_size)
        #第四层是一个结束层，将10个隐藏层维度转化为1个维度的值，得到关系值
        self.fc2 = nn.Linear(hidden_size,1)

    #关系网络的前馈方法
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = t.sigmoid(self.fc2(out))
        #print(out.size())
        return out

class RN(nn.Module):
    def __init__(self, input_size, linear_hidden_size=8, k=5, n=20, qk=15):
        super(RN, self).__init__()
        self.K = k
        self.N = n
        self.Embed = EmbeddingNet()
        self.EmbedOutSize = m.floor((m.floor((input_size-2)/2)-2)/2)
        self.LinearInputSize = 64*m.floor(m.floor(m.floor((m.floor((input_size-2)/2)-2)/2)/2)/2)
        self.Relation = RelationNetwork(self.LinearInputSize, linear_hidden_size)
        self.QK = qk

    def forward(self, *x):
        #分别为：类别数，一个类的支持集内部的样本数量，一个类的查询集内部的样本数量
        n = self.N
        k = self.K
        qk = self.QK

        # 支持集嵌入层输出大小为样本数量(n*k)*卷积核深度64*卷积输出尺寸62*62
        support_out = self.Embed(x[0])
        query_out = self.Embed(x[1])

        # TODO: 采用论文中的设定，query set和support set的大小之和为20
        # 即：如果k=5，那么每个类中：query set中包含20-5=15个样本
        #先将输出整理为：(类别),(类别内部的样本),... 的形式
        support_out = support_out.view(n, k, 64, self.EmbedOutSize, self.EmbedOutSize)

        # TODO: 为了减少配对数量，简化匹配流程，利用support set中的向量生成一个类别向量
        # 论文中生成类别向量的方式是单纯将类别内部的向量求和后,将类别内部向量维给约减掉
        support_out = t.sum(support_out, 1).squeeze(1)
        #支持集重复样本个数遍，代表每个样本都要与所有类进行比对
        support_out = support_out.repeat((qk*k,1,1,1,1))
        #询问集重复样本种类遍
        query_out = query_out.repeat((n,1,1,1,1)).transpose(0,1).contiguous()

        # 将支持集与询问集连接起来作为关系网络的输入
        # 连接的维度是按照卷积核的深度进行连接的
        relation_input = t.cat((support_out,query_out),2).view(-1,64*2,self.EmbedOutSize, self.EmbedOutSize)

        relations = self.Relation(relation_input)
        return relations



if __name__ == '__main__':
    pass


