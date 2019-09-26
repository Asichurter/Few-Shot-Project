import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math as m

from modules.utils.dlUtils import RN_repeat_query_instance


# 基于卷积神经网络的图像嵌入网络
class EmbeddingNet(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # 修改：考虑参考Resnet的实现，使用大卷积核和大的步伐
        # 第一层是一个1输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 由于卷积核的宽度是3，因此28x28变为64x25x25,经过了pool后变为64x13x13
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第二层是一个64输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 卷积核的宽度为3,13变为10，再经过宽度为2的pool变为5
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第三层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数,的卷积层
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第四层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数的卷积层
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    # 前馈函数，利用图像输入得到图像嵌入后的输出
    def forward(self, x, tracker=None):
        # 每一层都是以上一层的输出为输入，得到新的输出
        # print("0", x.size())
        x = self.layer1(x)
        # print("1",x.size())
        x = self.layer2(x)
        # print("2", x.size())
        x = self.layer3(x)
        # print("3", x.size())
        x = self.layer4(x)
        # print("4", x.size())
        return x

        # 输出的矩阵深度改为512
        # # TODO:添加了一个平均池化层以将嵌入输出为512x1x1的嵌入
        # return F.avg_pool2d(x, 8)

# 关系神经网络，用于在得到图像嵌入向量后计算关系的神经网络
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
# 第一层是128输入（因为两个深度为64的矩阵相加），64个3x3过滤器，周围补0，批正则化，relu为激活函数，2x2maxpool的卷积层
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,128,kernel_size=3,padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(128, affine=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2))
# 第二层是64输入，64个3x3过滤器，周围补0，批正则化，relu为激活函数，2x2maxpool的卷积层
        self.layer2 = nn.Sequential(
                        nn.Conv2d(128,128,kernel_size=3,padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(128, affine=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2))
        # 第三层是一个将矩阵展平的线性全连接层，输入64维度，输出隐藏层维度10维度
        self.fc1 = nn.Linear(128,hidden_size)
        # 第四层是一个结束层，将10个隐藏层维度转化为1个维度的值，得到关系值
        self.fc2 = nn.Linear(hidden_size,1)

    # 关系网络的前馈方法
    def forward(self,x, tracker=None):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = F.avg_pool2d(2)

        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = t.sigmoid(out)
        # out = F.relu(out)

        # TODO:论文中使用sigmoid来将值映射到[0,1]区间内，但是也可以考虑将sigmoid换为softmax，然后将损失函数换为交叉熵
        # out = t.sigmoid(self.fc2(out))
        #print(out.size())
        # out = t.sigmoid(out)
        return out

class RN(nn.Module):
    def __init__(self, linear_hidden_size=8):
        super(RN, self).__init__()
        self.Embed = EmbeddingNet()
        self.LinearInputSize = 64
        self.Relation = RelationNetwork(self.LinearInputSize, linear_hidden_size)

    def forward(self, support, query, tracker=None, feature_out=False):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)

        # 支持集嵌入层输出大小为样本数量(n*k)*卷积核深度64*卷积输出尺寸62*62
        support = self.Embed(support)
        query = self.Embed(query)

        w_ = support.size(2)

        # TODO: 采用论文中的设定，query set和support set的大小之和为20
        # 即：如果k=5，那么每个类中：query set中包含20-5=15个样本
        # 先将输出整理为：(类别),(类别内部的样本),... 的形式
        support = support.view(n, k, 64, w_, w_)

        # TODO: 为了减少配对数量，简化匹配流程，利用support set中的向量生成一个类别向量:修改了生成方式，利用平均值而非和
        # 论文中生成类别向量的方式是单纯将类别内部的向量求和后平均,将类别内部向量维给约减掉
        support = support.sum(dim=1).squeeze()

        if feature_out:
            return support,query

        # 支持集重复样本个数遍，代表每个样本都要与所有类进行比对
        support = support.repeat(qk,1,1,1,1)
        # 询问集重复样本种类遍
        query = query.repeat(n,1,1,1,1).transpose(0,1).contiguous()

        # 修正：如果要将所有查询样本的嵌入重复以供与支持集中的类嵌入一一比较的话，不能直接repeat，因为会导致错乱（如test中）
        # 例如(x1,x2)2次repeat的结果是(x1,x2,x1,x2)，但是基于支持集类嵌入的重复方式，想得到的是(x1,x1,x2,x2)
        # 因此只能使用外部的辅助方法，该方法是遍历第一维的所有向量，repeat以后再连在一起
        # query_out = RN_repeat_query_instance(query_out, n).view(-1,5,support_out.size(2),support_out.size(3),support_out.size(4))

        # TODO: 论文中直接将两个嵌入向量按深度连接起来组成关系模块的输入，可以考虑其他的特征连接方式
        # 将支持集与询问集连接起来作为关系网络的输入
        # 连接的维度是按照卷积核的深度进行连接的
        relation_input = t.cat((support,query),2).view(-1,64*2,w_, w_)
        # relation_input = (support_out-query_out).view(-1,64,self.EmbedOutSize,self.EmbedOutSize)

        relations = self.Relation(relation_input)
        return relations.view(-1,n)

if __name__ == '__main__':
    pass


