import torch as t
import torch.nn as nn
import torch.nn.functional as F

from modules.utils.dlUtils import RN_repeat_query_instance


# 基于卷积神经网络的图像嵌入网络
class ProtoNet(nn.Module):
    def __init__(self, metric="SqEuc", **kwargs):
        super(ProtoNet, self).__init__()

        self.metric = metric

        # 第一层是一个1输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 经过这层以后，尺寸除以4
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第二层是一个64输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 卷积核的宽度为3,13变为10，再经过宽度为2的pool变为5
        # 经过这层以后，尺寸除以4
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第三层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数,的卷积层
        # 经过这层以后，尺寸除以2
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第四层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数的卷积层
        # 经过这层以后，尺寸除以2
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # self.Transformer = nn.Linear(kwargs['feature_in'], kwargs['feature_out'])

    def forward(self, support, query, save_embed=False):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)

        self.forward_inner_var = None
        self.forward_outer_var = None

        # input shape: [n, k, d]
        def proto_mean(tensors):
            return tensors.mean(dim=1).squeeze()

        def proto_correct_attention(tensors):
            # 利用类内向量的均值向量作为键使用注意力机制生成类向量
            # 类均值向量
            # centers shape: [n,k,d]->[n,d]->[n,k,d]
            support_center = tensors.mean(dim=1).repeat(1, k).reshape(n, k, -1)

            # -------------------------------------------------------------
            # 支持集与均值向量的欧式平方距离
            # dis shape: [n,k,d]->[n,k]
            support_dis = ((tensors - support_center) ** 2).sum(dim=2).sqrt()
            # dis_mean_shape: [n,k]->[n]->[n,k]
            support_dis_mean = support_dis.mean(dim=1).unsqueeze(dim=1).repeat(1,k)
            support_dis = t.abs(support_dis-support_dis_mean).neg()
            # attention shape: [n,k]->[n,k,d]
            attention_map = t.softmax(support_dis, dim=1).unsqueeze(dim=2).repeat(1,1,d)

            # return shape: [n,k,d]->[n,d]
            return t.mul(tensors, attention_map).sum(dim=1).squeeze()
            # -------------------------------------------------------------

        def proto_attention(tensors):
            # 利用类内向量的均值向量作为键使用注意力机制生成类向量
            # 类均值向量
            # centers shape: [n,k,d]->[n,d]->[n,k,d]
            support_center = tensors.mean(dim=1).repeat(1, k).reshape(n, k, -1)

            # attention shape: [n,k,d]
            # 类内对每个向量的注意力映射，由负距离输入到softmax生成
            attention_map = t.softmax(((tensors - support_center) ** 2).sum(dim=2).neg(), dim=1)
            # [n,k]->[n,k,d]
            # 将向量的注意力系数重复到每个位置
            attention_map = attention_map.unsqueeze(dim=2).repeat(1, 1, d)
            # support: [n,k,d]->[n,d]
            # 注意力映射后的支持集中心向量
            return t.mul(tensors, attention_map).sum(dim=1).squeeze()

        # 每一层都是以上一层的输出为输入，得到新的输出、
        # 支持集输入是N个类，每个类有K个实例
        support = self.layer1(support)
        support = self.layer2(support)
        support = self.layer3(support)
        support = self.layer4(support)

        # 查询集的输入是N个类，每个类有qk个实例
        # 但是，测试的时候也需要支持单样本的查询
        query = self.layer1(query)
        query = self.layer2(query)
        query = self.layer3(query)
        query = self.layer4(query)

        # support = self.Transformer(support.view(self.n, self.k, -1))

        if save_embed:
            return support.view(n, k, -1),query.view(n, qk/n, -1)

        # 计算类的原型向量
        # shape: [n, k, d]
        support = support.view(n, k, -1)
        d = support.size(2)

        # proto shape: [n, d]
        # proto = proto_correct_attention(support)
        proto = proto_mean(support)
        self.forward_inner_var = ((support - proto.unsqueeze(dim=1).repeat(1,k,1)) ** 2).sum()
        self.forward_outer_var = proto.var(dim=0).sum()
        support = proto

        # 直接将均值向量作为原型向量
        # support = support.mean(dim=1).squeeze()

        # 利用类内向量的均值向量作为键使用注意力机制生成类向量
        # 类均值向量
        # centers shape: [n,k,d]->[n,d]->[n,k,d]
        # support_center = support.mean(dim=1).repeat(1, k).reshape(n, k, -1)

        # -------------------------------------------------------------
        # # 支持集与均值向量的欧式平方距离
        # # dis shape: [n,k,d]->[n,k]
        # support_dis = ((support - support_center) ** 2).sum(dim=2).sqrt()
        # # dis_mean_shape: [n,k]->[n]->[n,k]
        # support_dis_mean = support_dis.mean(dim=1).unsqueeze(dim=1).repeat(1,k)
        # support_dis = t.abs(support_dis-support_dis_mean).neg()
        # # attention shape: [n,k]->[n,k,d]
        # attention_map = t.softmax(support_dis, dim=1).unsqueeze(dim=2).repeat(1,1,d)
        # support = t.mul(support, attention_map).sum(dim=1).squeeze()
        # -------------------------------------------------------------


        # # 支持集与均值向量的欧式平方距离
        # # dis shape: [n,k,d]->[n]->[n,k]
        # support_dis_sum = ((support - support_center) ** 2).sum(dim=2).sum(dim=1).unsqueeze(dim=1).repeat(1, k)
        # # attention shape: [n,k,d]
        # # 类内对每个向量的注意力映射，由负距离输入到softmax生成
        # attention_map = t.softmax((((support - support_center) ** 2).sum(dim=2) / support_dis_sum).neg(), dim=1)
        # # [n,k]->[n,k,d]
        # # 将向量的注意力系数重复到每个位置
        # attention_map = attention_map.unsqueeze(dim=2).repeat(1, 1, d)
        # # support: [n,k,d]->[n,d]
        # # 注意力映射后的支持集中心向量
        # support = t.mul(support, attention_map).sum(dim=1).squeeze()

        # 将原型向量与查询集打包
        # shape: [n,d]->[qk, n, d]
        support = support.repeat((qk,1,1)).view(qk,n,-1)

        query = query.repeat(n,1,1,1,1).transpose(0,1).contiguous().view(qk,n,-1)

        # query = RN_repeat_query_instance(query, self.n).view(-1,self.n,support.size(1),support.size(2),support.size(3))

        if self.metric == "SqEuc":
            # 由于pytorch中的NLLLoss需要接受对数概率，根据官网上的提示最后一层改为log_softmax
            # 已修正：以负距离输入到softmax中,而不是距离
            posterior = F.log_softmax(t.sum((query-support)**2, dim=2).sqrt().neg(),dim=1)

        elif self.metric == 'cos':
            # 在原cos相似度的基础上添加放大系数
            scale = 10
            posterior = F.cosine_similarity(query, support, dim=2)*scale
            posterior = F.log_softmax(posterior, dim=1)

        return posterior
