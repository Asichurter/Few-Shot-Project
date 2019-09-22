import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.model.MalResnet import ResLayer

class ResProtoNet(nn.Module):
    def __init__(self, channels=[1,32,64,64,64,128], strides=[1,2,2,2]):
        super(ResProtoNet, self).__init__()
        assert len(channels)-2==len(strides), "stride与channel数量不一致！"
        self.Ahead = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            # 重叠池化
            nn.MaxPool2d(3,2,1)
        )
        # ResNet14
        layers = [ResLayer(2, channels[i+1], channels[i+2], strides[i]) for i in range(len(strides))]
        self.Layers = nn.Sequential(*layers)


    def forward(self, support, query):
        # support shape: [n,k,c,w,w]
        assert len(support.size())==5 and len(query.size())==4,\
                    "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"

        n = support.size(0)
        k = support.size(1)
        w = support.size(3)
        query_length = query.size(0)

        pool_size = w
        down_sample = [4,2,2,2]
        for i in down_sample:
            pool_size = pool_size // i

        support = support.view(n*k,1,w,w)
        query = query.view(query_length,1,w,w)

        support = F.avg_pool2d(self.Layers(self.Ahead(support)), pool_size)
        query = F.avg_pool2d(self.Layers(self.Ahead(query)), pool_size)

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

        support = support.view(n,k,-1)
        d = support.size(2)
        query = query.view(query_length,d)

        # support shape: [n,k,d]->[n,d]->[qk,n,d]
        support = proto_mean(support).repeat((query_length,1,1))
        # query shape: [qk,d]->[n,qk,d]->[qk,n,d]
        query = query.repeat((n,1,1)).transpose(0,1).contiguous()

        # 使用欧式平方距离作为距离度量
        return F.log_softmax(t.sum((query - support) ** 2, dim=2).neg(), dim=1)



