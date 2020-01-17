# 最佳表现的ChannelNet备份，对应v4.0

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import warnings

def normalize(tensors):
    dim = len(tensors.size())-1
    length = tensors.size(dim)
    repeat_index = [1 for i in range(dim)]
    repeat_index.append(length)
    norm = tensors.norm(dim=dim).unsqueeze(dim=dim).repeat(repeat_index)

    return tensors/norm

def get_block_1(in_feature, out_feature, stride=1, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

def get_block_2(in_feature, out_feature, stride=1, kernel=3, padding=1, nonlinear=True):
    layers = [
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(3,2,1)
    ]
    if not nonlinear:
        layers.pop(2)
    return nn.Sequential(*layers)

def get_attention_block(in_channel, out_channel, kernel_size, stride=1, padding=1, relu=True):
    block = nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding)#,
        # nn.ReLU(inplace=True)
    )
    if relu:
        block.add_module('relu', nn.ReLU(inplace=True))
    return block

class SppPooling(nn.Module):
    def __init__(self, levels=[1,2,4]):
        super(SppPooling, self).__init__()
        self.Pools = nn.ModuleList(
            [nn.AdaptiveMaxPool2d((i,i)) for i in levels]
        )

    def forward(self, x):
        assert len(x.size())==4, '输入形状不满足(n,c,w,w)'
        n = x.size(0)
        c = x.size(1)
        features = []
        for pool in self.Pools:
            features.append(pool(x).view(n,c,-1))
        re = t.cat(features, dim=2).view(n,-1)
        return re

class MultiLevelPooling(nn.Module):
    def __init__(self, levels=[1,2,4]):
        super(MultiLevelPooling, self).__init__()
        self.Pools = nn.ModuleList(
            [nn.MaxPool2d(i) for i in levels]
        )

    def forward(self, x):
        assert len(x.size())==4, '输入形状不满足(n,c,w,w)'
        n = x.size(0)
        c = x.size(1)
        features = []
        for pool in self.Pools:
            features.append(pool(x))
        return features[0].view(n,c,-1)
        # re = t.cat(features, dim=2).view(n,-1)
        # return re
        # return self.Pools[0](x)

# 基于卷积神经网络的图像嵌入网络
class ConvProtoNet(nn.Module):
    def __init__(self, k):
        super(ConvProtoNet, self).__init__()

        self.ProtoNorm = None
        channels = [1,32,64,128,256]
        strides = [2,1,1,1]
        paddings = [1,1,1,2]
        layers = [get_block_2(channels[i],channels[i+1],strides[i],padding=paddings[i]) for i in range(len(strides))]
        layers.append(nn.AdaptiveMaxPool2d((1,1)))
        # layers.append(SppPooling(levels=[1,2,4]))
        self.Layers = nn.Sequential(*layers)

        if k%2==0:
            warnings.warn("K=%d是偶数将会导致feature_attention中卷积核的宽度为偶数，因此部分将会发生一些变化")
            attention_paddings = [(int((k - 1) / 2), 0), (int((k - 1) / 2 + 1), 0), (0, 0)]
        else:
            attention_paddings = [(int((k - 1) / 2), 0), (int((k - 1) / 2), 0), (0, 0)]
        attention_channels = [1,32,64,1]
        attention_strides = [(1,1),(1,1),(k,1)]
        attention_kernels = [(k,1),(k,1),(k,1)]
        relus = [True,True,False]

        self.ProtoNet = nn.Sequential(
            *[get_attention_block(attention_channels[i],
                                  attention_channels[i+1],
                                  attention_kernels[i],
                                  attention_strides[i],
                                  attention_paddings[i],
                                  relu=relus[i])
              for i in range(len(attention_channels)-1)])

    def forward(self, support, query, save_embed=False, save_proto=False):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)


        # 每一层都是以上一层的输出为输入，得到新的输出、
        # 支持集输入是N个类，每个类有K个实例
        support = self.Layers(support)

        # shape: [qk,c,w,w]
        query = self.Layers(query).view(qk,-1)

        # 计算类的原型向量
        # support shape: [n*k, c, w', w']
        # support shape: [n*k, c, w', w']
        # proto shape: [n,1,1,w,w]->[n,d]
        # d = int(support.size(2)**2*256)
        d = 1*256   # (1*1)+(2*2)+(4*4)=21, 多层次池化
        proto = self.ProtoNet(support.view(n,1,k,d)).view(n,d)

        if save_embed:
            if save_proto:
                return support.view(n, k, -1),query.view(n, int(qk/n), -1),proto
            else:
                return support.view(n, k, -1),query.view(n, int(qk/n), -1)

        proto_norm = proto.detach().norm(dim=1).sum().item()
        self.ProtoNorm = proto_norm
        support = proto

        # 将原型向量与查询集打包
        # shape: [n,d]->[qk, n, d]
        support = support.repeat((qk,1,1)).view(qk,n,-1)

        # query shape: [qk,d]->[n,qk,d]->[qk,n,d]
        query = query.repeat(n,1,1).transpose(0,1).contiguous().view(qk,n,-1)

        # 在原cos相似度的基础上添加放大系数
        scale = 10
        posterior = F.cosine_similarity(query, support, dim=2)*scale
        posterior = F.log_softmax(posterior, dim=1)

        return posterior
