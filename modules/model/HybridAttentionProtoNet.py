import torch as t
import torch.nn as nn
import math
import warnings

def get_encoder_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1, pool=2):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool)
    )

def get_attention_block(in_channel, out_channel, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.ReLU(inplace=True),
    )

def get_embed_size(in_size, depth):
    assert math.log2(in_size) >= depth, "输入:%d 对深度%d过小"%(in_size,depth)
    for i in range(depth):
        in_size  = math.floor(in_size/2)
    return in_size

class InstanceAttention(nn.Module):
    def __init__(self, linear_in, linear_out, n, k, qk):
        self.g = nn.Linear(linear_in, linear_out)
        self.n = n
        self.qk = qk
        self.k = k

    def forward(self, support, query):
        # support/query shape: [qk*n*k, d]
        k = self.k
        qk = self.qk
        n = self.n
        d = support.size(1)
        support = self.g(support)
        query = self.g(query)
        # shape: [qk, n, k, d]->[qk, n, k]->[qk, n, k, d]
        attentions = t.tanh(t.mul(support, query).view(qk, n, k, d)).sum(dim=3).squeeze()
        # shape: [qk,n,k]->[qk,n,k,d]
        attentions = t.softmax(attentions, dim=2).unsqueeze(2).repeat(1,1,1,d)

        return t.mul(attentions, support)


class HAPNet(nn.Module):
    def __init__(self, input_size, n, k, qk, encoder_depth=4):
        self.n = n
        self.k = k
        self.qk = qk


        embed_size = get_embed_size(input_size, encoder_depth)
        self.d = embed_size
        encoders = [get_encoder_block(1,64) if i==0 else get_encoder_block(64,64) for i in range(encoder_depth)]

        # 图像的嵌入结构
        # 将整个batch整体输入
        self.Encoder = nn.Sequential(*encoders)

        # 获得样例注意力的模块
        # 将嵌入后的向量拼接成单通道矩阵后，有多少个支持集就为几个batch
        if k%2==0:
            warnings.warn("K=%d是偶数将会导致feature_attention中卷积核的宽度为偶数，因此部分将会发生一些变化")
            attention_paddings = [(int((k - 1) / 2), 0), (int((k - 1) / 2 + 1), 0), (0, 0)]
        else:
            attention_paddings = [(int((k - 1) / 2), 0), (int((k - 1) / 2), 0), (0, 0)]
        attention_channels = [1,32,64,1]
        attention_strides = [(1,1),(1,1),(k,1)]
        attention_kernels = [(k,1),(k,1),(k,1)]

        self.FeatureAttention = nn.Sequential(
            *[get_attention_block(attention_channels[i],
                                  attention_channels[i+1],
                                  attention_kernels[i],
                                  attention_strides[i],
                                  attention_paddings[i])
              for i in range(len(attention_channels)-1)])


        # 获得样例注意力的模块
        # 将support重复query次，query重复n*k次，因为每个support在每个query下嵌入都不同
        self.InstanceAttention = InstanceAttention(embed_size, embed_size, n, k, qk)


    def forward(self, support, query):
        n = self.n
        k = self.k
        qk = self.qk
        d = self.d

        support = self.Encoder(support)
        query = self.Encoder(query)

        assert qk==query.size(0), "qk与实际输入的query长度不一致"

        # 将嵌入的支持集展为合适形状
        # support shape: [n,k,d]->[n,k,d]
        support = support.view(n,k,d)

        # 将支持集嵌入视为一个单通道矩阵输入到特征注意力模块中获得特征注意力
        # 并重复qk次让基于支持集的特征注意力对于qk个样本相同
        # 输入: [n,k,d]->[n,1,k,d]
        # 输出: [n,1,1,d]->[n,1,d]->[n,qk,d]->[qk,n,d]
        feature_attentions = self.FeatureAttention(support.unsqueeze(dim=1)).squeeze().repeat(qk,1,1)

        # 将支持集重复qk次，将查询集重复n*k次以获得qk*n*k长度的样本
        # 便于在获得样例注意力时，对不同的查询集有不同的样例注意力
        # 将qk，n与k均压缩到一个维度上以便输入到线性层中
        # query_expand shape:[qk,d]->[n*k,qk,d]->[qk,n,k,d]
        # support_expand shape: [n,k,d]->[qk,n,k,d]
        support_expand = support.repeat((qk,1,1,1)).view(qk*n*k,d)
        query_expand = query.repeat((n*k,1,1)).transpose(0,1).contiguous().view(qk*n*k,d)

        # 利用样例注意力注意力对齐支持集样本
        # shape: [qk,n,k,d]
        support = self.InstanceAttention(support_expand, query_expand)

        # 生成对于每一个qk都不同的类原型向量
        # 注意力对齐以后，将同一类内部的加权的向量相加以后
        # proto shape: [qk,n,k,d]->[qk,n,d]
        support = support.sum(dim=2).squeeze()

        # query shape: [qk,d]->[qk,n,d]
        query = query.unsqueeze(dim=1).repeat(1,n,1)

        # dis_attented shape: [qk,n,d]->[qk,n,d]->[qk,n]
        dis_attented = (((support-query)**2)*feature_attentions).sum(dim=2).neg()

        return t.softmax(dis_attented, dim=1)








