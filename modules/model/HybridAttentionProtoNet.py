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

def get_encoder_block_2(in_channel, out_channel, kernel_size=3, stride=1, padding=1, pool=2):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(3,pool,1)
    )

def get_attention_block(in_channel, out_channel, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        # nn.ReLU(inplace=True)
        nn.LeakyReLU(inplace=True),
    )

def get_embed_size(in_size, depth):
    assert math.log2(in_size) >= depth, "输入:%d 对深度%d过小"%(in_size,depth)
    for i in range(depth):
        in_size  = in_size//2
    return in_size//2

class InstanceAttention(nn.Module):
    def __init__(self, linear_in, linear_out):
        super(InstanceAttention, self).__init__()
        self.g = nn.Linear(linear_in, linear_out)

    def forward(self, support, query, k, qk, n):
        # support/query shape: [qk*n*k, d]
        d = support.size(1)
        support = self.g(support)
        query = self.g(query)
        # shape: [qk, n, k, d]->[qk, n, k]
        attentions = t.tanh((support*query).view(qk, n, k, d)).sum(dim=3).squeeze()
        # shape: [qk,n,k]->[qk,n,k,d]
        attentions = t.softmax(attentions, dim=2).unsqueeze(3).repeat(1,1,1,d)

        return t.mul(attentions, support.view(qk, n, k, d))


class HAPNet(nn.Module):
    def __init__(self, n, k, qk, encoder_depth=4):
        super(HAPNet, self).__init__()
        self.n = n
        self.k = k
        self.qk = qk

        embed_size = 256#get_embed_size(input_size, encoder_depth)
        self.d = embed_size
        # self.d = embed_size
        channels = [1,32,64,128,256]
        strides = [2,2,2,2]
        encoders = [get_encoder_block(channels[i],channels[i+1],stride=strides[i]) for i in range(encoder_depth)]

        # 图像的嵌入结构
        # 将整个batch整体输入
        self.Encoder = nn.Sequential(*encoders)

        # 获得样例注意力的模块
        # 将嵌入后的向量拼接成单通道矩阵后，有多少个支持集就为几个batch
        if k%2==0:
            warnings.warn("K=%d是偶数将会导致feature_attention中卷积核的宽度为偶数，因此部分将会发生一些变化")
            attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
            # attention_paddings = [(int((k - 1) / 2), 0), (int((k - 1) / 2 + 1), 0), (0, 0)]
        else:
            attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
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
        self.InstanceAttention = InstanceAttention(embed_size, embed_size)


    def forward(self, support, query):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)

        support = self.Encoder(support).squeeze()
        query = self.Encoder(query).squeeze()

        # 将嵌入的支持集展为合适形状
        # support shape: [n,k,d]->[n,k,d]
        support = support.view(n,k,-1)
        # query shape: [qk, d]
        query = query.view(qk,-1)

        # 将支持集嵌入视为一个单通道矩阵输入到特征注意力模块中获得特征注意力
        # 并重复qk次让基于支持集的特征注意力对于qk个样本相同
        # 输入: [n,k,d]->[n,1,k,d]
        # 输出: [n,1,1,d]->[n,d]->[qk,n,d]
        feature_attentions = self.FeatureAttention(support.unsqueeze(dim=1)).squeeze().repeat(qk,1,1)

        # 将支持集重复qk次，将查询集重复n*k次以获得qk*n*k长度的样本
        # 便于在获得样例注意力时，对不同的查询集有不同的样例注意力
        # 将qk，n与k均压缩到一个维度上以便输入到线性层中
        # query_expand shape:[qk,d]->[n*k,qk,d]->[qk,n,k,d]
        # support_expand shape: [n,k,d]->[qk,n,k,d]
        support_expand = support.repeat((qk,1,1,1)).view(qk*n*k,-1)
        query_expand = query.repeat((n*k,1,1)).transpose(0,1).contiguous().view(qk*n*k,-1)

        # 利用样例注意力注意力对齐支持集样本
        # shape: [qk,n,k,d]
        support = self.InstanceAttention(support_expand, query_expand, k, qk, n)

        # 生成对于每一个qk都不同的类原型向量
        # 注意力对齐以后，将同一类内部的加权的向量相加以后
        # proto shape: [qk,n,k,d]->[qk,n,d]
        support = support.sum(dim=2).squeeze()
        # support = support.mean(dim=1).repeat((qk,1,1)).view(qk,n,-1)

        # query shape: [qk,d]->[qk,n,d]
        query = query.unsqueeze(dim=1).repeat(1,n,1)

        # dis_attented shape: [qk*n,n,d]->[qk*n,n,d]->[qk*n,n]
        # dis_attented = (((support-query)**2)).sum(dim=2).neg()
        dis_attented = (((support-query)**2)*feature_attentions).sum(dim=2).neg()

        return t.log_softmax(dis_attented, dim=1)








