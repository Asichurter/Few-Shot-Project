import torch as t
import torch.nn as nn
import torch.nn.functional as F
import time

def get_block(in_feature, out_feature, stride=1, relu=True, bn=True, pool=2):
    components = [nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=stride, padding=1, bias=False)]
    if bn:
        components.append(nn.BatchNorm2d(out_feature))
    if relu:
        components.append(nn.ReLU(inplace=True))
    if pool is not None:
        components.append(nn.MaxPool2d(3,pool,1))

    return nn.Sequential(*components)



class BiEmbedProtoNet(nn.Module):
    def __init__(self, induction='mean', in_channel=1):
        super(BiEmbedProtoNet, self).__init__()
        strides = [2,1,1,1]
        channels = [in_channel,64,64,64,64]
        encoder = [get_block(channels[i], channels[i+1], stride=strides[i]) for i in range(len(strides))]
        self.Encoder = nn.Sequential(*encoder)

        relus = [True, True, True, False]
        pools = [None, 2, 2, 2]
        embedder = [get_block(64, 64, relu=relus[i], pool=pools[i]) for i in range(len(pools))]
        self.Embedder = nn.Sequential(*embedder)

        self.Induction = induction

    def forward(self, support, query):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)

        support = self.Encoder(support).view(n,k,64,7,7)
        query = self.Encoder(query)

        # input shape: [n, k, c, w, w]->[n,c,w,w]
        def proto_mean(tensors):
            return tensors.mean(dim=1).squeeze()

        # 待完成...
        # def proto_correct_attention(tensors):
        #     # 利用类内向量的均值向量作为键使用注意力机制生成类向量
        #     # 类均值向量
        #     # centers shape: [n,k,c,w,w]->[n,c,w,w]->[n,k,c,w,w]
        #     support_center = tensors.mean(dim=1).unsqueeze(dim=1).repeat((1,k,1,1,1))#.repeat(1, k).reshape(n, k, -1)
        #
        #     # -------------------------------------------------------------
        #     # 支持集与均值向量的欧式平方距离
        #     # dis shape: [n,k,d]->[n,k]
        #     support_dis = ((tensors - support_center) ** 2).sum(dim=2).sqrt()
        #     # dis_mean_shape: [n,k]->[n]->[n,k]
        #     support_dis_mean = support_dis.mean(dim=1).unsqueeze(dim=1).repeat(1,k)
        #     support_dis = t.abs(support_dis-support_dis_mean).neg()
        #     # attention shape: [n,k]->[n,k,d]
        #     attention_map = t.softmax(support_dis, dim=1).unsqueeze(dim=2).repeat(1,1,d)
        #
        #     # return shape: [n,k,d]->[n,d]
        #     return t.mul(tensors, attention_map).sum(dim=1).squeeze()
            # -------------------------------------------------------------

        # 待完成...
        # def proto_attention(tensors):
        #     # 利用类内向量的均值向量作为键使用注意力机制生成类向量
        #     # 类均值向量
        #     # centers shape: [n,k,d]->[n,d]->[n,k,d]
        #     support_center = tensors.mean(dim=1).repeat(1, k).reshape(n, k, -1)
        #
        #     # attention shape: [n,k,d]
        #     # 类内对每个向量的注意力映射，由负距离输入到softmax生成
        #     attention_map = t.softmax(((tensors - support_center) ** 2).sum(dim=2).neg(), dim=1)
        #     # [n,k]->[n,k,d]
        #     # 将向量的注意力系数重复到每个位置
        #     attention_map = attention_map.unsqueeze(dim=2).repeat(1, 1, d)
        #     # support: [n,k,d]->[n,d]
        #     # 注意力映射后的支持集中心向量
        #     return t.mul(tensors, attention_map).sum(dim=1).squeeze()

        # shape: [n,c,w,w]
        support = proto_mean(support)

        support = self.Embedder(support)
        support = support.view(n,64)
        query = self.Embedder(query).view(qk,64)

        # support shape: [n,d]->[qk,n,d]
        support = support.repeat((qk,1,1))
        # query shape: [qk,d]->[qk,n,d]
        query = query.unsqueeze(dim=1).repeat((1,n,1))

        # return shape: [qk, n]
        return F.log_softmax(((support - query) ** 2).sum(dim=2).neg(), dim=1)






