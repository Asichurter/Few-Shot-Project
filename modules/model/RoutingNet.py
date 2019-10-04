import torch as t
import torch.nn as nn
import torch.nn.functional as F

def get_block_1(in_feature, out_feature, stride=1, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

def get_block_2(in_feature, out_feature, stride=1, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(3,2,1)
    )

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

class RoutingNet(nn.Module):
    def __init__(self, metric="SqEuc", iters=3):
        super(RoutingNet, self).__init__()

        self.metric = metric
        self.ProtoNorm = None
        channels = [1,16,32,64,128,256]
        strides = [2,1,1,1,1]
        layers = [get_block_2(channels[i],channels[i+1],strides[i]) for i in range(len(strides))]
        layers.append(nn.AdaptiveMaxPool2d((1,1)))
        self.Layers = nn.Sequential(*layers)

        self.Transformer = nn.Linear(channels[-1], channels[-1])
        self.Iters = iters

    def forward(self, support, query, save_embed=False, save_proto=False):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)

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

        def dynamic_routing(e, b):
            dim = e.size(2)
            # 先利用转换矩阵转换特征
            # e shape: [n,k,d]
            e = self.Transformer(e)
            # d shape: [n,k]->[n,k,d]
            # b shape: [n,k]
            d = t.softmax(b, dim=1).unsqueeze(dim=2).repeat((1,1,dim))

            # c shape: [n,k,d]->[n,d]
            c = (d*e).sum(dim=1)
            c_norm = c.norm(dim=1)

            # squashing
            coef = ((c_norm**2)/(c_norm**2+1)/c_norm).unsqueeze(dim=1).repeat((1,dim))
            c = c*coef

            # 更新b
            # [n,d]->[n,k,d]
            c_expand = c.unsqueeze(dim=1).repeat((1,k,1))
            delta_b = (c_expand*e).sum(dim=2)

            return b + delta_b, c

        support = self.Layers(support).squeeze()
        query = self.Layers(query).squeeze()

        # 计算类的原型向量
        # shape: [n, k, d]
        support = support.view(n, k, -1)
        d = support.size(2)

        # coupling shape: [n, d]
        coupling = t.zeros_like(support).sum(dim=2)
        proto = None
        # 使用动态路由来计算原型向量
        for i in range(self.Iters):
            coupling, proto = dynamic_routing(support, coupling)

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
