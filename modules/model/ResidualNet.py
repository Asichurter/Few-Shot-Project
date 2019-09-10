import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_
import numpy as np

from modules.model.AttentionLayer import AttentionLayer

from sklearn.manifold import MDS, t_sne

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, keep_dim=False):
        super(ResidualBlock, self).__init__()
        # self.Layer1 = nn.Sequential(
        #     nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding=int((kernel_size-1)/2), bias=False),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True))
        self.Layer1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channel))
        self.Pool = nn.MaxPool2d(2) if not keep_dim else None
        self.trans = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channel)) if in_channel!=out_channel else None

    def forward(self, x):
        left = self.Layer1(x)
        # left = self.Layer2(x)
        x = self.trans(x) if self.trans is not None else x
        x = x + left
        x = self.Pool(x) if self.Pool is not None else x
        return F.relu(x)

class ResidualNet(nn.Module):
    def __init__(self, input_size, n, k, qk, channel=64, block_num=4, metric="Proto",
                 attention=False, attention_num = 1,
                 **kwargs):
        super(ResidualNet, self).__init__()
        assert metric in ["Siamese","Proto","Relation"]
        self.metric = metric
        pars = {}
        pars['n'] = n
        pars['k'] = k
        pars['qk'] = qk
        pars['channel'] = channel
        pars['block_num'] = block_num
        self.pars = pars
        self.forward_inner_var = None
        self.forward_outer_var = None
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1,channel,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        max_pool_num = int(math.log2(int(input_size / 4)))
        # self.Layer2 = nn.ModuleList([ResidualBlock(channel,
        #                                            channel,
        #                                            kernel_size=3,
        #                                            stride=1,
        #                                            keep_dim=(block_num-i)>max_pool_num) for i in range(block_num)])

        # 由于存在一个前置的block用于快速缩小图片尺寸，因此注意力映射要比block的数量多一个
        # assert not attention or attention_num is not None, "注意力映射数量不能为None"
        # self.Attentions = nn.ModuleList([AttentionLayer(2*channel,attention_num) for i in range(block_num+1)]) \
        #                                                                             if attention else None
        self.Layer2 = nn.Sequential()
        for i in range(block_num):
            # 在block的数量大于最大池化次数时，设定前k个block不使用最大池化
            self.Layer2.add_module('block%d'%(i+1),ResidualBlock(channel,channel,kernel_size=3,stride=1,
                                                                 keep_dim=(block_num-i)>max_pool_num))

        # 用于接收类内向量生成类向量的转换器
        # shape: [n,k,d]->[n,d]
        # self.Transformer = nn.Sequential(
        #                         nn.Conv2d(k,k,kernel_size=3,padding=1,bias=False),
        #                         nn.BatchNorm2d(k),
        #                         nn.ReLU(inplace=True),
        #                         nn.Conv2d(k, 1, kernel_size=3, padding=1),
        # )

        conv_out = input_size/4/(2**block_num)
        out_size = int(channel*conv_out*conv_out)

        if metric=="Siamese":
            # shape: [d,1]
            self.fc1 = nn.Linear(out_size,1)
        elif metric=="Relation":
            # shape: [d,hidden]
            self.relation = ResidualBlock(channel*2,channel)
            out_size = int(64/(2**block_num)*channel)
            self.fc1 = nn.Linear(out_size,kwargs['hidden_size'])
            self.fc2 = nn.Linear(kwargs['hidden_size'],1)

    def forward(self, *x):
        n = self.pars['n']
        k = self.pars['k']
        qk = self.pars['qk']
        channel = self.pars['channel']
        block_num = self.pars['block_num']

        support = self.Layer1(x[0])
        query = self.Layer1(x[1])
        support = self.Layer2(support)
        query = self.Layer2(query)
        # support,query = self.Attentions[0](support,query) \
        #                     if self.Attentions is not None else support,query
        # support = self.Layer2(support)
        # for i in range(block_num):
        #     support = self.Layer2[i](support)
        #     query = self.Layer2[i](query)
            # 注意力模块的下标比block多1
            # support,query = self.Attentions[i+1](support,query) \
            #     if self.Attentions is not None else support,query

        # query = self.Layer1(x[1])
        # query = self.Layer2(query)

        query_size = query.size(0)
        support_size = support.size(0)

        if self.metric == "Siamese":
            # shape: [qk,n,k,d]
            support = support.view(n,k,-1).repeat(query_size,1,1,1)
            query = query.view(qk,-1).repeat(k*n,1,1).transpose(0,1).contiguous().view(query_size,n,k,-1)
        elif self.metric=="Proto":
            support = support.view(support_size,-1)
            query = query.view(query_size,-1)

            # 新增的feature转换matrix
            # support = self.Transformer(support)
            # query = self.Transformer(query)

            # shape: [b,d]->[n,k,d]
            support = support.view(n,k,-1)
            self.forward_inner_var = support.var(dim=1).sum()

            # 利用类内向量的均值向量作为键使用注意力机制生成类向量
            # 类均值向量
            # centers shape: [n,k,d]->[n,d]->[n,k,d]
            support_center = support.sum(dim=1).div(k).repeat(1,k).reshape(n,k,-1)
            # 支持集与均值向量的欧式平方距离
            # dis shape: [n,k,d]->[n]->[n,k]
            support_dis_sum = ((support-support_center)**2).sum(dim=2).sum(dim=1).unsqueeze(dim=1).repeat(1,k)
            # attention shape: [n,k,d]
            # 类内对每个向量的注意力映射，由负距离输入到softmax生成
            d = support.size(2)
            attention_map = t.softmax((((support-support_center)**2).sum(dim=2)/support_dis_sum).neg(), dim=1)
            # [n,k]->[n,k,d]
            # 将向量的注意力系数重复到每个位置
            attention_map = attention_map.unsqueeze(dim=2).repeat(1,1,d)
            # support: [n,k,d]->[n,d]
            # 注意力映射后的支持集中心向量
            support = t.mul(support, attention_map).sum(dim=1).squeeze()

            # support = support.sum(dim=1).div(k).squeeze(1)

            self.forward_outer_var = support.var(dim=0).sum()
            # shape: [n,k,d]->[qk,n,d]
            support = support.repeat(query_size,1,1)
            # shape: [qk,d]->[qk,n,d]
            query = query.view(query_size,-1).repeat(n,1,1).transpose(0,1)
        elif self.metric == "Relation":
            # shape: [n*k,channel,d,d]->[qk,n,channel,d,d]
            d = support.size(2)
            support = support.view(n,k,channel,d,d).sum(dim=1).div(k).squeeze(1).repeat(query_size,1,1,1,1)

            # shape: [qk,channel,d,d]->[qk,n,channel,d,d]
            query = query.view(query_size,channel,d,d).repeat(n,1,1,1,1).transpose(0,1)

            relation_input = t.cat((support,query), dim=2).view(-1,channel*2,d,d)

        if self.metric == "Siamese":
            out = t.abs(support-query_size)
            # shape: [qk,n,k]
            out = self.fc1(out)
            # shape: [qk,n]
            out = out.sum(dim=2).div(k).squeeze(2)
            return F.log_softmax(out, dim=1)
        elif self.metric == "Proto":
            out = ((support-query)**2).sum(dim=2)
            # 由于使用负对数损失函数，因此需要使用log_softmax
            return F.log_softmax(out.neg(), dim=1)
        elif self.metric == "Relation":
            relations = self.relation(relation_input).view(relation_input.size(0),-1)
            relations = self.fc1(relations)
            relations = self.fc2(relations)
            relations = relations.view(query_size,n)
            return t.sigmoid(relations)

    def proto_embed_reduction(self, support, query, metric="MDS"):
        assert self.metric == "Proto"

        n = self.pars['n']
        k = self.pars['k']

        support = self.Layer1(support)
        support = self.Layer2(support)
        support = support.view(support.size(0),-1)
        
        query = self.Layer1(query)
        query = self.Layer2(query)
        query = query.view(query.size(0),-1)

        support_size = support.size(0)

        merge = t.cat((support,query), dim=0).cpu().detach().numpy()

        if metric == "MDS":
            reducer = MDS(n_components=2, verbose=True)
        elif metric == "tSNE":
            reducer = t_sne.TSNE(n_components=2)
        else:
            assert False, "无效的metric"
        merge_transformed = reducer.fit_transform(merge)

        support = merge_transformed[:support_size]
        query = merge_transformed[support_size:]

        support_center = support.reshape((n,k,-1)).sum(axis=1)
        support_center = support_center/k
        support_center = support_center.reshape((n, -1))

        return support,query,support_center

