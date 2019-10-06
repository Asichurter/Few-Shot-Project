import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_
import numpy as np

from modules.model.AttentionLayer import AttentionLayer

from sklearn.manifold import MDS, t_sne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
        self.Pool = nn.MaxPool2d(3,2,1) if not keep_dim else None
        self.trans = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channel)) if in_channel!=out_channel else None
        self.NonLinear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        left = self.Layer1(x)
        # left = self.Layer2(x)
        x = self.trans(x) if self.trans is not None else x
        x = x + left
        x = self.Pool(x) if self.Pool is not None else x
        return self.NonLinear(x)

class ResidualNet(nn.Module):
    def __init__(self, input_size, block_num=4, metric="Proto", **kwargs):
        super(ResidualNet, self).__init__()
        assert metric in ["Siamese","Proto","Relation"]
        self.metric = metric
        pars = {}
        pars['block_num'] = block_num
        self.pars = pars
        self.forward_inner_var = None
        self.forward_outer_var = None

        channels = [1,32,64,128,256,512]
        self.Layer1 = nn.Sequential(
            nn.Conv2d(channels[0],channels[1],kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        max_pool_num = int(math.log2(int(input_size / 4)))
        self.Layer2 = nn.Sequential()
        for i in range(block_num):
            # 在block的数量大于最大池化次数时，设定前k个block不使用最大池化
            self.Layer2.add_module('block%d'%(i+1),ResidualBlock(channels[i+1],channels[i+2],kernel_size=3,stride=1,
                                                                 keep_dim=(block_num-i)>max_pool_num))

        conv_out = input_size/4/(2**block_num)
        out_size = int(channels[-1]*conv_out*conv_out)

        if metric=="Siamese":
            # shape: [d,1]
            self.fc1 = nn.Linear(out_size,1)
        elif metric=="Relation":
            # shape: [d,hidden]
            self.relation = ResidualBlock(channels[-1]*2,channels[-1])
            out_size = int(64/(2**block_num)*channels[-1])
            self.fc1 = nn.Linear(out_size,kwargs['hidden_size'])
            self.fc2 = nn.Linear(kwargs['hidden_size'],1)

    def forward(self, support, query):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = query.view(qk, 1, w, w)
        # channel = self.pars['channel']
        # block_num = self.pars['block_num']

        support = self.Layer1(support)
        query = self.Layer1(query)
        support = self.Layer2(support)
        query = self.Layer2(query)

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

        if self.metric == "Siamese":
            # shape: [qk,n,k,d]
            support = support.view(n,k,-1).repeat(qk,1,1,1)
            query = query.view(qk,-1).repeat(k*n,1,1).transpose(0,1).contiguous().view(query_size,n,k,-1)
        elif self.metric=="Proto":
            # support = support.view(support_size,-1)
            # query = query.view(query_size,-1)

            # 新增的feature转换matrix
            # support = self.Transformer(support)
            # query = self.Transformer(query)

            # shape: [b,d]->[n,k,d]
            support = support.view(n,k,-1)
            # self.forward_inner_var = support.var(dim=1).sum()
            #
            # support = support.mean(dim=1).squeeze()
            #
            # self.forward_outer_var = support.var(dim=0).sum()

            support = proto_mean(support)

            # shape: [n,d]->[qk,n,d]
            support = support.repeat(qk,1,1)
            # shape: [qk,d]->[qk,n,d]
            query = query.view(qk,-1).repeat(n,1,1).transpose(0,1).contiguous()
        elif self.metric == "Relation":
            # shape: [n*k,channel,d,d]->[qk,n,channel,d,d]
            d = support.size(2)
            support = support.view(n,k,512,d,d).sum(dim=1).div(k).squeeze(1).repeat(qk,1,1,1,1)

            # shape: [qk,channel,d,d]->[qk,n,channel,d,d]
            query = query.view(qk,512,d,d).repeat(n,1,1,1,1).transpose(0,1)

            relation_input = t.cat((support,query), dim=2).view(-1,512*2,d,d)

        if self.metric == "Siamese":
            out = t.abs(support-query)
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
            relations = relations.view(qk,n)
            return t.sigmoid(relations)

    def proto_embed_reduction(self, support, query, metric="MDS"):
        assert self.metric == "Proto"

        n = self.pars['n']
        k = self.pars['k']
        qk = self.pars['qk']

        support = self.Layer1(support)
        support = self.Layer2(support)
        support = support.view(support.size(0),-1)
        
        query = self.Layer1(query)
        query = self.Layer2(query)
        query = query.view(query.size(0),-1)

        support_size = support.size(0)

        merge = t.cat((support,query), dim=0).cpu().detach().numpy()
        support_np = support.cpu().detach().numpy()
        support_labels = np.array([[i for j in range(k)] for i in range(n)]).reshape(-1)

        if metric == "MDS":
            reducer = MDS(n_components=2, verbose=True)
        elif metric == "tSNE":
            reducer = t_sne.TSNE(n_components=2)
        elif metric == 'LDA':
            reducer = LinearDiscriminantAnalysis(n_components=2)
            transformed = reducer.fit_transform(support_np, support_labels)
            return transformed
        else:
            assert False, "无效的metric"
        merge_transformed = reducer.fit_transform(merge)

        support = merge_transformed[:support_size]
        query = merge_transformed[support_size:]

        support_center = support.reshape((n,k,-1)).sum(axis=1)
        support_center = support_center/k
        support_center = support_center.reshape((n, -1))

        return support,query,support_center

