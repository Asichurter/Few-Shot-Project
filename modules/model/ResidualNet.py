import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        # self.Layer1 = nn.Sequential(
        #     nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding=int((kernel_size-1)/2), bias=False),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True))
        self.Layer1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channel))
        self.Pool = nn.MaxPool2d(2)
        self.trans = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channel)) if in_channel!=out_channel else None

    def forward(self, x):
        left = self.Layer1(x)
        # left = self.Layer2(x)
        x = self.trans(x) if self.trans is not None else x
        x = x + left
        x = self.Pool(x)
        return F.relu(x)

class ResidualNet(nn.Module):
    def __init__(self, input_size, n, k, qk, channel=64, block_num=4, metric="Proto", **kwargs):
        super(ResidualNet, self).__init__()
        assert metric in ["Siamese","Proto","Relation"]
        self.metric = metric
        pars = {}
        pars['n'] = n
        pars['k'] = k
        pars['qk'] = qk
        pars['channel'] = channel
        self.pars = pars
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1,channel,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.Layer2 = nn.Sequential()
        for i in range(block_num):
            self.Layer2.add_module('block%d'%(i+1),ResidualBlock(channel,channel,kernel_size=3,stride=1))


        conv_out = input_size/4/(2**block_num)
        out_size = int(channel*conv_out*conv_out)

        if metric=="Siamese":
            # shape: [d,1]
            self.fc1 = nn.Linear(out_size,1)
        elif metric=="Relation":
            # shape: [d,hidden]
            self.relation = ResidualBlock(channel*2,channel)
            out_size = int(out_size/(2*2))
            self.fc1 = nn.Linear(out_size,kwargs['hidden_size'])
            self.fc2 = nn.Linear(kwargs['hidden_size'],1)

    def forward(self, *x):
        n = self.pars['n']
        k = self.pars['k']
        qk = self.pars['qk']
        channel = self.pars['channel']

        support = self.Layer1(x[0])
        support = self.Layer2(support)

        query = self.Layer1(x[1])
        query = self.Layer2(query)
        query_size = query.size(0)


        if self.metric == "Siamese":
            # shape: [qk,n,k,d]
            support = support.view(n,k,-1).repeat(query_size,1,1,1)
            query = query.view(qk,-1).repeat(k*n,1,1).transpose(0,1).contiguous().view(query_size,n,k,-1)
        elif self.metric=="Proto":
            # shape: [n,k,d]->[qk,n,d]
            support = support.view(n,k,-1).sum(dim=1).div(k).squeeze(1).repeat(query_size,1,1)
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

