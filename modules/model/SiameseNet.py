import torch as t
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self, input_size, k, n):
        super(SiameseNet, self).__init__()
        self.k = k
        self.n = n
        linear_input_size = int(((input_size/4/2/2/2)**2)*64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # self.fc = nn.Linear(linear_input_size, 1)

    def forward(self, *x):
        n = self.n
        k = self.k
        query_size = x[1].size(0)
        support_out = self.layer1(x[0])
        support_out = self.layer2(support_out)
        support_out = self.layer3(support_out)
        support_out = self.layer4(support_out)
        support_out = support_out.view(n,k,-1)
        # 还是按照之前的实现，求出类别向量代表整个类
        # 可以考虑替换该实现
        # support_out = t.sum(support_out, dim=1).div().squeeze(1)

        # shape: [qk,n,k,d]
        support_out = support_out.repeat(query_size,1,1,1)

        
        query_out = self.layer1(x[1])
        query_out = self.layer2(query_out)
        query_out = self.layer3(query_out)
        query_out = self.layer4(query_out)
        query_out = query_out.view(query_size,-1)

        # support_out = support_out.repeat(query_size,1,1)
        #query_out = query_out.repeat(n,1,1).transpose(0,1).contiguous()

        # shape:
        query_out = query_out.repeat(k*n,1,1).transpose(0,1).view(query_size,n,k,-1)

        # shape: [qk,n]
        merge = t.abs(support_out-query_out).sum(dim=2).sum(dim=2).div(k)
        # 负距离作为概率输入到softmax
        merge = F.softmax(merge.neg(), dim=1)

        # merge = t.abs(support_out-query_out)
        # merge = self.fc(merge)
        return merge

