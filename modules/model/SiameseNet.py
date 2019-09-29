import torch as t
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self, k, n):
        super(SiameseNet, self).__init__()
        self.k = k
        self.n = n
        linear_input_size = 7*7*256
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(linear_input_size, 1)

    def forward(self, support, query):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, 1, w, w)
        query = support.view(qk, 1, w, w)

        support = self.layer1(support)
        support = self.layer2(support)
        support = self.layer3(support)
        support = self.layer4(support)
        support = support.view(n,k,-1)

        support = support.mean(dim=1)

        # shape: [qk,n,d]
        support = support.repeat(qk,1,1)

        
        query = self.layer1(query)
        query= self.layer2(query)
        query = self.layer3(query)
        query = self.layer4(query)

        # shape:
        query = query.repeat(k*n,1,1).transpose(0,1).contiguous().view(qk,n,k,-1)

        # shape: [qk,n,d]
        merge = self.fc(t.abs(support-query).view(qk*n,-1)).view(qk,n)
        # 负距离作为概率输入到softmax
        merge = t.sigmoid(merge)

        return merge

