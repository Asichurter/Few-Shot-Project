import torch as t
import torch.nn as nn
import torch.nn.functional as F

def get_block(in_feature, out_feature, stride=1, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(3,2,1)
    )

class ChannelProtoNet(nn.Module):
    def __init__(self, k):
        self.k = k
        channels = [k, 32, 64, 128, 256, 512]
        strides = [1,1,1,1,1]
        layers = [get_block(channels[i], channels[i+1], stride=strides[i]) for i in range(len(strides))]
        layers.append(nn.AdaptiveMaxPool2d((1,1)))
        self.Layers = nn.Sequential(*layers)

    def forward(self, support, query):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        assert k==self.k, '给定的样本中的k与定义模型时的k不同'
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        # 将支持集样本按通道连接输入网络
        support = support.view(n, k, w, w)
        query = query.view(qk, 1, w, w)
