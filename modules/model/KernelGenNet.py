import torch as t
import torch.nn as nn
import torch.nn.functional as F

def get_block(in_feature, out_feature, stride=1, relu=True):
    components = [nn.Conv2d(in_feature, out_feature,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False), nn.BatchNorm2d(out_feature)]
    if relu:
        components.append(nn.ReLU(inplace=True))
    components.append(nn.MaxPool2d(3,2,1))
    return nn.Sequential(*components)

class KernelGenNet(nn.Module):
    def __init__(self, kernel_size=3, channel=64):
        super(KernelGenNet, self).__init__()
        channels = [1,64,64,64,64]
        strides = [2,2,1,1]
        self.Generator = [get_block(channel[i],channel[i+1],strides[i],i<3) for i in range(len(strides))]
        self.Channel = channel

    def forward(self, support, query):
        # support shape: [n,k,c,w,w]
        assert len(support.size())==5 and len(query.size())==4,\
                    "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"

        n = support.size(0)
        k = support.size(1)
        w = support.size(3)
        query_length = query.size(0)
        channel = self.Channel
        kernel_size = w
        for i in range(4):
            if i < 2:
                kernel_size = kernel_size//4
            else:
                kernel_size = kernel_size//2

        # shape: []
        kernels = self.Generator(support.view(n*k,1,w,w)).view(n,k,channel,kernel_size,kernel_size).mean(dim=1)


