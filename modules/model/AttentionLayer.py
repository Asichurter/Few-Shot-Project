import torch as t
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, channel_size, hidden_layer_num=1):
        layer_1 = nn.Sequential()
        for i in range(hidden_layer_num):
            layer_1.add_module('conv%d'%i,nn.Sequential(
                nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                nn.ReLU(inplace=True)
            ))
        layer_2 = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_size)
        )
        self.Channel = channel_size
        self.Layer_1 = layer_1
        self.Layer_2 = layer_2

    def forward(self, *x):
        # x shape: [2, batch, channel, w, h]->[batch, 2*channel, w, h]
        maps = t.cat((x[0],x[1]), dim=1)

        maps = self.Layer_1(maps)
        maps = self.Layer_2(maps)

        # 将注意力系数归一化
        maps = t.sigmoid(maps)

        # x shape: [batch, 2*channel, w, h]
        return t.mul(maps[:,:self.Channel,:,:], x[0]),\
               t.mul(maps[:,self.Channel:,:,:], x[1])



