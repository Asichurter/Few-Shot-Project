import torch as t
import torch.nn as nn
import math

def get_encoder_block(in_channel, out_channel, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

def get_embed_size(in_size, depth):


class HAPNet(nn.Module):
    def __init__(self, input_size, n, k, qk, encoder_depth=4):
        encoders = [get_encoder_block(input_size/(2**i),
                                      input_size/(2**(i+1))) for i in range(encoder_depth)]
        self.Encoder = nn.Sequential(*encoders)
        self.InstanceAttention = nn.Sequential()