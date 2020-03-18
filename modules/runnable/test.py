import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import random as rd
import torchvision.transforms as T
import torch as t
import numpy as np
import math
import matplotlib.pyplot as plt
import collections
from operator import itemgetter
import json

from modules.utils.datasets import ImagePatchDataset

rnn = nn.GRU(10, 20, 2, batch_first=True, bidirectional=False)
input = t.randn(3, 7, 10)   # seq len, batch len, feature dim
h0 = t.randn(2, 3, 20)
output, hn = rnn(input, h0)










































