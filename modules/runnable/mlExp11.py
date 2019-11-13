import matplotlib.pyplot as plt
import os
import numpy as np
import random as rd
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modules.utils.extract import extract_infos

n = 20
N = 20

path = 'D:/peimages/PEs/virusshare/'

samples = []
colors = []

colors_ = ['black', #1
          'darkgray',  #2
          'gainsboro',  #3
          'rosybrown',  #4
          'lightcoral',  #5
          "indianred",  #6
          'firebrick',  #7
          'maroon',  #8
          'red',  #9
          'coral',  #10
          'orangered',  #11
          'sienna',  #12
          'chocolate',  #13
          'sandybrown',  #14
          'bisque',  #15
          'wheat',  #16
          'gold',  #17
          'khaki',  #18
          'olive',  #19
          'lightyellow',  #20
          'yellow',  #21
          'olivedrab',  #22
          'yellowgreen',  #23
          'darkseagreen',  #24
          'palegreen',  #25
          'darkgreen',  #26
          'lime',  #27
          'mediumseagreen',  #28
          'aquamarine',  #29
          'turquoise',  #30
          'lightseagreen',  #31
          'lightcyan',  #32
          'teal',  #33
          'lightblue',  #34
          'deepskyblue',  #35
          'steelblue',  #36
          'navy',  #37
          'blue',  #38
          'slateblue',  #39
          'darkslateblue',  #40
          'mediumpurple',  #41
          'blueviolet',  #42
          'plum',  #43
          'violet',  #44
          'purple',  #45
          'magenta',  #46
          'mediumvioletred',  #47
          'hotpink',  #48
          'crimson',  #49
          'pink',  #50
          ]

color_pool = rd.sample(colors_, n)

cls_names = os.listdir(path)

rd.seed(time()%7356550)
sampled_cls = rd.sample(cls_names, n)

for i, cls in enumerate(sampled_cls):
    print(i,cls)
    rd.seed(time()%7356550)

    all_insts = os.listdir(path+cls+'/')
    assert len(all_insts) >= N, '类 %s 中的总数量不及%d采样数量%d'%(cls, len(all_insts), N)
    insts = rd.sample(all_insts, N)

    for inst in insts:
        sample = extract_infos(path+cls+'/'+inst)
        if sample is None:
            continue
        samples.append(sample)
        colors.append(colors_[i])

scaler = StandardScaler()
samples = scaler.fit_transform(samples)
pca = PCA(n_components=2)
samples = pca.fit_transform(np.array(samples))

plt.figure(figsize=(10,8))
for x,c in zip(samples, colors):
    plt.scatter([x[0]], [x[1]], marker='o', color=c)
plt.axis('off')
plt.show()

