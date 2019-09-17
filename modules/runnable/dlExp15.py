import torch as t
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt

def get_datas(c, R, num):
    datas = []
    for i in range(num):
        radius = rd.uniform(0, R)
        theta = rd.uniform(0, 2*math.pi)
        data = [c[0]+radius*math.cos(theta), c[1]+radius*math.sin(theta)]
        datas.append(data)
    return datas

def plot(p, num, title, group=3, colors=['red','blue','green']):
    plt.title(title)
    for g in range(group):
        plt.scatter([x[0] for x in p[g*num:(g+1)*num]],
                 [x[1] for x in p[g*num:(g+1)*num]],
                 marker='o',
                 color=colors[g])
    plt.show()

def get_loss(p, num, alpha=0.1, beta=0.1, margin=1):
    p = p.view(-1,num)
    p_inner_var = p.var(dim=1).sum()
    p_outer_var = p.mean(dim=1).var()
    return alpha*p_inner_var - beta*p_outer_var + margin


centers = [[0,2],[-2,0],[2,0]]
r = 3
point_num = 50
episode = 20
lr = 0.1

feature_size = [2, 16, 16, 2]
transformers = [nn.Linear(feature_size[i], feature_size[i+1]) for i in range(len(feature_size)-1)]
transformer = nn.Sequential(*transformers)

opt = SGD(transformer.parameters(), lr=lr)

for i in range(episode):
    points = []
    for c in centers:
        points += get_datas(c, r, point_num)
    # plot(points, point_num, "%dth before transformation"%(i+1), group=3)
    points = t.Tensor(points)
    points.requires_grad_(True)

    transformer.zero_grad()
    transformed_points = transformer(points)
    loss = get_loss(transformed_points, point_num)

    print(i, loss)

    loss.backward()
    opt.step()

    plot(transformed_points.detach().numpy(), point_num, '%dth transformation'%(i+1))



