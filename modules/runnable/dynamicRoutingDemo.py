import torch as t
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random as rd

def dynamic_routing(e, b):
    dim = e.size(2)
    k = e.size(1)
    d = t.softmax(b, dim=1).unsqueeze(dim=2).repeat((1,1,dim))
    print('d:', t.softmax(b, dim=1).squeeze())
    c = (d*e).sum(dim=1)
    c_norm = c.norm(dim=1)
    coef = ((c_norm**2)/(1+c_norm**2)/c_norm).unsqueeze(dim=1).repeat((1,dim))
    c = c*coef

    # c shape: [n,d]->[n,k,d]
    c_expand = c.unsqueeze(dim=1).repeat((1,k,1))

    delta_b = (c_expand*e).sum(dim=2)

    next_b = b + delta_b
    normal_next_b = t.softmax(next_b, dim=1).squeeze()
    coupling_hist.append(normal_next_b.tolist())
    print('next b',next_b)

    return next_b, c

regions = [[-1,1],[1,1],[-1,-1],[1,-1],[-1,1],[1,1],[-1,-1],[1,-1]]
outlier_region = [0,4]
all_region = regions.append(outlier_region)
points = []
for r in regions:
    x = rd.random()*2+r[0]
    y = rd.random()*2+r[1]
    points.append([x,y])
points = t.Tensor([points])

iters = 5
protos = []
coupling_hist = []
coupling = t.zeros_like(points).sum(dim=2)
for i in range(iters):
    coupling, proto = dynamic_routing(points, coupling)
    protos.append(proto.squeeze().tolist())


points = points.squeeze().tolist()
mean = np.mean(points, axis=0)
normal_mean = np.mean(points[:-1], axis=0)
def_color = "blue"
outlier_color = "red"
colors = ['green',"orange",'red', "purple", "black"]
plt.xlim(-2,4)
plt.scatter([x[0] for x in points[:-1]], [x[1] for x in points[:-1]], marker='o', color=def_color, label="point")
plt.scatter([points[-1][0]], [points[-1][1]], marker='o', color=outlier_color, label="outlier point")
plt.scatter([mean[0]],[mean[1]], marker='x', color=def_color, label="mean")
plt.scatter([normal_mean[0]],[normal_mean[1]], marker='^', color=def_color, label="normal mean")
for i,p in enumerate(protos):
    plt.scatter([p[0]],[p[1]], marker='x', color=colors[i], label='%d th proto'%(i+1))
plt.legend()
plt.show()

colors = ['blue','orange','red','green','purple','brown','cyan','gray','olive']
coupling_hist = np.array(coupling_hist)
x = [i+1 for i in range(iters)]
plt.xticks(x)
plt.xlabel('routing epoch')
plt.ylabel('couping coefficient')
for i in range(len(regions)):
    print(i)
    plt.plot(x,coupling_hist[:,i],color=colors[i],label='point %d'%(i+1))
plt.legend()
plt.show()


# print(points)


# x = t.Tensor([[1,3],[2,7],[3,2]])
# x = np.arange(10)*0.2
# y = np.random.rand(100)
# bar_frequency(y)


