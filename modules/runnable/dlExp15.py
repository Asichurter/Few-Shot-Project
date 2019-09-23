import torch as t
import torch.nn as nn
from torch.optim import SGD
from torch.nn.init import normal_
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN

def get_datas(c, R, num):
    datas = []
    for i in range(num):
        radius = rd.uniform(0, R)
        theta = rd.uniform(0, 2*math.pi)
        data = [c[0]+radius*math.cos(theta), c[1]+radius*math.sin(theta)]
        datas.append(data)
    return datas

def plot(ps, num, title, group=3, colors=['red','blue','green'], marker=['o','x']):
    if title is not None:
        plt.title(title)
    for i,p in enumerate(ps):
        for g in range(group):
            plt.scatter([x[0] for x in p[g*num[i]:(g+1)*num[i]]],
                     [x[1] for x in p[g*num[i]:(g+1)*num[i]]],
                     marker=marker[i],
                     color=colors[g])
    plt.show()

def get_loss(p, num, alpha=0.01, beta=0.01, margin=1):
    # shape: [c,n,d]
    p = p.view(-1,num,2)
    p_inner_var = p.var(dim=1).sum()
    p_outer_var = p.mean(dim=1).var(dim=0).sum()
    # print('inner',p_inner_var.item())
    # print('outer',p_outer_var.item())
    return alpha*p_inner_var - beta*p_outer_var + margin

def get_block(in_feature, out_feature, relu=True, bn=True):
    block_parts = [nn.Linear(in_feature, out_feature)]
    if relu:
        block_parts.append(nn.ReLU())
    if bn:
        block_parts.append(nn.BatchNorm1d(out_feature))
    return nn.Sequential(*block_parts)

def cal_norm(pars):
    norm = 0
    for par in pars:
        norm += par.data.detach().norm()
    return norm.item()

def get_classify_loss(trans_datas, test_datas, labels, num, loss_func=nn.CrossEntropyLoss()):
    trans_datas = trans_datas.view(-1,num,2)
    # shape: [c,n,d]->[c,d]->[c,n,d]
    trans_mean = trans_datas.mean(dim=1).unsqueeze(dim=1).repeat((1,num,1))
    # shape: [c,n.d]->[c,n]->[c,n,d]
    attention = t.softmax(t.abs(trans_datas-trans_mean).sum(dim=2).neg(), dim=1).unsqueeze(dim=2).repeat((1,1,2))
    # shape: [c,n,d]->[c,d]->[l,c,d]
    mean_vec = (attention * trans_datas).sum(dim=1).repeat((len(test_datas),1,1))

    # shape: [l,d]->[l,c,d]
    test_datas = test_datas.unsqueeze(dim=1).repeat((1,3,1))

    prob = t.softmax(((test_datas-mean_vec)**2).sum(dim=2).neg(), dim=1)
    return loss_func(prob, labels)


centers = [[0,5],[0,3],[0,-1]]
groups = 3
r = 4
point_num = 50
episode = 1000
lr = 1
test_num = 20
penalty_coef = 1

feature_size = [2, 16, 16, 2]
transformers = [get_block(feature_size[i], feature_size[i+1], relu=False) #(i==len(feature_size)-2)
                                            for i in range(len(feature_size)-1)]
transformer = nn.Sequential(*transformers)

for name,par in transformer.named_parameters():
    if name.find("weight") != -1:
        print(name)
        normal_(par, 0, 0.1)
    if name.find('bias') != -1:
        par.data.fill_(0)

opt = SGD(transformer.parameters(), lr=lr)

acc_gain = 0
for i in range(episode):
    points = []
    for c in centers:
        points += get_datas(c, r, point_num)
    test_x = []
    for c in centers:
        test_x += get_datas(c, r, test_num)
    train_labels = np.concatenate([np.array([i]*point_num) for i in range(groups)], axis=0)
    test_labels = np.concatenate([np.array([i]*test_num) for i in range(groups)], axis=0)

    knn = KNN(n_neighbors=5)
    knn.fit(points, train_labels)
    predict = knn.predict(test_x)

    before_acc = (predict==test_labels).sum()/len(predict)
    # plot([points, test_x],
    #      [point_num, test_num],
    #      title="before %dth transformation\nacc=%.4f"%(i+1, before_acc),
    #      marker=['o','x'])

    points_t = t.Tensor(points)
    points_t.requires_grad_(True)

    transformer.zero_grad()
    transformed_points = transformer(points_t)
    loss = get_loss(transformed_points, point_num, alpha=0, beta=0.01)
    norm = cal_norm(transformer.parameters())
    loss += norm * penalty_coef

    print(i, "norm:", norm)
    print(i, "loss", loss.item())

    loss.backward()
    opt.step()

    # plot(transformed_points.detach().numpy(), point_num, 'after %dth transformation' % (i + 1), show=True)

    knn = KNN(n_neighbors=5)
    test_x = t.Tensor(test_x)
    test_x.requires_grad_(False)
    test_x = transformer(test_x).detach().numpy()
    transformed_points = transformed_points.detach().numpy()

    knn.fit(transformed_points, train_labels)
    predict = knn.predict(test_x)

    acc = (predict==test_labels).sum()/len(predict)
    acc_gain += acc-before_acc
    # print('acc:', acc)

    # test_points = np.concatenate([transformed_points.detach().numpy()[i*point_num:i*point_num+test_num]
    #                               for i in range(groups)], axis=0)
    # plot([transformed_points, test_x],
    #      [point_num, test_num],
    #      title="after %dth transformation\nacc=%.4f\nloss=%.4f"%(i+1, acc, loss),
    #      marker=['o','x'])

print('avg acc gain',acc_gain/episode)




    




