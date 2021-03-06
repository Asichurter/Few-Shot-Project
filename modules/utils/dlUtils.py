import torch
import math
import numpy as np
import os
import PIL.Image as Image
import torchvision.transforms as T
from sklearn.neighbors import KNeighborsClassifier as KNN
from torch.nn.init import kaiming_normal_, xavier_normal_, constant_, normal_
from torch import nn as nn

def RN_labelize(support, query, k, n, type="float", expand=True):
    # support = torch.LongTensor([support[i].item() for i in range(0,len(support),k)])
    support = support[::k]
    # support = support.cuda()
    support = support.repeat(len(query))
    query = query.view(-1,1)
    query = query.repeat((1,n)).view(-1)
    assert support.size(0) == query.size(0), "扩展后的支持集和查询集的标签长度不一致，无法生成得分向量！"
    if not expand:
        label = torch.argmax((query==support).view(-1,n), dim=1)
    else:
        label = (query==support).view(-1,1)
    if type=="float":
        return label.float()
    else:
        return label.long()


#每个神经网络层的权重的初始化方法，用于传递给module中所有的子模块的函数参数
def RN_weights_init(m):
    classname = m.__class__.__name__
    #如果是卷积层
    if classname.find('Conv') != -1:
        #计算卷积核的长x宽x数量，得到总共需要初始化的个数
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #将权重向量初始化为以0为均值，2/n为标准差 的正态分布
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #如果该层存在偏置项，则将偏置项置为0
        if m.bias is not None:
            m.bias.data.zero_()
    #否则该层为批正则化
    elif classname.find('BatchNorm') != -1:
        #将数据全部置为1
        m.weight.data.fill_(1)
        #偏置项置为0
        m.bias.data.zero_()
    #否则为线性层时
    elif classname.find('Linear') != -1:
        #n为线性层的维度
        n = m.weight.size(1)
        #权重全部初始化为简单正态分布
        m.weight.data.normal_(0, 0.01)
        #偏置项全部置为1
        m.bias.data = torch.ones(m.bias.data.size()).cuda()

def net_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        for name,par in m.named_parameters():
            if name.find('weight') != -1:
                kaiming_normal_(par, nonlinearity="relu")
            elif name.find('bias') != -1:
                constant_(par, 0)

    elif classname.find("Linear") != -1:
        for name,par in m.named_parameters():
            if name.find('weight'):
                try:
                    xavier_normal_(par)
                except ValueError:
                    print(name)
            elif name.find('bias') != -1:
                # constant_(par, 0)
                normal_(par, 0, 0.01)

def RN_baseline_KNN(supports, queries, support_labels, query_labels, k):
    '''
    作为RelationNetwork的基准线，将Embed模块的输出使用knn进行分类评估
    :return: knn分类正确率
    '''
    support_labels = np.array([support_labels[i].item() for i in range(0,len(support_labels),k)])
    query_labels = query_labels.cpu().detach().numpy()
    supports = supports.cpu().detach().numpy().reshape(supports.shape[0],-1)
    queries = queries.cpu().detach().numpy().reshape(queries.shape[0],-1)

    knn = KNN(n_neighbors=1)
    knn.fit(supports, support_labels)
    predicts = knn.predict(queries)
    return (predicts==query_labels).sum()/query_labels.shape[0]

def RN_repeat_query_instance(instances, n):
    result = torch.FloatTensor([]).cuda()
    for tensor in torch.unbind(instances):
        repeat_ones = tensor.repeat((n,1,1,1))
        result = torch.cat((result, repeat_ones), dim=0)
    return result

def cos_sim(x,y,dim):
    x_length = torch.sqrt((x**2).sum(dim=dim))
    y_length = torch.sqrt((y**2).sum(dim=dim))
    dot = torch.mul(x,y).sum(dim=dim)
    cos = dot/(x_length*y_length)*0.5 + 0.5
    return cos

def calculate_data_mean_std(base, split=True, excepts=['models']):
    data = []
    transformer = T.ToTensor()
    for c_o in os.listdir(base) if split else [base]:
        if c_o in excepts:
            continue
        path = base + c_o + "/" if split else base
        for c_i in os.listdir(path):
            print(c_o,c_i)
            inner_path = path + c_i + "/"
            for image in os.listdir(inner_path):
                image = Image.open(inner_path+image)
                image = transformer(image)
                size = image.size
                data.append((image.sum(dim=2).sum(dim=1)/(size(1)*size(2))).tolist())

    return np.mean(data, axis=0), np.std(data, axis=0)#,data

def cal_beliefe_interval(datas, split=5):
    '''
    计算数据95%的置信区间。依据的是t分布的95%置信区间公式
    '''
    # assert len(datas)%split == 0, '数据不可被split等分。数据长度=%d  split=%d'%(len(datas, split))

    # if type(datas) == list:
    #    datas = np.array(datas)
    # datas = datas.reshape(split,-1)
    # means = datas.mean(axis=1).reshape(-1)
    # std = np.std(means)

    z = 1.95996
    s = np.std(datas, ddof=1)
    n = len(datas)

    return z*s/np.sqrt(n)

def get_block_1(in_feature, out_feature, stride=1, kernel=3, padding=1, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

def get_block_2(in_feature, out_feature, stride=1, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(3,2,1)
    )

def get_attention_block(in_channel, out_channel, kernel_size, stride=1, padding=1, relu=True, drop=None, bn=True):
    block = nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding)#,
        # nn.ReLU(inplace=True)
    )
    if bn:
        block.add_module('bn', nn.BatchNorm2d(out_channel))
    if relu:
        block.add_module('relu', nn.ReLU(inplace=True))
    if drop is not None:
        block.add_module('drop', nn.Dropout2d(drop))
        # block.add_module('drop', nn.Dropout())
    return block

def labels_normalize_(labels, n, batch_length):
    labels_ = labels.numpy()
    for i in range(0, len(labels), batch_length):
        batch = labels_[i:i+batch_length]
        batch_labels = np.unique(batch)
        # batch_labels = batch[::k]
        assert n == len(batch_labels), 'batch内，指定的类别数量%d与实际的类别数量%d不一致'%\
                                       (n, len(batch_labels))
        mapper = {i:l for i,l in zip(batch_labels, [j for j in range(n)])}

        for ii,l in enumerate(batch):
            labels_[i+ii] = mapper[l]

    return torch.Tensor(labels_).long()

def labels_normalize(labels, k, n, batch_length):
    labels_ = labels.numpy()
    for i in range(0, len(labels), batch_length):
        batch = labels_[i:i+batch_length]
        # batch_labels = np.unique(batch)
        batch_labels = batch[::k]
        assert n == len(batch_labels), 'batch内，指定的类别数量%d与实际的类别数量%d不一致'%\
                                       (n, len(batch_labels))
        mapper = {i:l for i,l in zip(batch_labels, [j for j in range(n)])}

        for ii,l in enumerate(batch):
            labels_[i+ii] = mapper[l]

    return torch.Tensor(labels_).long()

def labels_normalize__(s_labels, q_labels, n):
    label_space = np.unique(s_labels.numpy())
    assert n == len(label_space), 'batch内，指定的类别数量%d与实际的类别数量%d不一致' % \
                                   (n, len(label_space))
    mapper = {i: l for i, l in zip(label_space, [j for j in range(n)])}

    support_labels = s_labels.numpy()
    query_labels = q_labels.numpy()
    for i in range(len(support_labels)):
        support_labels[i] = mapper[support_labels[i]]
    for i in range(len(query_labels)):
        query_labels[i] = mapper[query_labels[i]]

    return torch.Tensor(support_labels).long(),\
           torch.Tensor(query_labels).long()

def labels_one_hot(labels, n):
    # 假定输入的标签已经在一个batch内被标准化
    labels_ = labels.numpy()
    one_hots = []
    for l in labels_:
        one_hot = [1 if j==l else 0 for j in range(n)]
        one_hots.append(one_hot)

    return torch.Tensor(one_hots).float()



if __name__ == "__main__":
    print(calculate_data_mean_std('D:/peimages/New/miniImageNet/train/', split=False))






