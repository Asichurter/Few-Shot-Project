import torch
import math
import numpy as np
import os
import PIL.Image as Image
import torchvision.transforms as T
from sklearn.neighbors import KNeighborsClassifier as KNN
from torch.nn.init import kaiming_normal_, xavier_normal_, constant_, normal_

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
        label = (query==support).view(-1,k)
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
                image = transformer(image).squeeze()
                size = image.size
                data.append(image.sum()/(size(0)*size(1)))

    return np.mean(data),np.std(data),data

if __name__ == "__main__":
    print(calculate_data_mean_std('D:/peimages/New/cluster_2/train/', split=False))






