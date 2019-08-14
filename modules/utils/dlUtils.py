import torch
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

def RN_labelize(support, query, num_instance):
    support = torch.LongTensor([support[i].item() for i in range(0,len(support),num_instance)])
    support = support.cuda()
    support = support.repeat(len(query))
    query = query.view(-1,1)
    query = query.repeat((1,num_instance)).view(-1)
    assert support.size()[0] == query.size()[0], "扩展后的支持集和查询集的标签长度不一致，无法生成得分向量！"
    label = (query==support).view(-1,num_instance)
    return label.float()

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





