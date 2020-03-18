import torch as t
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from modules.utils.dlUtils import get_block_1

class ItemQueryAttention(nn.Module):
    '''
    基于项的注意力机制。使用查询集序列对支持集的样本序列进行注意力对齐，
    得到一个支持集样本的注意力上下文向量。由于注意力向量不依赖于RNN的
    上下文向量，因此该注意力属于基于项的注意力，可以并行化处理
    '''
    def __init__(self, feature_size, hidden_size):
        super(ItemQueryAttention, self).__init__()
        # 假定输入的支持集隐藏态和查询集数据通过连接合并
        self.W = nn.Linear(feature_size, hidden_size)

    # qs: [q_batch, seq, feature_size]
    # hs: [s_batch, seq, feature_size]
    def forward(self, qs, hs):
        assert len(qs.size())==3 and len(hs.size())==3, '输入attention的尺寸不符！'
        s_size = hs.size(0)
        q_size = qs.size(0)

        # 假定支持集和查询集的嵌入特征长度相同
        feature_size = qs.size(2)

        # 假定支持集嵌入和查询集嵌入的序列长度相同
        seq_size = hs.size(1)

        # both: [q_batch, s_batch, seq, seq, feature_size]
        # 此处需特别注意向量的位置
        # qs第一次转置是为了在每一次样本查询时，对所有支持集样本，查询集样本一样
        # qs第二次转置是为了在每一次序列查询时，对所有的支持集样本序列，使用同一个查询集样本序列
        qs = qs.repeat((s_size,1,1,1)).transpose(0,1).contiguous().unsqueeze(2).repeat(1,1,seq_size,1,1).transpose(2,3)
        hs = hs.repeat((q_size,1,1,1)).unsqueeze(2).repeat(1,1,seq_size,1,1)

        # attention: [q_batch, s_batch, seq, seq]
        att = t.sum(t.tanh(self.W(qs) * self.W(hs)), dim=4).softmax(dim=3).squeeze()
        # expanded attention: [q_batch, s_batch, seq, seq, feature_size]
        att = att.unsqueeze(dim=4).repeat((1,1,1,1,feature_size))

        # attended hs: [q_batch, s_batch, seq, feature]
        hs = (att * hs).sum(dim=3)

        return hs

class ReMalAttNet(nn.Module):
    def __init__(self, in_channel, in_size, hidden_size=128):
        super(ReMalAttNet, self).__init__()
        channels = [in_channel,32,64,128,256]
        strides = [2,1,1,1]
        paddings = [1,1,1,1]
        layers = [get_block_1(channels[i],
                              channels[i+1],
                              strides[i],
                              padding=paddings[i]) for i in range(len(strides))]
        self.Encoder = nn.Sequential(*layers)
        self.Channels = channels

        # 得到特征以后，对所有通道的同一位置提取为一个特征，因此特征
        # 的维度等于最终通道数
        feature_size = channels[-1]

        # 最终的feature map的大小为序列长度，与stride
        # 和layer的maxpool数量有关
        self.seq_size = int((in_size // (2**len(layers)) // reduce(lambda x,y: x*y, strides))**2)

        # # 使用注意力查询隐藏态以后，得到的语义向量之后输入解析RNN得到关联性
        # self.QueryAlphaCell = nn.GRUCell()

        self.Attention = ItemQueryAttention(feature_size, hidden_size=64)

        # 将注意力对齐的支持集样本与查询集嵌入并联输入
        self.QueryRNN = nn.GRU(feature_size*2,
                               hidden_size=hidden_size,
                               num_layers=2,
                               batch_first=True,
                               # bidirectional=True
                               )

        self.Transform = nn.Linear(hidden_size, 1)
        self.feature_size = feature_size

    def forward(self, support, query):
        # sup size: [n, k, c, w, h]
        # query size: [qk, c, w, h]
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, self.Channels[0], w, w)

        feature_size = self.feature_size
        seq_size = self.seq_size

        # sup size: [n*k, c', w, w]->[n*k, w*w, c']
        # query size: [qk, c', w, w]->[qk, w*w, c']
        # 利用转置，得到，每个样本的feature map大小个特征，每个特征长度为通道数
        support = self.Encoder(support).view(n*k, feature_size, seq_size).transpose(1,2).contiguous()
        query = self.Encoder(query).view(qk, feature_size, seq_size).transpose(1,2).contiguous()

        # print('w:',w,'k:',k,'qk:',qk,'n:',n)

        # att_sup size: [qk, n*k, w*w(seq), feature(c')] -> [qk*n*k, seq, feature]
        attended_support = self.Attention(query, support).view(qk*n*k, seq_size, feature_size)

        # out size: [qk*n*k, seq, hidden]
        query = query.unsqueeze(dim=1).repeat(1,n*k,1,1).view(qk*n*k, seq_size, -1)
        print('attended:', attended_support.size())
        print('query:', query.size())
        # print(t.cat((attended_support, query), dim=2).size())
        out,_ = self.QueryRNN(t.cat((attended_support, query), dim=2))
        # result size: [qk, n]
        class_level_result = self.Transform(out[:,-1,:].squeeze()).view(qk, n, k).sum(dim=2).squeeze()

        return F.log_softmax(class_level_result, dim=1)











