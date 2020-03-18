import torch as t
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import ResNet

from modules.model.MalResnet import ResNet
from modules.utils.dlUtils import get_attention_block

class ResChannelNet(nn.Module):
    def __init__(self, k, in_channel):
        super(ResChannelNet, self).__init__()
        self.K = k
        self.InChannel = in_channel
        self.Channels = [64, 128, 256, 512, 512]
        self.Encoder = ResNet(channel=in_channel, out=None, channels=self.Channels)

        attention_paddings = [(int((k - 1) / 2), 0), (int((k - 1) / 2), 0), (int((k - 1) / 2), 0), (0, 0)]
        attention_channels = [1,32,64,1]
        attention_strides = [(1,1),(1,1),(k,1)]
        attention_kernels = [(k,1),(k,1),(k,1)]
        attention_relus = [True,True,False]
        attention_drops = [None, None, None]     # 仿照HAPP中的实现，在最终Conv之前施加一个Dropout
        attention_bns = [False, False, False]

        self.ProtoNet = nn.Sequential(
            *[get_attention_block(attention_channels[i],
                                  attention_channels[i+1],
                                  attention_kernels[i],
                                  attention_strides[i],
                                  attention_paddings[i],
                                  relu=attention_relus[i],
                                  drop=attention_drops[i],
                                  bn=attention_bns[i])
              for i in range(len(attention_channels)-1)])

    def forward(self, support, query):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        assert k==self.K, 'K-shot的数目 %d 与模型的K-shot数目 %d 不一致'%(k, self.K)

        support = support.view(n*k, self.InChannel, w, w)
        query = query.view(qk, self.InChannel, w, w)

        query = self.Encoder(query)
        query = query.view(qk, -1)
        d = query.size(1)
        support = self.Encoder(support).view(n, 1, k, d)

        support = self.ProtoNet(support).view(n, d)

        # 将原型向量与查询集打包
        # shape: [n,d]->[qk, n, d]
        support = support.repeat((qk,1,1)).view(qk,n,-1)

        # query shape: [qk,d]->[n,qk,d]->[qk,n,d]
        query = query.repeat(n,1,1).transpose(0,1).contiguous().view(qk,n,-1)

        # 在原cos相似度的基础上添加放大系数
        scale = 10
        posterior = F.cosine_similarity(query, support, dim=2)*scale
        posterior = F.log_softmax(posterior, dim=1)

        return posterior



