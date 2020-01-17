import torch as t
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from torch.optim.adam import Adam

from modules.utils.dlUtils import get_block_1

def rename(name, token='-'):
    '''
    由于ParameterDict中不允许有.字符的存在，因此利用本方法来中转，将
    .字符转化为-来避开
    '''
    return name.replace('.',token)

class BaseLearner(nn.Module):
    def __init__(self, input_size, n):
        super(BaseLearner, self).__init__()
        self.Size = input_size
        self.channels = [1, 64, 64, 64, 64]
        self.strides = [2, 2, 2, 1]
        self.kernels = [3, 3, 3, 3]
        self.paddings = [1, 1, 1, 1]
        layers = [get_block_1(
            in_feature=self.channels[i],
            out_feature=self.channels[i+1],
            stride=self.strides[i],
            kernel=self.kernels[i],
            padding=self.paddings[i]) for i in range(4)]
        self.Encoder = nn.Sequential(*layers)

        out_size = input_size  //2 //2 //2 //2      # 池化导致的尺寸减小
        for stride in self.strides:
            out_size = out_size // stride           # 步长导致的尺寸减小
        assert out_size >= 1, '特征提取后的特征图的宽度小于1！尝试减少stride！'
        if out_size > 1:
            warn('特征提取后的feature map宽度不为1！为:%d!'%out_size)

        self.fc = nn.Linear(64*out_size*out_size, n)

    def forward(self, x, params=None):
        length = x.size(0)
        assert x.size(2)==x.size(3)==self.Size,\
                    '图片尺寸不对应！'

        if params is None:
            x = self.Encoder(x).view(length, -1)
            x = self.fc(x)
        else:
            # 使用适应后的参数来前馈
            for i in range(4):      # conv-4结构
                x = F.conv2d(
                    x,
                    params['Encoder.%d.0.weight'%i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
                x = F.batch_norm(
                    x,
                    params['Encoder.%d.1.running_mean'%i],
                    params['Encoder.%d.1.running_var'%i],
                    params['Encoder.%d.1.weight'%i],
                    params['Encoder.%d.1.bias'%i],
                    momentum=1,
                    training=True)
                x = F.relu(x, inplace=True)
                x = F.max_pool2d(x, 2)
            x = x.view(length, -1)
            x = F.linear(
                x,
                params['fc.weight'],
                params['fc.bias'],
            )

        return F.log_softmax(x, dim=1)

    def clone_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

class MetaSGD(nn.Module):
    def __init__(self, input_size, n, loss_fn, lr=1e-3):
        super(MetaSGD, self).__init__()
        self.Learner = BaseLearner(input_size, n)   # 基学习器内部含有beta
        self.Alpha = nn.ParameterDict({
            rename(name):nn.Parameter(lr * t.ones_like(val, requires_grad=True))
            for name,val in self.Learner.named_parameters()
        })      # 初始化alpha
        self.LossFn = loss_fn

    def forward(self, support, query, s_label):
        # for support, query, s_label, q_label in zip(support_list, query_list, s_label_list, q_label_list):
        s_predict = self.Learner(support)
        loss = self.LossFn(s_predict, s_label)
        self.Learner.zero_grad()        # 先清空基学习器梯度
        grads = t.autograd.grad(loss, self.Learner.parameters(), create_graph=True)
        adapted_state_dict = self.Learner.clone_state_dict()

        # 计算适应后的参数
        for (key, val), grad in zip(self.Learner.named_parameters(), grads):
            # 利用已有参数和每个参数对应的alpha调整系数来计算适应后的参数
            adapted_state_dict[key] = val - self.alpha(key) * grad

        # 利用适应后的参数来生成测试集结果
        return self.Learner(query, params=adapted_state_dict)

    def alpha(self, key):
        return self.Alpha[rename(key)]

if __name__ == '__main__':
    loss_fn = nn.NLLLoss()
    model = MetaSGD(input_size=32, n=5, loss_fn=loss_fn)
    opt = Adam(model.parameters(), lr=1e-3)
    batch_size = 10
    supports = []
    queries = []
    support_labels = []
    query_labels = []

    for i in range(batch_size):
        supports.append(t.randn((10,1,32,32)))
        support_labels.append(t.zeros((10), dtype=t.long))
        queries.append(t.randn((5,1,32,32)))
        query_labels.append(t.ones((5), dtype=t.long))

    model.zero_grad()
    meta_loss = 0.
    epoch = 0
    for support, query, s_label, q_label in zip(supports, queries, support_labels, query_labels):
        print(epoch)
        epoch += 1

        predict = model(support, query, s_label)
        meta_loss += loss_fn(predict, q_label)

    opt.zero_grad()
    meta_loss.backward()
    opt.step()
    print('done！')


