import torch as t
import torch.nn as nn
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad

from modules.model.ResidualNet import ResidualNet
from modules.utils.dlUtils import RN_labelize
from modules.utils.datasets import FewShotRNDataset, get_RN_modified_sampler

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20

TEST_EPISODE = 50

VALIDATE_PATH = "D:/peimages/New/ProtoNet_5shot_5way_exp/test/"
MODEL_SAVE_PATH = "D:/peimages/New/RN_5shot_5way_exp/"+"Residual_best_acc_model_%dshot_%dway_v%d.0.h5" % (k, n, 5)

input_size = 256
test_classes = 30

TEST_CLASSES = [i for i in range(test_classes)]

net = ResidualNet(input_size,n,k,qk,metric="Proto")
net.load_state_dict(t.load(MODEL_SAVE_PATH))
net = net.cuda()

entro = nn.NLLLoss().cuda()
net.eval()

with no_grad():
    test_acc = 0.
    test_loss = 0.
    for j in range(TEST_EPISODE):
        # 每一轮开始的时候先抽取n个实验类
        support_classes = rd.sample(TEST_CLASSES, n)
        # 训练的时候使用固定的采样方式，但是在测试的时候采用固定的采样方式
        support_sampler, test_sampler = get_RN_modified_sampler(support_classes, k, qk, N)
        # print(list(support_sampler.__iter__()))
        test_dataset = FewShotRNDataset(VALIDATE_PATH, N)

        test_support_dataloader = DataLoader(test_dataset, batch_size=n * k,
                                             sampler=support_sampler)
        test_test_dataloader = DataLoader(test_dataset, batch_size=qk * n,
                                            sampler=test_sampler)

        supports, support_labels = test_support_dataloader.__iter__().next()
        tests, test_labels = test_test_dataloader.__iter__().next()

        supports = supports.cuda()
        support_labels = support_labels.cuda()
        tests = tests.cuda()
        test_labels = test_labels.cuda()

        # test_labels = RN_labelize(support_labels, test_labels, k, type="float", expand=True)
        test_labels = RN_labelize(support_labels, test_labels, k, type="long", expand=False)
        test_relations = net(supports, tests)

        test_loss += entro(test_relations, test_labels).item()
        acc = (t.argmax(test_relations, dim=1)==test_labels).sum().item()/test_labels.size(0)
        test_acc += acc

        print("test %d acc:%f"%(j,acc))
        # test_acc += (t.argmax(test_relations, dim=1)==t.argmax(test_labels,dim=1)).sum().item()/test_labels.size(0)

    print("****************************************")
    print("val acc: ", test_acc/TEST_EPISODE)
    print("val loss: ", test_loss/TEST_EPISODE)
    print("****************************************")