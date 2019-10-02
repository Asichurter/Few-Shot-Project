# 本实验利用训练好的模型将验证集可视化

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import os
from torch.utils.data.dataloader import DataLoader

from modules.model.PrototypicalNet import ProtoNet
from modules.utils.datasets import FewShotRNDataset, FewShotFileDataset, get_RN_sampler

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = 20

class_contents_num = 5

folder = 'cluster'
version = 38

MODEL_PATH = "D:/peimages/New/%s/models/"%folder+"ProtoNet_best_acc_model_5shot_5way_v%d.0.h5"%version
DATA_PATH = "D:/peimages/New/%s/test.npy"%folder
DATA_LENGTH = 50#len(os.listdir(DATA_PATH))
BATCH_LENGTH = int(DATA_LENGTH/n)#int(len(os.listdir(DATA_PATH))/n)

# dataset = FewShotRNDataset(DATA_PATH, N, rotate=False)
dataset = FewShotFileDataset(DATA_PATH, N, DATA_LENGTH, rotate=False, squre=True)
datas = []
reducer = MDS(n_components=2)
# reducer = PCA(n_components=2)
# reducer = t_sne.TSNE(n_components=2)

# def transform(path, transforms=T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])):
#     img = Image.open(path)
#     return transforms(img)
#
#
# datas = []
#
# for c in os.listdir(DATA_PATH):
#     class_datas = []
#     class_path = DATA_PATH+c+"/"
#     for item in os.listdir(class_path):
#         item = transform(class_path+item)
#         class_datas.append(item)
#     datas.append(class_datas)

net = ProtoNet().cuda()
for i in range(BATCH_LENGTH):
    classes = [i*n+j for j in range(n)]
    print(classes)
    support_sampler, test_sampler = get_RN_sampler(classes, k, qk, N)
    test_support_dataloader = DataLoader(dataset, batch_size=n * k,
                                         sampler=support_sampler)
    test_test_dataloader = DataLoader(dataset, batch_size=qk * n,
                                      sampler=test_sampler)

    supports, support_labels = test_support_dataloader.__iter__().next()
    tests, test_labels = test_test_dataloader.__iter__().next()

    supports = supports.cuda()
    tests = tests.cuda()

    supports,tests = net(supports.view(n,k,1,256,256), tests.view(n*qk,1,256,256), save_embed=True)
    batch_datas = supports.view(n, k, -1).cpu().detach().numpy().tolist()
    # batch_datas = t.cat((supports, tests), dim=1).view(n, qk+k, -1).cpu().detach().numpy().tolist()

    datas += batch_datas

datas = np.array(datas).reshape(DATA_LENGTH*class_contents_num, -1)
reduced_datas = reducer.fit_transform(datas)
reduced_datas = reduced_datas.reshape(-1,class_contents_num,2)
class_num = reduced_datas.shape[0]

colors = ['black', #1
          'darkgray',  #2
          'gainsboro',  #3
          'rosybrown',  #4
          'lightcoral',  #5
          "indianred",  #6
          'firebrick',  #7
          'maroon',  #8
          'red',  #9
          'coral',  #10
          'orangered',  #11
          'sienna',  #12
          'chocolate',  #13
          'sandybrown',  #14
          'bisque',  #15
          'wheat',  #16
          'gold',  #17
          'khaki',  #18
          'olive',  #19
          'lightyellow',  #20
          'yellow',  #21
          'olivedrab',  #22
          'yellowgreen',  #23
          'darkseagreen',  #24
          'palegreen',  #25
          'darkgreen',  #26
          'lime',  #27
          'mediumseagreen',  #28
          'aquamarine',  #29
          'turquoise',  #30
          'lightseagreen',  #31
          'lightcyan',  #32
          'teal',  #33
          'lightblue',  #34
          'deepskyblue',  #35
          'steelblue',  #36
          'navy',  #37
          'blue',  #38
          'slateblue',  #39
          'darkslateblue',  #40
          'mediumpurple',  #41
          'blueviolet',  #42
          'plum',  #43
          'violet',  #44
          'purple',  #45
          'magenta',  #46
          'mediumvioletred',  #47
          'hotpink',  #48
          'crimson',  #49
          'pink',  #50
          ]
cnames = {
'aliceblue':   '#F0F8FF',
'antiquewhite':   '#FAEBD7',
'aqua':     '#00FFFF',
'aquamarine':   '#7FFFD4',
'azure':    '#F0FFFF',
'beige':    '#F5F5DC',
'bisque':    '#FFE4C4',
'black':    '#000000',
'blanchedalmond':  '#FFEBCD',
'blue':     '#0000FF',
'blueviolet':   '#8A2BE2',
'brown':    '#A52A2A',
'burlywood':   '#DEB887',
'cadetblue':   '#5F9EA0',
'chartreuse':   '#7FFF00',
'chocolate':   '#D2691E',
'coral':    '#FF7F50',
'cornflowerblue':  '#6495ED',
'cornsilk':    '#FFF8DC',
'crimson':    '#DC143C',
'cyan':     '#00FFFF',
'darkblue':    '#00008B',
'darkcyan':    '#008B8B',
'darkgoldenrod':  '#B8860B',
'darkgray':    '#A9A9A9',
'darkgreen':   '#006400',
'darkkhaki':   '#BDB76B',
'darkmagenta':   '#8B008B',
'darkolivegreen':  '#556B2F',
'darkorange':   '#FF8C00',
'darkorchid':   '#9932CC',
'darkred':    '#8B0000',
'darksalmon':   '#E9967A',
'darkseagreen':   '#8FBC8F',
'darkslateblue':  '#483D8B',
'darkslategray':  '#2F4F4F',
'darkturquoise':  '#00CED1',
'darkviolet':   '#9400D3',
'deeppink':    '#FF1493',
'deepskyblue':   '#00BFFF',
'dimgray':    '#696969',
'dodgerblue':   '#1E90FF',
'firebrick':   '#B22222',
'floralwhite':   '#FFFAF0',
'forestgreen':   '#228B22',
'fuchsia':    '#FF00FF',
'gainsboro':   '#DCDCDC',
'ghostwhite':   '#F8F8FF',
'gold':     '#FFD700',
'goldenrod':   '#DAA520',
'gray':     '#808080',
'green':    '#008000',
'greenyellow':   '#ADFF2F',
'honeydew':    '#F0FFF0',
'hotpink':    '#FF69B4',
'indianred':   '#CD5C5C',
'indigo':    '#4B0082',
'ivory':    '#FFFFF0',
'khaki':    '#F0E68C',
'lavender':    '#E6E6FA',
'lavenderblush':  '#FFF0F5',
'lawngreen':   '#7CFC00',
'lemonchiffon':   '#FFFACD',
'lightblue':   '#ADD8E6',
'lightcoral':   '#F08080',
'lightcyan':   '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':   '#90EE90',
'lightgray':   '#D3D3D3',
'lightpink':   '#FFB6C1',
'lightsalmon':   '#FFA07A',
'lightseagreen':  '#20B2AA',
'lightskyblue':   '#87CEFA',
'lightslategray':  '#778899',
'lightsteelblue':  '#B0C4DE',
'lightyellow':   '#FFFFE0',
'lime':     '#00FF00',
'limegreen':   '#32CD32',
'linen':    '#FAF0E6',
'magenta':    '#FF00FF',
'maroon':    '#800000',
'mediumaquamarine':  '#66CDAA',
'mediumblue':   '#0000CD',
'mediumorchid':   '#BA55D3',
'mediumpurple':   '#9370DB',
'mediumseagreen':  '#3CB371',
'mediumslateblue':  '#7B68EE',
'mediumspringgreen': '#00FA9A',
'mediumturquoise':  '#48D1CC',
'mediumvioletred':  '#C71585',
'midnightblue':   '#191970',
'mintcream':   '#F5FFFA',
'mistyrose':   '#FFE4E1',
'moccasin':    '#FFE4B5',
'navajowhite':   '#FFDEAD',
'navy':     '#000080',
'oldlace':    '#FDF5E6',
'olive':    '#808000',
'olivedrab':   '#6B8E23',
'orange':    '#FFA500',
'orangered':   '#FF4500',
'orchid':    '#DA70D6',
'palegoldenrod':  '#EEE8AA',
'palegreen':   '#98FB98',
'paleturquoise':  '#AFEEEE',
'palevioletred':  '#DB7093',
'papayawhip':   '#FFEFD5',
'peachpuff':   '#FFDAB9',
'peru':     '#CD853F',
'pink':     '#FFC0CB',
'plum':     '#DDA0DD',
'powderblue':   '#B0E0E6',
'purple':    '#800080',
'red':     '#FF0000',
'rosybrown':   '#BC8F8F',
'royalblue':   '#4169E1',
'saddlebrown':   '#8B4513',
'salmon':    '#FA8072',
'sandybrown':   '#FAA460',
'seagreen':    '#2E8B57',
'seashell':    '#FFF5EE',
'sienna':    '#A0522D',
'silver':    '#C0C0C0',
'skyblue':    '#87CEEB',
'slateblue':   '#6A5ACD',
'slategray':   '#708090',
'snow':     '#FFFAFA',
'springgreen':   '#00FF7F',
'steelblue':   '#4682B4',
'tan':     '#D2B48C',
'teal':     '#008080',
'thistle':    '#D8BFD8',
'tomato':    '#FF6347',
'turquoise':   '#40E0D0',
'violet':    '#EE82EE',
'wheat':    '#F5DEB3',
'white':    '#FFFFFF',
'whitesmoke':   '#F5F5F5',
'yellow':    '#FFFF00',
'yellowgreen':   '#9ACD32'}

# plt.title("Acc = %.4f"%acc)
plt.axis("off")
plt.figure(figsize=(15,12))
for i in range(DATA_LENGTH):
    plt.plot([reduced_datas[i][j][0] for j in range(class_contents_num)],
                [reduced_datas[i][j][1] for j in range(class_contents_num)],
                marker="o", color=cnames[colors[i]])
    # plt.scatter([x[0] for x in samples_trans[i*k:(i+1)*k]],[x[1] for x in samples_trans[i*k:(i+1)*k]],
    #          marker="o", color=colors[i])
    # plt.scatter([x[0] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
    #          [x[1] for j,x in enumerate(queries_trans) if query_labels[j]==sample_labels[i]],
    #          marker="^", color=colors[i])
plt.show()
plt.figure(figsize=(15,12))
for i in range(DATA_LENGTH):
    plt.scatter([reduced_datas[i][j][0] for j in range(class_contents_num)],
                [reduced_datas[i][j][1] for j in range(class_contents_num)],
                marker="o", color=cnames[colors[i]])
plt.show()








