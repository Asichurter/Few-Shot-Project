import matplotlib.pyplot as plt
import numpy as np

name = ['filtered \nLarge PE', 'unfiltered \nLarge PE', 'VirusShare']

_5shot_proto = [95.44, 89.88, 86.61]
_5shot_channel = [67.11, 30.28, 50.77]
_10shot_proto = [90.62, 205.89, 205.69]
_10shot_channel = [50.27, 56.18, 45.66]

width = 0.75
text_width = 0.2
extra_width = 0.2
font_size = 9
x = np.arange(len(name))*4
# x_extra = np.array([0,0,extra_width,extra_width])
# x = x + x_extra

plt.figure(dpi=600)
plt.bar(x, _5shot_proto,  width=width, label='5-shot ProtoNet')
plt.bar(x + width, _5shot_channel, width=width, label='5-shot ConvProtoNet')
plt.bar(x + 2 * width, _10shot_proto, width=width, label='10-shot ProtoNet')
plt.bar(x + 3 * width, _10shot_channel, width=width, label='10-shot ConvProtoNet')

# 显示在图形上的值
for a, b in zip(x,_5shot_proto):
    plt.text(a, b+text_width, b, ha='center', va='bottom', fontsize=font_size)
for a,b in zip(x,_5shot_channel):
    plt.text(a+width, b+text_width, b, ha='center', va='bottom', fontsize=font_size)
for a,b in zip(x, _10shot_proto):
    plt.text(a+2*width, b+text_width, b, ha='center', va='bottom', fontsize=font_size)
for a,b in zip(x, _10shot_channel):
    plt.text(a+3*width, b+text_width, b, ha='center', va='bottom', fontsize=font_size)

plt.xticks([1.1,5,9],name, fontsize=12)
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('Average KL divergence')
# plt.xlabel('datasets')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素 # 分辨率
# plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
plt.show()