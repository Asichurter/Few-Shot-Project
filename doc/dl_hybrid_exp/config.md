# 运行配置

## 第1次运行

1. 大部分按照论文中的方式进行实现，做了少部分修改：
    1. 将Encoder改为与原型网络中相同的4层结构，使用Conv2d+BN+Relu+Maxpool。
        通道数除了输入均为64, 步长除了最后一层为1其余均为2
    2. 特征注意力FeatureAttention按照论文中进行实现：通道数为[1,32,64,1],
        步长除了最后一个Conv为(k,1)其余均为(1,1)，卷积核尺寸均为(k,1)，padding
        除了最后一个Conv中直接使用尺寸为k的卷积核将k个样本的特征合并为1个特征因此
        padding为0，其他层均使用same padding策略
    3. 样例注意力InstanceAttention使用一个线性层将尺寸为(batch, d)的支持集和
        查询集转换后按位相乘，直接作用tanh，然后将d维度上的值相加后就生成了每一个
        样本的注意力
    4. 最后在样例注意力下生成了支持向量，利用支持向量得到了与查询集的距离，利用特征注意力
        对该距离进行加权后得到最终距离，取负以后输入到softmax中得到预测值

2. 修改了初始化方式：不使用论文中的初始化方式（会发生梯度消失），而使用自定义的初始化方
    式。对卷积的权重使用
    kaiming初始化，对线性层的权重使用xavier初始化，对所有的bias选择填充0初始化
    
3. 使用Adam优化，但是学习率衰减改为15000轮变为原来的0.1

4. Encoder的Conv的通道数改为1->32->64->128->256，且使用LeakyReLU，池化使用重叠池化

5. 只使用InstanceAttention，没有使用FeatureAttention

## 2

1. cluster数据集

2. 5-way 5-shot

## 3

1. cluster数据集

2. 10-shot 20-way  

## 4

1. cluster数据集

2. 5-shot 20-way

## 5

1. cluster数据集

2. 10-shot 5-way

## 6

1. virushare_20数据集

2. 5-shot 5-way

## 7

1. virushare_20数据集

2. 5-shot 20-way

## 8 
1. virushare_20数据集

2. 10-shot 20-way

## 9

1. virushare_20数据集

2. 10-shot 5-way

## 10

1. test数据集

2. 5-shot 5-way

## 11

1. test数据集

2. 5-shot 20-way

## 12 

1. test数据集

2. 10-shot 20-way

## 13

1. test数据集

2. 10-shot 5-way


