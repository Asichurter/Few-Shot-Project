## 第1次运行

1. 使用的是cluster数据集

2. 主要是使用(k,1)的卷积核卷积嵌入后特征向量的每一个维度，将k个特征向量生成为1个类向量

## 第2次运行

1. 使用的是cluster_2数据集

## 第3次运行

1. 改用cluster数据集

2. 移除了生成类向量时最后一层的ReLU

3. 使用了多层次池化，分别为1,2和4，为了让最后一层网络输出为8，最后一个Conv中的padding
为2

## 第4次运行

1. 只移除了ProtoNet中最后一个ReLU，没有使用多层次池化

## 第5次运行

1. 减少了ProtoNet部分的通道数：从1->32->64->1改为1->8->16->1试图减缓过拟合(没有采纳，过少的通道数使得模型
难以收敛)

2. 仿照HAPP中的实现，在ProtoNet最终Conv前施加了Dropout

3. 将Encoder部分的padding从[1,1,1,2]改为[1,1,1,1]

## 第6次运行

1. 将ProtoNet的通道数从[1,32,64,1]改为[1,16,32,1]

## 第7次运行

1. 将ProtoNet的通道数改为[1,32,64,32,1]

## 第8次运行

1. 保持7的通道数，在test数据集上测试未去噪前的效果

## 第9次运行

1. 重复第8次实验

## 第10次运行

1. 将卷积核数量改为[1,32,64,32,1]，同时没有dropout

## 第11次运行

1. 将Conv4结构中的LeakyReLU改为普通ReLU

## 第12次运行

1. 将Conv4中的ReLU改回LeakReLU

2. 将dropout改为dropout2d，卷积核数量为[1,32,64,1]

## 第13次运行(模型已被覆盖)

1. 将通道数改为[1,32,64,64,1]，其余不变

## 第14次运行

1. 将通道数改为[1,16,32,64,1]，其余不变

## 第15次运行

1. 将通道数改为[1,32,64,1]

2. 在Conv4中使用PReLU

## 第16次运行

1. 重新试验普通Conv4结构+不移除ReLU

## 第17次运行

1. 重新试验改进Conv4结构+不移除ReLU

## 第18次运行

1. 重新试验普通Conv4结构+移除ReLU

## 第19次运行

1. 重新实验改进Conv4+移除ReLU

## 第20次运行

1. 使用改进后的Conv4+移除ReLU

2. 实验在最后一层前施加0.2的Dropout2D

## 第21次运行

1. 测试k=5,n=20，同时设置qk=5

2. 使用的是未改进的Conv4+无BN+移除ReLU

## 22

1. 使用新的采样器

## 23

1. 5-shot 20-way

## 24

1. 5-shot 20-way

2. 使用cluster_2数据集

## 25

1. 10-shot 5-way

## 26

1. 10-shot 20-way，为了适应显存大小，将裁剪尺寸改为192

## 27

1. virusshare 数据集

2. 5-shot 5-way

## 28

1. test数据集

2. 5-shot 5-way

## 29

1. virushare_20数据集

2. 5-shot 20-way

## 30

1. virushare_20 数据集

2. 10-shot 5-way

## 31 

1. virushare_20数据集

2. 10-shot 20-way

## 32 （重复29）

1. virushare_20数据集

2. 5-shot 20-way

## 33 (重复28)

1. 使用test数据集

2. 5-shot 5-way

## 34

1. 使用test数据集

2. 5-shot 20-way

3. 50000episodes

## 35

1. test数据集

2. 10-shot 20-way

## 36

1. test数据集

2. 10-shot 5-way

## 37

1. cluster数据集

2. 5-shot 5-way

## 38

1. drebin15数据集

2. 5-shot 5-way

## 39

1. drebin15数据集

2. 5-shot 5-way

3. 192crop， 20000episode

## 40

1. drebin15数据集

2. 10-shot 5-way

3. 192crop， 20000episode

## 41

1. drebin_10数据集

2. 5-shot 5-way

3. 192crop， 20000episode

## 42

1. drebin_10数据集

2. 5-shot 10-way

3. 192crop， 20000episode

## 43（重复34）

1. test数据集

2. 5-shot 20-way

## 45（重复31）

1. virushare20数据集

2. 5-shot 20-way

## 48

1. 使用miniImageNet数据集

2. 5-shot，5-way，84crop

3. 使用Channel Pooling


## 49

1. 使用miniImageNet数据集

2. 5-shot，5-way，84crop

3. 不使用Channel Pooling


