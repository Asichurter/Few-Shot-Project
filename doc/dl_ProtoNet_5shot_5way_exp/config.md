# 描述记录

## 第1次运行

1. 按照论文实现，卷积核前两层有最大池化，后两层没有，
且前三层都有stride，最后一层没有

2. 仿照RN的实现，将迁入后的support先求出类原型向量，再与之做平方欧氏距离作为
softmax的输入，以该输出作为类别判定的概率分布

3. 使用负对数损失函数训练

4. 初始化采用的是RN原始代码中的初始化方式

## 第2次运行

1. 修改了网络结构，增加了stride的数量，使得卷积输出为每个通道1x1的图像

2. 增加了数据集规模，使得训练集有300个，测试集有111个

## 第3次运行

1. 修正了致命错误：距离没有取负就直接输入到了softmax中

2. 减小了数据集规模，训练集和测试集均为30个

3. 减小了训练次数至1000次，学习率100轮降低一次

## 第4次运行

1. 一共5000轮，学习率500轮下降一次(由于会引起严重得过拟合，因此还是采用只训练1000轮的设置，与第3次运行时的配置一样)

## 第5次运行

1. 改为使用SGD优化，同时将权重衰竭的系数增大到1e-3，动量值设置为0.9

2. 训练5000轮，500轮降低一次学习率

## 第6次运行

1. 权重衰竭降低至5e-4

## 第7次运行

1. 权重衰竭降低至1e-4

2. 训练20000轮，在10000轮时可以终止

3. 使用自定义的初始化方式，而不是论文中的方式