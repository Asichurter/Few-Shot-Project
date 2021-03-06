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

## 第8次运行

1. 改变了数据集为修正后的数据集，采用随机抽取类内样本的方式，且采用224x224的随机裁剪

2. 训练了10000轮

## 第9次运行

1. 与第8次完全相同，只是训练了约31000轮

## 第10次运行

1. 修正了一个错误：在验证时，采样方法没有使用类内随机采样


## 第11次运行
1. 改变了生成类向量的策略：使用每个类内向量与类均值向量的负距离输入到softmax
中生成注意力系数来注意力对齐生成类向量

2. 学习率10000轮下降一次

## 第12次运行

1. 使用规定了大类为Trojan的数据集

## 第13次运行

1. 取消了大类规定，只使用大小为16~256的文件

## 第14次运行

1. 使用了新生成的无约束数据集，与12,13次作对比

## 第15次运行

1. 重新生成数据集重复了相同大类的实验，大类为trojan

## 第16次运行

1. 重新生成了数据集，重复限制大小的实验，大小的范围是16~1024

## 第17次运行

1. 修复了数据的错误，使得同类之间不同包含不属于一个类的样本

2. 无大小和超类约束

## 第18次运行

1. 重复17次实验，重新生成了数据集

## 第19次运行

1. 将生成原型向量的方式更改为修正后的注意力机制

2. 数据集使用无约束的数据集test

## 第20次运行

1. 重复第19次，只是将验证集改为了测试集

## 第21次运行

1. 重新生成了数据集，数据集分配数量不变 

2. 原型向量的生成方式为改进后的注意力机制

## 第22次运行

1. 原型向量的生成方式改为简单的向量均值用于与第21次做对比

## 第23次运行

1. 原型向量的生成方式改为改进前的注意力机制

## 第24次运行

1. 使用Transformer转换特征用于fine-tuning

2. 原型向量的生成方式为简单的转换后的均值向量

## 第25次运行

1. 使用Transformer+var 进行训练

2. 为了使得类间方差和类内方差处于同一数量级上，在类间方差上乘上放大系数

## 第26次运行

1. 测试10-way，5-shot的情况。为了加快训练速度减小显存消耗，qk调整为10

2. 学习率改为20000轮下降一次，gamma设置为0.1

## 第27次运行

1. 保持10-way，5-shot，使用修正后的注意力生成原型向量

## 第28次运行

1. 距离度量改用线性放大的余弦相似度

2. 修改了网络的输入方式，使得网络能够支持动态的n-way k-shot

## 第29次运行

1. 10-way 5-shot，测试10-way性能的同时，测试是否能用更高way训练
的模型来测试更低way的数据（用10-way测试5-way）

## 第30次运行

1. dataset中标准化数据的方式改为：通过计算训练集中样本均值和标准差，
利用该统计数据标注化所有数据，包括训练集和测试集

2. 生成原型向量的方式为单纯的均值向量

## 第31次运行

1. 将同类的样本数量从20提高到50（模型位于50_sample）

## 第32次运行

1. 使用了文件数据集的Dataloader测试速度

## 第33次运行

1. 使用了128x128模糊化的数据集

2. 学习率设置为15000轮下降一次

## 第34次运行

1. 修改卷积层的通道数为1->32->64->128->256

## 第35次运行

1. 使用了聚类的数据集cluster，该数据集通过了重新计算train的均值和标准差以后重新
指定的标准化参数

2. 修改了卷积层的通道参数，使得从第一个conv到第四个conv的通道数为
1->32->64->128->256

## 第36次运行

1. 使用聚类后的数据集

2. 使用修正后的注意力机制

## 第37次运行

1. 修改了网络结构：让最后一层的输出大小为4x4，使用了多层次的池化代替了原最后一层
的Maxpool

## 第38次运行

1. 修改了网络结构：增加至Conv5，通道数改为1->16->32->64->256->512

## 第39次运行

1. 修改了网络结构，通道数：1->16->32->64->128->256

2. 修改类别数n=10

## 第40次运行

1. 测试修正后的注意力机制效果

## 第41次运行

1. 测试修正前的注意力机制

## 第42次运行

1. 将度量方式改为修正后的cosine相似度

## 第43次运行

1. 利用Transformer将求得的嵌入转换后，使用修正后的注意力机制

2. 距离度量改回平方欧式距离

## 第44次运行

1. 使用了150训练集+30验证集+28测试集的数据集cluster_2

## 第45次运行

1. 测试5-shot 20-way情况

## 46

1. 10-shot 5-way实验

## 47

1. 10-shot 20-way

## 48

1. 使用virushare_20数据集

2. 5-shot 5-way

## 49

1. 使用virushare_20数据集

2. 5-shot 20-way

## 50

1. 使用virushare_20数据集

2. 5-shot 20-way

## 51

1. 使用virushare_20数据集

2. 10-shot 20-way

## 52

1. 使用virushare_20数据集

2. 10-shot 5-way

## 53

1. test数据集

2. 5-shot 5-way

## 54

1. test数据集

2. 10-shot 20-way

## 55

1. test数据集

2. 10-shot 5-way

## 56

1. test数据集

2. 5-shot 20-way

## 57

1. drebin_15数据集

2. 5-shot 5-way

## 58

1. drebin_10数据集

2. 5-shot 5-way

3. 20000 episode， 192 crop

## 59

1. drebin_10数据集

2. 5-shot 10-way

3. 20000 episode， 192 crop