# 运行描述

## 第1次运行

1. 提取特征的卷积层遵循3x3大小，除了第一层
含有stride以外其他都不含有stride

2. 均使用MaxPool

3. query样本分别求k个类样本

4. 由于梯度消失问题，没有按照论文中最后使用sigmoid

5. 使用交叉熵作为损失函数

6. 取消了全连接层，改为计算卷积结果的L1距离，并将负距离
输入到softmax中得到概率输出