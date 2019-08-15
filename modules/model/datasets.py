import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Sampler
import random as rd


# 文件夹数据集
# 目录下有benign和malware两个文件夹
class DirDataset(Dataset):
    def __init__(self, base, transforms=None):
        data = []
        label = []
        for child_dir in os.listdir(base):
            path = base + child_dir + '/'
            columns = os.listdir(path)
            data += [os.path.join(path, column) for column in columns]
            # 添加样本数量个对应的标签
            label += [1 if child_dir == 'malware' else 0 for i in range(len(columns))]
            assert len(data) == len(label), '数据与标签的数量不一致!'
            # print(child_dir,':',len(columns))
        self.Datas = data
        self.Labels = label
        # 假设图像是单通道的
        # 归一化到[-1,1]之间
        if transforms:
            self.Transform = transforms
        else:
            self.Transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __getitem__(self, index):
        path = self.Datas[index]
        image = Image.open(path)
        if self.Transform is not None:
            image = self.Transform(image)
        return image, self.Labels[index]

    def __len__(self):
        return len(self.Datas)

#目录下有多个文件夹，每个文件夹是一个单独的种类
class ClassifyDataset(Dataset):
    def __init__(self, path, classes, transforms=None):
        assert len(os.listdir(path))==classes, "给定种类数目:%d和路径下的文件夹数目:%d 不一致！"%(len(os.listdir(path)), classes)
        data = []
        label = []
        for i,Dir in enumerate(os.listdir(path)):
            col_path = path+Dir+"/"
            columns = os.listdir(col_path)
            data += [os.path.join(col_path, column) for column in columns]
            label += [i]*len(columns)
        self.Data = data
        self.Label = label
        # 假设图像是单通道的
        # 归一化到[-1,1]之间
        if transforms:
            self.Transform = transforms
        else:
            self.Transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __getitem__(self, index):
        path = self.Data[index]
        image = Image.open(path)
        if self.Transform is not None:
            image = self.Transform(image)
        return image, self.Label[index]

    def __len__(self):
        return len(self.Data)


# torchvision自带的resnet适用的数据集
# 需要将图像转为224x224，同时还需要将单通道转为3通道
class PretrainedResnetDataset(DirDataset):
    def __init__(self, base):
        # 将图片尺寸调整为resnet的224x224尺寸，同时转化为三通道的图像，再标准化
        transforms = T.Compose([T.Grayscale(num_output_channels=3),
                                T.Resize(224),
                                T.ToTensor(),
                                T.Normalize([0.5], [0.5])])
        super().__init__(base, transforms)

    def __getitem__(self, index):
        # 先得到的图像不能转变为向量，因为需要
        im, label = super().__getitem__(index)
        return im, label
        # return im,label

    def __len__(self):
        return super().__len__()

# class Rotate:
#     def __init__(self, angle):
#         self.angle = angle
#
#     def __call__(self, x):
#         return x.rotate(self.angle)

class FewShotRNDataset(Dataset):
    # 直接指向support set或者query set路径下
    def __init__(self, base, n, transform=None):
        self.Data = []
        self.Label = []
        # assert num_class==len(os.listdir(path)), "实际种类数目%d与输入种类数目不一致！"%(num_class, len(os.listdir(path)))
        for i,c in enumerate(os.listdir(base)):
            assert n == len(os.listdir(base+c+"/")), "实际类别内样本数目%d不等同输入样本数目%d！" % (n, len(os.listdir(base+c+"/")))
            for instance in os.listdir(base+c+"/"):
                self.Data.append(base+c+"/"+instance)
            self.Label += [i]*len(os.listdir(base+c+"/"))
        self.Transform = transform if transform is not None else T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __getitem__(self, index):
        img = Image.open(self.Data[index])
        # 依照论文代码中的实现，为了增加泛化能力，使用随机旋转
        rotation = rd.choice([0,90,180,270])
        img = img.rotate(rotation)
        img = self.Transform(img)
        label = self.Label[index]

        return img,label

    def __len__(self):
        return len(self.Data)

class RNSamlper(Sampler):
    def __init__(self, instances, classes, num_per_class, shuffle):
        '''
        用于组成训练时sample set/query set和测试时support set和test set的采样器\n
        sample和query来自相同的类中，均为采样得到的
        :param classes: 选到的要采样的类
        :param num_per_class: 每个类的最大样本数量
        :param shuffle: 是否随机打乱顺序
        '''
        self.classes = classes
        self.num_per_class = num_per_class
        self.shuffle = shuffle
        self.instances = instances

    def __iter__(self):
        batch = []
        for c in self.classes:
            for i in self.instances:
                batch.append(self.num_per_class*c+i)
        if self.shuffle:
            rd.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

class RNModifiedSamlper(Sampler):
    def __init__(self, classes, num_per_class, shuffle, k, qk=None):
        '''
        用于组成训练时sample set/query set和测试时support set和test set的采样器\n
        sample和query来自相同的类中，sample和query的划分是固定的，不是采样得到的
        :param classes: 选到的要采样的类
        :param num_per_class: 每个类的最大样本数量
        :param shuffle: 是否随机打乱顺序
        '''
        self.classes = classes
        self.num_per_class = num_per_class
        self.shuffle = shuffle
        if qk is None:
            self.Start = 0
            self.End = k
        else:
            self.Start = k
            self.End = qk+k

    def __iter__(self):
        batch = []
        for c in self.classes:
            for i in range(self.Start, self.End):
                batch.append(self.num_per_class*c+i)
        if self.shuffle:
            rd.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_RN_sampler(classes, train_num, test_num, num_per_class, seed):
    assert train_num+test_num <= num_per_class, "单类中样本总数:%d少于训练数量加测试数量:%d！"%(num_per_class, train_num+test_num)
    instance_pool = [i for i in range(num_per_class)]
    instances = rd.sample(instance_pool, train_num+test_num)
    rd.seed(seed)
    rd.shuffle(instances)

    train_instances= instances[:train_num]
    test_instances = instances[train_num:]

    return RNSamlper(train_instances,classes,num_per_class,False),\
           RNSamlper(test_instances,classes,num_per_class,True)

def get_RN_modified_sampler(classes, train_num, test_num, num_per_class):
    assert train_num+test_num <= num_per_class, "单类中样本总数:%d少于训练数量加测试数量:%d！"%(num_per_class, train_num+test_num)

    return RNModifiedSamlper(classes, num_per_class, shuffle=False, k=train_num),\
            RNModifiedSamlper(classes, num_per_class, shuffle=True, k=train_num, qk=test_num)



if __name__ == '__main__':
    pass
    # dataset = CNNTestDataset()
    # dataset = PretrainedResnetDataset(r'D:/peimages/validate/')