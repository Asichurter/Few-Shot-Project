import os
import torch as t
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Sampler
import random as rd
import time

magicNum = 7355608

def get_seed():
    return time.time()%magicNum

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
    def __init__(self, base, n, transform=None, rd_crop_size=None, rotate=True):
        self.Data = []
        self.Label = []
        self.RandomCrop = T.RandomCrop(rd_crop_size) if rd_crop_size is not None else None
        self.Rotate = rotate
        # assert num_class==len(os.listdir(path)), "实际种类数目%d与输入种类数目不一致！"%(num_class, len(os.listdir(path)))
        for i,c in enumerate(os.listdir(base)):
            assert n == len(os.listdir(base+c+"/")), "实际类别%s内样本数目%d不等同输入样本数目%d！" % (c, n, len(os.listdir(base+c+"/")))
            for instance in os.listdir(base+c+"/"):
                self.Data.append(base+c+"/"+instance)
            self.Label += [i]*len(os.listdir(base+c+"/"))
        self.Transform = transform if transform is not None else T.Compose(
            [T.ToTensor(),
             T.Normalize([0.3934904], [0.10155067])]
        )  # mean=0.3934904, std=0.10155067)
    def __getitem__(self, index):
        img = Image.open(self.Data[index])
        if self.RandomCrop is not None:
            img = self.RandomCrop(img)
        # 依照论文代码中的实现，为了增加泛化能力，使用随机旋转
        if self.Rotate:
            rotation = rd.choice([0,90,180,270])
            img = img.rotate(rotation)
        img = self.Transform(img)
        label = self.Label[index]

        return img,label

    def __len__(self):
        return len(self.Data)

class FewShotFileDataset(Dataset):
    # 直接指向support set或者query set路径下
    def __init__(self, base, n, class_num, rd_crop_size=None, rotate=True, squre=True):
        self.Data = np.load(base, allow_pickle=True)
        self.Label = []
        self.CropSize = rd_crop_size
        self.Rotate = rotate
        self.Width = self.Data.shape[2] if squre else None
        # assert num_class==len(os.listdir(path)), "实际种类数目%d与输入种类数目不一致！"%(num_class, len(os.listdir(path)))
        for i in range(class_num):
            self.Label += [i]*n
        assert len(self.Label)==len(self.Data), "数据和标签长度不一致!(%d,%d)"%(len(self.Label),len(self.Data))

    def __getitem__(self, index):
        w = self.Width
        crop = self.CropSize
        img = t.FloatTensor(self.Data[index])
        if crop is not None:
            assert self.Width is not None and self.Data.shape[2]==self.Data.shape[3], "crop不能作用在非正方形图像上!"
            bound_width = w-crop
            x_rd,y_rd = rd.randint(0,bound_width),rd.randint(0,bound_width)
            img = img[:, x_rd:x_rd+crop, y_rd:y_rd+crop]
        # 依照论文代码中的实现，为了增加泛化能力，使用随机旋转
        if self.Rotate:
            rotation = rd.choice([0,1,2,3])
            img = t.rot90(img, k=rotation, dims=(1,2))
        label = self.Label[index]

        return img,label

    def __len__(self):
        return len(self.Data)

class FewShotFileUnbalanceDataset(Dataset):
    # 直接指向support set或者query set路径下
    def __init__(self, base, num_list, class_num, rd_crop_size=None, rotate=True, squre=True):
        self.Data = np.load(base, allow_pickle=True)
        self.Label = []
        self.CropSize = rd_crop_size
        self.Rotate = rotate
        self.Width = self.Data.shape[2] if squre else None
        assert class_num == len(num_list), "实际种类数目%d与输入种类数目不一致！"%(class_num , len(num_list))
        for i in range(class_num):
            self.Label += [i]*num_list[i]
        assert len(self.Label)==len(self.Data), "数据和标签长度不一致!(%d,%d)"%(len(self.Label),len(self.Data))
    def __getitem__(self, index):
        w = self.Width
        crop = self.CropSize
        img = t.FloatTensor(self.Data[index])
        if crop is not None:
            assert self.Width is not None and self.Data.shape[2]==self.Data.shape[3], "crop不能作用在非正方形图像上!"
            bound_width = w-crop
            x_rd,y_rd = rd.randint(0,bound_width),rd.randint(0,bound_width)
            img = img[:, x_rd:x_rd+crop, y_rd:y_rd+crop]
        # 依照论文代码中的实现，为了增加泛化能力，使用随机旋转
        if self.Rotate:
            rotation = rd.choice([0,1,2,3])
            img = t.rot90(img, k=rotation, dims=(1,2))
        label = self.Label[index]

        return img,label

class ImagePatchDataset(Dataset):
    def __init__(self, path, num_per_class, width=32):
        self.BasePath = path
        self.ClassNames = os.listdir(path)
        self.ItemNames = [os.listdir(path+c+'/') for c in self.ClassNames]
        self.N = num_per_class
        self.ImgWidth = 16

    def __getitem__(self, index):
        cls_index = index // self.N
        item_index = index % self.N

        img = Image.open(self.BasePath+
                         self.ClassNames[cls_index]+'/'+
                         self.ItemNames[cls_index][item_index])
        img = T.ToTensor()(img)

        # 返回图片序列：序列长度, 通道数, 图像长 / 宽
        return img.view(-1, 1, self.ImgWidth, self.ImgWidth)

    def __len__(self):
        return len(self.N * len(self.ClassNames))



class ClassSampler(Sampler):
    def __init__(self, cla, sub_classes, counter, num_per_class, shuffle, k, qk=None):
        # TODO:从指定的大类和子类中获取起始下标
        start = 0
        for i in range(cla):
            # 加上指定的大类之前的大类的子类数目x每个子类的样本数量=指定大类的起始下标
            start += counter[i]*num_per_class
        assert max(sub_classes)<=counter[cla], \
            "指定的子类中存在%d比当前大类存在子类数目%d下标更大的值！"%(max(sub_classes),counter[cla])
        self.sub_classes = sub_classes
        self.num_per_class = num_per_class
        self.shuffle = shuffle
        self.class_start_index = start
        if qk is None:
            self.Start = 0
            self.End = k
        else:
            self.Start = k
            self.End = qk + k

    def __iter__(self):
        batch = []
        for c in self.sub_classes:
            for i in range(self.Start, self.End):
                batch.append(self.class_start_index+self.num_per_class * c + i)
        if self.shuffle:
            rd.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

class SingleClassSampler(Sampler):
    def __init__(self, cls_index, N, n):
        assert n <= N, '单类采样数量大于单类总数量！'
        rd.seed(time.time()%magicNum)
        indexes = rd.sample([i for i in range(N)], n)
        self.Indexes = [cls_index*N+i for i in indexes]

    def __iter__(self):
        return iter(self.Indexes)

    def __len__(self):
        return 1

class SNAIL_sampler(Sampler):
    def __init__(self, k, n, total_class, num_per_class, batch_size):
        self.num_per_class = num_per_class
        self.batchSize = batch_size
        self.instances = []

        for i in range(batch_size):
            instances_seeds = rd.sample([j for j in range(magicNum)], n)
            classes = rd.sample([j for j in range(total_class)], n)
            instances = dict.fromkeys(classes)
            for cla, seed in zip(classes, instances_seeds):
                rd.seed(seed)
                instances[cla] = set(rd.sample([i for i in range(num_per_class)], k))

            # rd.seed(get_seed())
            # test_sample_label = rd.choice(classes)
            # all_indexes = set([j for j in range(num_per_class)])
            # candidate_indexes = all_indexes.difference(instances[test_sample_label])
            # rd.seed(get_seed())
            # test_sample_index = rd.choice(list(candidate_indexes))
            # instances[test_sample_label].add(test_sample_index)
            self.instances.append(instances)

    def __iter__(self):
        aggr_batch = []
        for batch in self.instances:
            minibatch = []

            rd.seed(get_seed())
            test_sample_label = rd.choice(list(batch.keys()))

            all_indexes = set([i for i in range(self.num_per_class)])
            test_sample_index = rd.sample(all_indexes.difference(batch[test_sample_label]), 1)[0]

            for c,instances in batch.items():
                for i in instances:
                    minibatch.append(self.num_per_class*c+i)
            rd.shuffle(minibatch)
            aggr_batch += minibatch
            aggr_batch.append(test_sample_label*self.num_per_class+test_sample_index)

        return iter(aggr_batch)

    def __len__(self):
        return 1


class RNSamlper(Sampler):
    def __init__(self, k, qk, instance_seeds, mode, classes, num_per_class, shuffle):
        '''
        用于组成训练时sample set/query set和测试时support set和test set的采样器\n
        sample和query来自相同的类中，均为采样得到的\n
        :param classes: 选到的要采样的类
        :param num_per_class: 每个类的最大样本数量
        :param shuffle: 是否随机打乱顺序
        '''
        self.classes = classes
        self.num_per_class = num_per_class
        self.shuffle = shuffle
        self.instances = dict.fromkeys(self.classes)
        if mode == 'train':
            # 为每一个类，根据其种子生成抽样样本的下标
            for cla,seed in zip(classes,instance_seeds):
                rd.seed(seed)
                self.instances[cla] = set(rd.sample([i for i in range(num_per_class)], k))
        elif mode == 'test':
            for cla,seed in zip(classes, instance_seeds):
                rd.seed(seed)
                train_instances = set(rd.sample([i for i in range(num_per_class)], k))
                test_instances = set([i for i in range(num_per_class)]).difference(train_instances)
                self.instances[cla] = rd.sample(test_instances, qk)

    def __iter__(self):
        batch = []
        for c,instances in self.instances.items():
            for i in instances:
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




def get_RN_sampler(classes, train_num, test_num, num_per_class, seed=None):
    if seed is None:
        seed = time.time()%1000000

    assert train_num+test_num <= num_per_class, "单类中样本总数:%d少于训练数量加测试数量:%d！"%(num_per_class, train_num+test_num)
    # instance_pool = [i for i in range(num_per_class)]
    # instances = rd.sample(instance_pool, train_num+test_num)
    # rd.seed(seed)
    # rd.shuffle(instances)
    #
    # train_instances= instances[:train_num]
    # test_instances = instances[train_num:]

    # 先利用随机种子生成类中的随机种子
    rd.seed(seed)
    instance_seeds = rd.sample([i for i in range(100000)], len(classes))

    return RNSamlper(train_num, test_num, instance_seeds, 'train', classes, num_per_class,False),\
           RNSamlper(train_num, test_num, instance_seeds, 'test', classes, num_per_class, True)

def get_RN_modified_sampler(classes, train_num, test_num, num_per_class):
    assert train_num+test_num <= num_per_class, "单类中样本总数:%d少于训练数量加测试数量:%d！"%(num_per_class, train_num+test_num)

    return RNModifiedSamlper(classes, num_per_class, shuffle=False, k=train_num),\
            RNModifiedSamlper(classes, num_per_class, shuffle=True, k=train_num, qk=test_num)

def get_class_sampler(dataset, class_num, n, k, qk, num_per_class):
    task = rd.randint(0,class_num-1)
    counter = dataset.SubClassCounter
    sampled_sub_classes = rd.sample([i for i in range(counter[task])], n)

    return ClassSampler(task, sampled_sub_classes, counter, num_per_class, False, k),\
            ClassSampler(task, sampled_sub_classes, counter, num_per_class, True, k, qk=qk)




if __name__ == '__main__':
    a = ImagePatchDataset(path='D:/peimages/New/cluster/train/',
                          num_per_class=20)
    p = a.__getitem__(123)
    # dataset = CNNTestDataset()
    # dataset = PretrainedResnetDataset(r'D:/peimages/validate/')