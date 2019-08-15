import torch as t
import numpy as np
import PIL.Image as Image
import os
from modules.utils.extract import extract_infos
import random
import shutil
#import pandas as pd
import warnings
import sklearn
import random as rd

HOME = r'C:/Users/10904/Desktop/'
EXES = ['exe', 'dll', 'ocx', 'sys', 'com']
BENIGN_BASE = r'C:/Windows/'
MALWARE_BASE = r'D:/pe/'
TEST_CHILD_DIR = ['backdoor1/', 'net-worm/']
SIZE_RANGE = [15, 3000]

deprecated_benign = ['87机械合金幻彩版', '360', 'LuDaShi']

WIDTH = 256
WIDTH_SIZE = 10
UNIT = 1 / 25

next_times = 0
return_times = 0

#检查一个地址的文件扩展名是否是可执行文件
def check_if_executable(path, size_thre=SIZE_RANGE):
    try:
        #需要去掉最后一个斜杠/
        extension_name = path[:-1].split('.')[-1]
        #除以1024单位为千字节KB
        size = int(os.path.getsize(path[:-1])/1024)
        #只有是pe文件且大小在范围之内的文件的绝对路径才会被返回
        return extension_name in EXES and size >= size_thre[0] and size <= size_thre[1]
    except FileNotFoundError:
        return False
    except OSError:
        return False

#在windows目录下查找所有可执行文件的目录
#本函数必须在有管理员权限下才能使用
def get_benign_exe_abspath(base=BENIGN_BASE):
    if os.path.isdir(base):
        for dirs in os.listdir(base):
            if dirs in deprecated_benign:
                continue
            #加上斜杠保证以后的递归能继续在文件夹中进行
            for ele in get_benign_exe_abspath(base+dirs+'/'):
                if check_if_executable(ele):
                    yield ele
    else:
        if check_if_executable(base):
            yield base

def convert_to_images(base, destination, mode='file', method='normal',
                      padding=False, num_constrain=None, sample=False, cluster=None):
    '''
    base:目标文件或者目标所在的文件夹\n
    destination:转换后存储的文件夹\n
    mode:转换的模式：单个文件还是该文件夹下所有的文件\n
    method:转换的方法，是否要标准化\n
    padding:是否填充0而不拉伸图像\n
    num_constrain:数量限制。不填就是不设上限
    sample:是否采样
    cluster:是否只转换指定的名称簇
    '''
    assert method in ['plain', 'normal'], '选择的转换算法不在预设中！'
    assert mode in ['file', 'dir'], '转换的对象类型不在预设中！'
    # 健壮性处理
    if destination[-1] != '/':
        destination += '/'
    if type(base) is not str:
        num = 0
        while num < num_constrain:
            try:
                benign_path = next(base)[:-1]
            except PermissionError:
                continue
            except StopIteration:
                warnings.warn("迭代器内部总数量:%d 小于需求数量: %d"%(num, num_constrain), RuntimeWarning)
                break
            benign_name = benign_path.split('/')[-1]
            print(num)
            # 为了不在相同名字的文件下重复覆盖来无意义增加num，添加时判断是否同名者已经存在
            if os.path.exists(str(destination + benign_name + '.jpg')):
                continue
            im = convert(benign_path, method, padding)
            im.save(destination + benign_name + '.jpg', 'JPEG')
            num += 1
        return
    elif mode == 'dir':
        if not os.path.isdir(base):
            raise Exception(base + ' is not a director!\n')
        files = os.listdir(base)
        files = list(filter(lambda x: cluster in [x.split(".")[-2], x.split(".")[-2]], files)) if cluster is not None else files
        # assert cluster is None or not sample, '限制名字和采样不能同时进行！'
        assert len(files)>=num_constrain, "规定cluster以后，数量:%d不够到num_constrain:%d!"%(len(files), num_constrain)
        if sample:
            files = random.sample(files, num_constrain)
        num = 0
        for one in files:
            if num_constrain is not None and num == num_constrain:
                break
            print(num)
            im = convert(base + one, method, padding)
            im.save(destination + one + '.jpg', 'JPEG')
            num += 1

    elif mode == 'file':
        if os.path.isdir(base):
            raise Exception(base + ' is indeed a directory!\n')
        im = convert(base, method, padding)
        name = base.split('/')[-1]
        im.save(destination + name + '.jpg', 'JPEG')


#
def convert(path, method, padding):
    '''
    单个图像的转换函数，返回Image对象\n
    path:文件的路径\n
    method:使用的转换方式，plain:256宽度，长度任意 normal:先正方形，再转为256x256
    padding:对于normal方式，不足256时是否填充0
    '''
    file = open(path, "rb")
    image = np.fromfile(file, dtype=np.byte)
    im = None
    if method == 'plain':
        # 将不足宽度大小的剩余长度的像素点都过滤掉
        if image.shape[0] % WIDTH != 0:
            image = image[:-(image.shape[0] % WIDTH)]
        # print(image.shape)
        image = image.reshape((-1, WIDTH))
        image = np.uint8(image)
        im = Image.fromarray(image)
    else:
        crop_w = int(image.shape[0] ** 0.5)
        image = image[:crop_w ** 2]
        image = image.reshape((crop_w, crop_w))
        image = np.uint8(image)
        if padding and crop_w < WIDTH:
            image = np.pad(image, (WIDTH - crop_w), 'constant', constant_values=(0))
        im = Image.fromarray(image)
        im = im.resize((WIDTH, WIDTH), Image.ANTIALIAS)
    file.close()
    return im

def make_noise_image(path, num, size=(WIDTH,WIDTH), mode="uniform", type="JPEG", prefix="", constant=0):
    postfix={"JPEG":"jpg", "PNG":"png"}
    for i in range(1, num+1):
        print(i)
        if mode == "uniform":
            img = np.random.randint(0,256,size, dtype=np.uint8)
        elif mode == "gauss":
            img = np.uint8(np.random.normal(128, 40, size=size))
        elif mode == "constant":
            img = np.ones(size, dtype=np.uint8)*constant
        else:
            assert False, "选择的生成数据的模式 %s 并不支持！"%mode
        img = Image.fromarray(img)
        #print(path+prefix+str(i)+"."+postfix[type])
        img.save(path+prefix+str(i)+"."+postfix[type], type)


def check_continuing_decrease(history, window=3):
    '''
    用于检测提前终止的条件\n
    history:历史记录的列表\n
    window:检测的窗口大小，越大代表检测越长的序列\n
    '''
    if len(history) <= window:
        return False
    decreasing = True
    for i in range(window):
        decreasing ^= (history[-(i + 1)] < history[-(i + 2)])
    return decreasing


def create_malware_images(dest=r'D:/peimages/validate/', base=r'D:/pe/', num_per_class=80, using=None):
    '''
    从每个恶意代码类中随机抽取一定量的样本转为图片放入指定文件夹中\n
    dest:目标文件夹\n
    base:恶意代码文件夹\n
    num_per_class:每个类挑选的数量。不足该数量时会取总数量一半\n
    '''
    if base[-1] != '/':
        base += '/'
    num = 0
    all_columns = os.listdir(base)
    if using is not None:
        for using_one in using:
            assert using_one in all_columns, "使用列表中的%s不在待选列表中！"%using_one
    using = all_columns if using is None else using
    for child in using:
        child_columns = os.listdir(base + child)
        size = num_per_class if len(child_columns) > num_per_class else int(len(child_columns) / 2)
        samples = random.sample(child_columns, size)
        for sample in samples:
            path = base + child + '/' + sample
            convert_to_images(path, destination=dest, mode='file',
                              padding=False)
            num += 1
            print(num)


def split_datas(src=r'D:/peimages/test for cnn/no padding/malware/', dest=r'D:/peimages/validate/malware/',
                ratio=0.2, mode='x'):
    '''
    将生成的样本按比例随机抽样分割，并且移动到指定文件夹下，用于训练集和验证集的制作
    src:源文件夹
    dest:目标文件夹
    ratio:分割比例或者最大数量
    '''
    assert mode in ['c', 'x'], '选择的模式错误，只能复制c或者剪切x'
    All = os.listdir(src)
    size = int(len(All) * ratio) if ratio < 1 else ratio
    assert len(All) >= size, '分割时，总数量没有要求的数量大！'
    samples_names = random.sample(All, size)
    num = 0
    for item in All:
        if item in samples_names:
            num += 1
            path = src + item
            if mode == 'x':
                shutil.move(path, dest)
            else:
                shutil.copy(src=path, dst=dest)
            print(num)


def create_benign(dest, num,
                  base='D:/peimages/benign_collection/',
                  using=[0, 1, 2],
                  using_dict={0: '来自windows/', 1: '来自C盘的ProgramFiles/',
                              2: '来自C盘的ProgramFilesx86/', 3: "来自D盘/"},
                  redundant=200):
    max_num = num + redundant  # 为了防止重名文件出现的数量遗漏
    using_list = [using_dict[i] for i in using]
    for candidate in using_list:
        split_datas(src=base + candidate, dest=dest, ratio=int(max_num / len(using_list)), mode='c')
    files = os.listdir(dest)
    assert len(files) >= num, '数量不够num！'
    # 为了达到指定的数量，应该随机删去一些为了防止重新多设置的文件
    delete_files = random.sample(files, len(files) - num)
    for file in files:
        if file in delete_files:
            os.remove(dest + file)


def validate(model, dataloader, Criteria, return_predict=False):
    '''
    使用指定的dataloader验证模型\n
    model:训练的模型\n
    dataloader:验证的数据加载器\n
    criteria:损失函数\n
    '''
    val_a = 0
    val_c = 0
    val_loss = 0.
    real_label = []
    predict_label = []
    # 将模型调整为测试状态
    model.eval()
    for data, label in dataloader:
        data = data.cuda()
        out = model(data)

        # 同训练阶段一样
        labels = [[1, 0] if L == 0 else [0, 1] for L in label]
        labels = t.FloatTensor(labels).cuda()

        loss = Criteria(out, labels)
        val_loss += loss.data.item()
        pre_label = t.LongTensor([0 if x[0] >= x[1] else 1 for x in out])

        val_a += pre_label.shape[0]
        val_c += (pre_label == label).sum().item()

        real_label += label.int().tolist()
        predict_label += pre_label.int().tolist()

    if not return_predict:
        return val_c / val_a, val_loss
    else:
        return val_c / val_a, val_loss, np.array(real_label), np.array(predict_label)

def classfy_validate(model, dataloader, Criteria, classes, return_predict=False):
    '''
    使用指定的dataloader验证模型\n
    model:训练的模型\n
    dataloader:验证的数据加载器\n
    criteria:损失函数\n
    '''
    val_a = 0
    val_c = 0
    val_loss = 0.
    real_label = []
    predict_label = []
    # 将模型调整为测试状态
    model.eval()
    for data, label in dataloader:
        data = data.cuda()
        out = model(data)

        # 同训练阶段一样
        # labels = sklearn.preprocessing.label_binarize(label, [i for i in range(classes)])
        labels = t.LongTensor(label).cuda()
        label = label.cuda()

        loss = Criteria(out, labels)
        val_loss += loss.data.item()
        pre_label = t.argmax(out, dim=1)

        val_a += pre_label.shape[0]
        val_c += (pre_label == label).sum().item()

        real_label += label.int().tolist()
        predict_label += pre_label.int().tolist()

    if not return_predict:
        return val_c / val_a, val_loss
    else:
        return val_c / val_a, val_loss, np.array(real_label), np.array(predict_label)

if __name__ == "__main__":
    next_times = 0
    return_times = 0
    '''
    path = get_benign_exe_abspath()
    for i,p in enumerate(path):
        if i >= 10:
            break
        convert_to_images()
        #print(os.path.getsize(p[:-1])/1024)
        #print(p+'\n')
    #print(check_if_executable(r'C:/Windows/System32/1029/VsGraphicsResources.dll/'))
    '''
    # convert_to_images(base=r'D:/pe/virus/',
    #                   destination=r'D:/peimages/class default/validate/malware/',
    #                   mode='dir',
    #                   padding=False,
    #                   num_constrain=200)
    # benign = get_benign_exe_abspath(base=r'C:/Program Files (x86)/')#base=r'C:/Program Files/'
    #
    # split_datas(src=r'D:/peimages/one class 4/train/benign/',
    #             dest=r'D:/peimages/few-shot test/class4_basedon_class1/train/benign/',
    #             ratio=205,
    #             mode='c')
    # split_datas(src=r'D:/peimages/few-shot test/class4_basedon_class1/train/benign/',
    #             dest=r'D:/peimages/few-shot test/class4_basedon_class1/validate/benign/',
    #             ratio=200,
    #             mode='x')
    # create_benign(dest='D:/peimages/oneClasses/trojan1.Buzus/train/benign/',num=1200)
    # create_malware_images(dest=r'D:/peimages/one class 2/extern validate/malware/',
    #                       num_per_class=30,
    #                       deprecated=['aworm','trojan0', 'trojan1', 'trojan2', 'trojan3-2',
    #                                   'trojan4', 'trojan5'])
    # convert_to_images(base=r'D:/pe/trojan0/',
    #                   destination=r'D:/peimages/one class 2/train/malware/',
    #                   mode='dir',
    #                   padding=False,
    #                   num_constrain=1200,
    #                   cluster='OnLineGames',
    #                   sample=False)
    convert_to_images(base=r'D:/pe/trojan5/',
                      destination=r'D:/peimages/New/RN_5shot_5way_exp/train/29/',
                      mode='dir',
                      padding=False,
                      num_constrain=20,
                      cluster='Pixoliz',
                      sample=True)


    #为了将D盘的文件生成为良性文件集合
    # targets = ["Adobe", "Anaconda3", "Arduino", "CodeBlocks", "eclipse",
    #            "ffmpeg", "Git", "JsNode", "JAVA",
    #            "mysql-8.0.11-winx64","Notepad++","R-3.6.0","RationalRose","VmWare"]
    # for target in targets:
    #     benign = get_benign_exe_abspath("D:/"+target+"/")
    #     convert_to_images(benign,destination='D:/peimages/benign_collection/来自D盘/',
    #                        mode='dir',padding=False,num_constrain=3000)

    # 生成整体测试数据集，训练和测试用良性数据集之间是分开的
    # create_benign(dest="D:/peimages/New/class_default_noisebenign_exp/backdoor_default/train/benign/",
    #               num=450,
    #               using=[0])
    # create_malware_images(dest="D:/peimages/New/class_default_noisebenign_exp/backdoor_default/train/malware/",
    #                       num_per_class=100,
    #                       using=["aworm","dos","email","virus",
    #                              "exploit","net-worm","trojan0","packed","rootkit"])
    # create_malware_images(dest="D:/peimages/New/RN_5shot_5way_exp/train/query/0/",
    #                       num_per_class=30,
    #                       using=["backdoor1"])
    # split_datas(src='D:/peimages/New/class_default_retrain_exp/backdoor_default/train/malware/',
    #             dest='D:/peimages/New/class_default_retrain_exp/backdoor_default/validate/malware/',
    #             ratio=1000,
    #             mode="x")
    # make_noise_image(path="D:/peimages/New/class_default_noisebenign_exp/backdoor_default/train/benign/",
    #                  num=450, prefix="gauss_noise_", mode="gauss")

