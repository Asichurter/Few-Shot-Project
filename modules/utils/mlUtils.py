import os
import numpy as np
import random as rd
from modules.utils.extract import extract_infos


#可执行文件的后续集合
EXES = ['exe', 'dll', 'ocx', 'sys', 'com']
#选取的标本的大小范围
SIZE_RANGE = [15, 3000]
#默认的良性文件目录
BENIGN_BASE = r'C:/Windows/'
#弃置不用的良性文件的目录
deprecated_benign = ['87机械合金幻彩版', '360', 'LuDaShi']
#恶性文件的默认目录
MALWARE_BASE = r'D:/pe/'
#良性文件路径存储文件
BENIGN_PATHS = ["D:/Few-Shot-Project/data/benign_Ddisk_data.npy","D:/Few-Shot-Project/data/benign_windowsAndProgFile_data.npy"]

def check_if_executable(path, size_thre=SIZE_RANGE):
    '''
    检查一个地址的文件扩展名是否是可执行文件
    :param path: 检查文件的绝对路径
    :param size_thre: 大小范围
    :return: 是否是可执行文件
    '''
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


def get_benign_exe_abspath(base=BENIGN_BASE):
    '''
    在windows目录下查找所有可执行文件的目录
    本函数必须在有管理员权限下才能使用
    因为遍历所有的良性可执行文件不可取，因此使用迭代器做到按需分配
    :param base: 良性文件的父目录
    :return: 良性文件的迭代器
    '''
    if os.path.isdir(base):
        for dirs in os.listdir(base):
            if dirs in deprecated_benign:
                continue
            #加上斜杠保证以后的递归能继续在文件夹中进行
            try:
                for ele in get_benign_exe_abspath(base+dirs+'/'):
                    if check_if_executable(ele):
                        yield ele
            except PermissionError:
                continue
    else:
        if check_if_executable(base):
            yield base

def get_malwares(mal_base=MALWARE_BASE, each_num=100, super_targets=None, sub_targets=None):
    '''
    按照给定的目标列表获得恶意样本数据
    :param mal_base: 恶意样本的路径
    :param each_num: 每个种类的数量
    :param super_targets: 待抽取的大类的列表
    :param sub_targets: 给定的恶意样本大类中的小类的列表
    :return: 恶意样本的数据和标签
    '''

    #先检查大类指定是否符合规范
    assert super_targets is None or type(super_targets)==list, "超类的targets对象类型无效！"
    #再检查小类指定是否符合规范
    if super_targets is None:
        if type(sub_targets)==list:
            assert len(sub_targets)==len(os.listdir(mal_base)), "在没有指定超类时，子类的指定不为None且数目不匹配！"
        else:
            assert sub_targets is None, "指定子类时，类型不为list也不为None！"
    else:
        if sub_targets is not None:
            assert len(sub_targets)==len(super_targets), "指定子类时，list长度与超类不一致！"
            for target in sub_targets:
                assert target is None or type(target)==list, "指定子类时，元素类型不为None也不为list！"


    data = []
    for ex_i, mal_type in enumerate(os.listdir(mal_base) if super_targets is None else super_targets):

        collections = os.listdir(str(mal_base + mal_type))
        if sub_targets[ex_i] is not None:
            #按照指定的目标过滤
            collections = list(filter(lambda x: x.split(".")[-2] in sub_targets[ex_i], collections))

        assert len(collections) >= each_num, "符合要求的大类样本数目: %d 小于指定抽取的数量: %d！" %(len(collections), each_num)
        sub_type = sub_targets[ex_i] if sub_targets is not None and sub_targets[ex_i] is not None else "all"

        # 采取在文件夹内抽样的形式，而非按顺序抽取
        samples = rd.sample(collections, each_num)
        for in_i, mal_name in enumerate(samples):
            print(ex_i, ' ', mal_type, sub_type, ' : ', in_i)
            pe_data = extract_infos(mal_base + mal_type + '/' + mal_name)
            if pe_data is None:
                continue
            data.append(pe_data)
            if in_i >= each_num - 1:
                break
    return data,[1]*len(data)

def get_benigns(num, benign_file=BENIGN_PATHS[0]):
    '''
    按照给定数量，从良性样本库中随机抽样得到良性样本和标签
    :param num: 指定的数目
    :param benign_file: 良性样本库路径
    :return: 良性样本数据和标签
    '''
    benign_datas = np.load(benign_file).tolist()
    assert len(benign_datas) >= num, "良性数据长度不足!良性总长度: %d, 需求长度: %d" %(len(benign_datas), num)
    datas = rd.sample(benign_datas, num)
    return datas,[0]*len(datas)


def mix_samples(mal_base=MALWARE_BASE, benign_base=BENIGN_BASE, each_num=100, split=0.5, seed=2, target_list=None):
    '''
    读取pe文件的特征同时向量化，将良性文件和恶性文件混合返回。本方法默认为每个目录抽取num个，可以指定需要抽取的文件夹
    :param mal_base: 恶性文件的父目录
    :param benign_base: 良性文件的父目录
    :param each_num: 每个类别抽样数目
    :param split: 训练集和测试集的比例
    :param seed: 随机种子
    :param target_list: 目标集合，可以指定要抽取哪些文件夹中的文件
    :return: 列表，依次为：训练集样本，训练集标签，测试集样本，测试集标签
    '''
    # old_num = 0
    # new_num = 0
    data,label = get_malwares(mal_base, each_num, target_list)
    mal_length = len(data)
    benign_data,benign_label = get_benigns(mal_length)
    data += benign_data
    label += benign_label

    data = np.array(data)
    label = np.array(label)

    #使用相同的种子来打乱数据和标签才能保证结果正确
    assert len(data)==len(label), '数据和标签数量不一致!'
    np.random.seed(seed)
    data = np.random.permutation(data)
    np.random.seed(seed)
    label = np.random.permutation(label)

    # train_d = data[:split]
    # train_d += data[mal_length:(mal_length + split)]
    # train_l = label[:split]
    # train_l += label[mal_length:(mal_length + split)]
    #
    # test_d = data[split:mal_length]
    # test_d += data[(mal_length + split):]
    # test_l = label[split:mal_length]
    # test_l += label[(mal_length + split):]
    #
    # return np.array(train_d), np.array(train_l), np.array(test_d), np.array(test_l)

    if split > 1:
        return data[:split], label[:split], data[split:], label[split:]
    elif split >= 0:
        threshold = int(len(data) * split)
        return data[:threshold], label[:threshold], data[threshold:], label[threshold:]
    else:
        return data, label


def normalize_data(data):
    '''
    将数据标准化，避免各维度上数据差异过大
    :param data: 带标准化的数据
    :return: 标准化后的数据
    '''
    mean = np.mean(data, axis=0)

    std = np.std(data, axis=0)
    normalize_func = lambda x: (x - mean) / std
    data = np.apply_along_axis(normalize_func, axis=1, arr=data)
    # 由于最后一个维度上，数据均为0，因此会出现除0错误而出现nan，因此需要将nan转为0后返回
    return np.nan_to_num(data)


#
def collect_save_data(path, normalize=True, num=100, seed=2, target_list=None, split=0):
    '''
    调用混合数据方法生成数据后保存至文件
    :param path: 保存路径
    :param normalize: 是否标准化，默认为是
    :param num: 每个种类抽取的数量
    :param seed: 随机种子
    :param target_list: 目标列表，可以指定需要抽取的种类
    :param split: 测试与训练集的分隔比例
    '''
    if split >= 0:
        train_data, train_label, test_data, test_label = mix_samples(each_num=num, seed=seed,
                                                                     split=split,
                                                                     target_list=target_list)
        np.save(path + 'raw_train_data.npy', train_data)
        np.save(path + 'raw_test_data.npy', test_data)
        np.save(path + 'train_label.npy', train_label)
        np.save(path + 'test_label.npy', test_label)
        if normalize:
            train_data = normalize_data(train_data)
            test_data = normalize_data(test_data)
            np.save(path + 'train_data.npy', train_data)
            np.save(path + 'test_data.npy', test_data)
        print('--- done! ---')
    else:
        data, label = mix_samples(each_num=num, seed=seed, target_list=target_list, split=split)
        #np.save(path + 'raw_data.npy', data)
        np.save(path + 'label.npy', label)
        if normalize:
            data = normalize_data(data)
            np.save(path + 'data.npy', data)
        print('---Done---')

def collect_class_data(path, train_num_each=100, test_num_each=200, seed=1,
                       train_targets=None, test_targets=None, raw=False,
                       no_benign=False, benign_split=False):
    '''
    从样本中获得单类缺省或者类型分割的数据
    :param path: 数据保存路径
    :param train_num_each: 训练集中，每个类别的抽取数量
    :param test_num_each: 测试集中每个类别的抽取数量
    :param train_targets: 训练集的抽取目标文件夹集合
    :param test_targets: 测试集的抽取目标文件夹集合
    :param raw: 保存原始数据还是标准化后的数据
    :param no_benign: 训练集中是否有良性样本数据
    :param benign_split: 是否在训练集和测试集中采用不同的良性样本集
    '''
    train_data,train_label = get_malwares(MALWARE_BASE, train_num_each, train_targets)
    test_data,test_label = get_malwares(MALWARE_BASE, test_num_each, test_targets)
    if not no_benign:
        if benign_split:
            benign_paths = rd.sample(BENIGN_PATHS, 2)
            benign_train_data, benign_train_label = get_benigns(len(train_data), benign_file=benign_paths[0])
            benign_test_data, benign_test_label = get_benigns(len(test_data), benign_file=benign_paths[1])
            train_data += benign_train_data
            train_label += benign_train_label
            test_data += benign_test_data
            test_label += benign_test_label
        else:
            benign_data,benign_label = get_benigns(len(train_data)+len(test_data))
            split_length = len(train_data)

            #将良性文件分割给训练集合测试集
            train_data += benign_data[:split_length]
            train_label += benign_label[:split_length]
            test_data += benign_data[split_length:]
            test_label += benign_label[split_length:]
    #如果选择无良性样本，则训练集中不应该出现良性样本，但测试集中应该存在良性样本已验证其是否能否区分恶意和非恶意，而不是良性或者非良性
    else:
        benign_data,benign_label = get_benigns(len(test_data))
        test_data += benign_data
        test_label += benign_label


    if raw:
        np.save(path+"train_data_raw.npy", train_data)
        np.save(path+"test_data_raw.npy", test_data)
        np.save(path+"train_label_raw.npy", train_label)
        np.save(path+"test_label_raw.npy", test_label)

    else:
        #数据合并再标准化，再将数据分还给训练和测试集
        data = train_data + test_data
        data = normalize_data(data)
        train_data = data[:len(train_data)]
        test_data = data[len(train_data):]

        #训练和测试集内部排列顺序的打乱
        assert len(train_data)==len(train_label) and len(test_data)==len(test_label), "数据和标签长度不一致！"
        np.random.seed(seed)
        train_data = np.random.permutation(train_data)
        np.random.seed(seed)
        train_label = np.random.permutation(train_label)
        np.random.seed(seed)
        test_data = np.random.permutation(test_data)
        np.random.seed(seed)
        test_label = np.random.permutation(test_label)

        #保存文件
        np.save(path+"train_data.npy", train_data)
        np.save(path+"train_label.npy", train_label)
        np.save(path+"test_data.npy", test_data)
        np.save(path+"test_label.npy", test_label)
    print("--- Done ---")

def collect_few_shot_data(save_path, super_class, sub_class=None, channel=5,
                          test_num=200,
                          benign_split=True,
                          seed=1):
    benign_paths = rd.sample(BENIGN_PATHS, 2)
    datas,labels = get_malwares(each_num=channel+test_num, super_targets=[super_class], sub_targets=[sub_class])

    np.random.seed(seed)
    datas = np.random.permutation(datas).tolist()
    train_data = datas[:channel]
    train_label = labels[:channel]
    test_data = datas[channel:]
    test_label = labels[channel:]

    if benign_split:
        train_benign,tr_l = get_benigns(channel, benign_paths[0])
        test_benign,te_l = get_benigns(len(test_data), benign_paths[1])

        train_data += train_benign
        train_label += tr_l
        test_data += test_benign
        test_label += te_l

    else:
        benigns,ben_l = get_benigns(len(train_data)+len(test_data), benign_file=benign_paths[0])
        np.random.seed(1+seed)
        benigns = np.random.permutation(benigns)

        train_data += benigns[:len(train_data)]
        train_label += ben_l[:len(train_data)]
        test_data += benigns[len(train_data):]
        test_label += ben_l[len(train_data):]

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    if sub_class is not None:
        sub_name = ""
        for n in sub_class:
            sub_name += (n+"+")
        sub_name = sub_name[:-1]
    else:
        sub_name = "all"
    np.save(save_path+sub_name+"_"+"train_data", train_data)
    np.save(save_path+sub_name+"_"+"train_label", train_label)
    np.save(save_path+sub_name+"_"+"test_data", test_data)
    np.save(save_path+sub_name+"_"+"test_label", test_label)



def make_benign_file(total, base, save_path, targets=None):
    '''
    将指定位置的文件视为良性文件，从基路径中指定的目录中，将pe文件的路径以字符串的形式写入到一个numpy文件中存储起来方便查找
    :param total: 总数量
    :param base: 路径的基目录
    :param save_path: 文件的保存位置
    :param targets: 指定的目标文件夹列表
    '''
    num = 0
    benign_all = []
    for dir in os.listdir(base) if targets is None else targets:
        if num >= total:
            break
        benign_gen = get_benign_exe_abspath(base+dir+"/")
        while num < total:
            try:
                print('benign: ', num)
                # 过滤吊最后的斜杠字符
                benign_path = next(benign_gen)[:-1]
                benign_all.append(benign_path)
                num += 1
            except StopIteration:
                break
    benign_all = np.array(benign_all)
    np.save(save_path, benign_all)
    print("---- Make Benign File Done ----")

def convert_benign_file_to_array(load_path, save_path):
    paths = np.load(load_path)
    datas = []
    i = 0
    for path in paths:
        print("benign ", i)
        try:
            pe_data = extract_infos(path)
        except FileNotFoundError:
            continue
        if pe_data is None:
            continue
        datas.append(pe_data)
        i += 1
    datas = np.array(datas)
    np.save(save_path, datas)
    print("---- Done ----")


if __name__ == "__main__":
    # 对应mlExp1
    # collect_save_data(path="D:/Few-Shot-Project/data/ExtClassEach200/",
    #                   normalize=False,
    #                   num=100,
    #                   seed=7,
    #                   target_list=['aworm','backdoor2','dos','email','exploit','net-worm','packed','rootkit','trojan3','virus'],
    #                   split=0.8)
    #对应mlExp2
    # collect_class_data(path="D:/Few-Shot-Project/data/ExtClassOneDefaultTestSplit/email/",
    #                     train_num_each=100,
    #                     test_num_each=1000,
    #                     seed=5,
    #                     train_targets=['aworm','dos','net-worm','exploit','virus','packed','rootkit','backdoor2','trojan3'],
    #                     test_targets=['email'],
    #                     raw=True,
    #                     benign_split=True)
    #对应mlExp3
    collect_few_shot_data(save_path="D:/Few-Shot-Project/data/ExtClassFewShot/virus/",
                          super_class="virus",
                          sub_class=["VB"],
                          channel=5,
                          test_num=250,
                          seed=41)
    #用于产生所有的良性文件路径
    # benign = get_benign_exe_abspath()
    # All = []
    # i = 1
    # while True:
    #     try:
    #         print('benign: ', i)
    #         # 过滤吊最后的斜杠字符
    #         benign_base = next(benign)[:-1]
    #         All.append(benign_base)
    #         i += 1
    #     except StopIteration:
    #         break
    # benign = get_benign_exe_abspath(base="C:/Program Files/")
    # while True:
    #     try:
    #         print('benign: ', i)
    #         # 过滤吊最后的斜杠字符
    #         benign_base = next(benign)[:-1]
    #         All.append(benign_base)
    #         i += 1
    #     except StopIteration:
    #         break
    # All = np.array(All)
    # np.save("D:/Few-Shot-Project/data/benign.npy", All)
    # make_benign_file(30000, "D:/", "D:/Few-Shot-Project/data/benign_Ddisk.npy",
    #                  targets=["Adobe", "Anaconda3", "Arduino", "CodeBlocks", "eclipse", "ffmpeg", "Git", "JsNode", "JAVA",
    #                           "mysql-8.0.11-winx64","Notepad++","R-3.6.0","RationalRose","VmWare"])
    #convert_benign_file_to_array("D:/Few-Shot-Project/data/benign_windowsAndProgFile.npy", "D:/Few-Shot-Project/data/benign_windowsAndProgFile_data.npy")

