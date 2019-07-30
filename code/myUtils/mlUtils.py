import os
import numpy as np
from .extract import extract_infos

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
            for ele in get_benign_exe_abspath(base+dirs+'/'):
                if check_if_executable(ele):
                    yield ele
    else:
        if check_if_executable(base):
            yield base


#
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
    data = []
    label = []
    # 获得良性文件的迭代器
    benign = get_benign_exe_abspath(base=benign_base)
    if target_list is None or type(target_list) == list:
        for ex_i, mal_type in enumerate(os.listdir(mal_base) if target_list is None else target_list):
            if mal_type != 'aworm':
                for in_i, mal_name in enumerate(os.listdir(str(mal_base + mal_type))):
                    print(ex_i, ' ', mal_type, ' : ', in_i)
                    pe_data = extract_infos(mal_base + mal_type + '/' + mal_name)
                    if pe_data is None:
                        continue
                    data.append(pe_data)
                    label.append(1)
                    if in_i >= each_num - 1:
                        break
            else:
                for child_dir in os.listdir(str(mal_base + mal_type)):
                    for in_i, mal_name in enumerate(os.listdir(str(mal_base + mal_type + '/' + child_dir))):
                        print(ex_i, ' ', child_dir, ' : ', in_i)
                        pe_data = extract_infos(mal_base + mal_type + '/' + child_dir + '/' + mal_name)
                        if pe_data is None:
                            continue
                        data.append(pe_data)
                        label.append(1)
                        if in_i >= (each_num - 1) / 2:
                            break
    else:
        raise Exception('待选列表不是None或者list而是一个非法类型: ', str(type(target_list)))
    mal_length = len(data)
    for i in range(mal_length):
        try:
            print('benign: ', i)
            # 过滤吊最后的斜杠字符
            benign_base = next(benign)[:-1]
            pe_data = extract_infos(benign_base)
            if pe_data is None:
                continue
            data.append(pe_data)
            label.append(0)
        except StopIteration:
            assert 1==0,'良性pe文件的数量不足'

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
