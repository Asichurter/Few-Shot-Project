import os
import shutil
import numpy as np
import json

from modules.utils.nGram import FrqNGram
from modules.utils.extract import extract_infos

img_path = 'D:/peimages/New/virushare_20/test/'
pe_src_path = 'D:/peimages/PEs/virusshare_origin/'
# pe_src_path = 'H:/DL/Drebin/'
pe_dst_path = 'D:/peimages/PEs/virushare_20/test/'

# d = np.load('D:/Few-Shot-Project/data/clusters_0.5eps_20minnum.npy', allow_pickle=True).item()
def mk_dirs():
    for dir_ in os.listdir(img_path):
        if not os.path.exists(pe_dst_path+dir_):
            os.mkdir(pe_dst_path+dir_+'/')

def find_malware_path(img_name):
    img_mal_name = '.'.join(img_name.split('.')[:-1])   # 去掉结尾的jpg就成了对应的病毒名称
    print(img_mal_name)
    for mal_dir in os.listdir(pe_src_path):
        mals_in_dir = os.listdir(pe_src_path+mal_dir+'/')
        if img_mal_name in mals_in_dir:
            return pe_src_path+mal_dir+'/'+img_mal_name,img_mal_name

    assert False, '%s 并没有在病毒的路径中被找到!'%img_name

def find_invalid_pe_meta_class(path, N, ratio=0.5):
    invalid_class_list = []
    for i,cls in enumerate(os.listdir(path)):
        print(i,cls, end=' ')
        invalid_cnt = 0
        for item in os.listdir(path+cls+'/'):
            meta_data = extract_infos(path+cls+'/'+item, verbose=False)
            if meta_data is None:
                invalid_cnt += 1

        if invalid_cnt >= N*ratio:
            invalid_class_list.append(cls)
            print('invalid')
        else:
            print('valid')

    return invalid_class_list


# folders = ['drebin_10']
# # folders = ['cluster', 'test', 'virushare_20', 'drebin_10']
#
# path = 'D:/peimages/PEs/%s/train/'
# json_path = 'D:/peimages/PEs/%s/train/'
# N = 3
# L = 65535
#
# def make_ngram(path, dst_path):
#     for i,cls in enumerate(os.listdir(path)):
#         if not os.path.exists(dst_path+cls+'/'):
#             os.mkdir(dst_path+cls+'/')
#         for j,inst in enumerate(os.listdir(path+cls+'/')):
#             print(i,j)
#             ngram = FrqNGram(path+cls+'/'+inst,N, L, None)
#             with open(dst_path+cls+'/'+inst+'.json', 'w') as f:
#                 json.dump(ngram.Counter, f)
#
# for p in folders:
#     make_ngram(path%p, json_path%(p+'_json'))

# mk_dirs()
# for i,dir_ in enumerate(os.listdir(img_path)):
#     dir_path = img_path + dir_ + '/'
#     for img in os.listdir(dir_path):
#         print(i, dir_path+img)
#         mal_path,mal_name = find_malware_path(img)
#         shutil.copy(mal_path, pe_dst_path+dir_+'/'+mal_name)
#
# for dir_ in os.listdir(pe_dst_path):
#     items = os.listdir(pe_dst_path+dir_+'/')
#     if len(items) != 20:
#         print(dir_,':',len(items))

test_path = 'D:/peimages/PEs/virushare_20/train/'
print(find_invalid_pe_meta_class(test_path, N=20, ratio=0.5))



