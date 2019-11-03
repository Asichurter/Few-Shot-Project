import os
import shutil
import numpy as np

img_path = path = 'D:/peimages/New/test/train/'
pe_src_path = 'D:/pe/'
pe_dst_path = 'D:/peimages/PEs/test/train/'

# d = np.load('D:/Few-Shot-Project/data/clusters_0.5eps_20minnum.npy', allow_pickle=True).item()
def mk_dirs():
    for dir_ in os.listdir(img_path):
        if not os.path.exists(pe_dst_path+dir_):
            os.mkdir(pe_dst_path+dir_+'/')

def find_malware_path(img_name):
    img_mal_name = '.'.join(img.split('.')[:-1])   # 去掉结尾的jpg就成了对应的病毒名称
    print(img_mal_name)
    for mal_dir in os.listdir(pe_src_path):
        mals_in_dir = os.listdir(pe_src_path+mal_dir+'/')
        if img_mal_name in mals_in_dir:
            return pe_src_path+mal_dir+'/'+img_mal_name,img_mal_name

    assert False, '%s 并没有在病毒的路径中被找到!'%img_name

mk_dirs()
for i,dir_ in enumerate(os.listdir(img_path)):
    dir_path = img_path + dir_ + '/'
    for img in os.listdir(dir_path):
        print(i, dir_path+img)
        mal_path,mal_name = find_malware_path(img)
        shutil.copy(mal_path, pe_dst_path+dir_+'/'+mal_name)


