# GIST特征
import os
import shutil
import random

def convert_img_to_gist_file(src_path, dest_path):
    print('converting...')
    gist_exe_path = 'D:/DL/GIST-global-Image-Descripor-master/GIST-global-Image-Descripor-master/gist-tool/gist.exe'
    shell_temp = '%s -i %s -o %s'

    os.system(shell_temp%(gist_exe_path, src_path, dest_path))

def cumulate_gist_data():
    s_path = 'D:/peimages/New/cluster/gist/train/'
    d_path = 'D:/peimages/New/cluster/gist/train.npy'

    # for c in os.listdir()

def mv_slcsamples_to_dir(indexes, src_path, dest_path):
    labels = {}

    # 清空temp文件夹
    for del_item in os.listdir(dest_path):
        os.remove(dest_path+del_item)

    # 逐一将图像复制到temp中
    dir_names = os.listdir(src_path)
    for label,index in enumerate(indexes):
        slc_dir = dir_names[index] + '/'
        print(slc_dir)
        for item in os.listdir(src_path+slc_dir):
            shutil.copy(src_path+slc_dir+item, dest_path+item)
            # 文件对应的标签不是原文件中的文件夹下标，而是这k个文件夹的相对下标
            labels[item] = label

    return labels

if __name__ == '__main__':
    s_path = 'D:/peimages/New/cluster/train/'
    d_path = 'D:/peimages/New/cluster/gist/train/'
    temp_path = 'D:/peimages/New/cluster/gist/temp/'

    k = 5
    total_size = len(os.listdir(s_path))
    slc_indexes = random.sample([i for i in range(total_size)], k)

    # mv_slcsamples_to_dir(slc_indexes, s_path, temp_path)
    # convert_img_to_gist_file(temp_path, d_path)
    with open(d_path+'/gist.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(len(line))
        # while True:
        #     command = input('command:')
        #     if command != 'quit':
        #         exec(command)
        #     else:
        #         break
    # s_path = 'D:/peimages/New/cluster/gist/train/'
    # for c in os.listdir(s_path):
    #     with open(s_path+c+'/gist.txt', 'r') as f:
    #         lines = f.readlines()
    #         # print(c,len(lines[0]))
    #         while True:
    #             command = input(c)
    #             if command != 'quit':
    #                 exec(command)
    #             else:
    #                 break

