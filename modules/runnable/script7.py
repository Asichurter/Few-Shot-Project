# 本脚本用于根据VirusTotal的扫描结果来将文件分类至对应的文件夹中

import os
import shutil
from modules.utils.imageUtils import convert

# labels_load_path = 'D:/BaiduNetdiskDownload/VirusShare_labels/malware.labels'
#
# file_load_path = 'D:/BaiduNetdiskDownload/VirusShare_00177/'
# file_out_path = 'D:/BaiduNetdiskDownload/VirusShare_Malwares/'
#
# families = set()
#
# with open(labels_load_path, 'r') as f:
#     lines = f.readlines()
#     for i,line in enumerate(lines):
#         line = line.replace('\t', ' ')
#         line = line.replace('\n', '')
#         items = line.split(' ')
#         md5, label = items
#         print(i, md5)
#         if label.startswith('SINGLETON'):
#             continue
#         # 若label第一次出现，则在目标处新建一个类文件夹
#         if label not in families:
#             families.add(label)
#             if not os.path.exists(file_out_path+label+'/'):
#                 os.mkdir(file_out_path+label+'/')
#         try:
#             shutil.move(file_load_path + 'VirusShare_' + md5, file_out_path + label + '/' + 'VirusShare_' + md5)
#         except FileNotFoundError:
#             continue

file_from_path = 'D:/peimages/PEs/virusshare_origin/'
file_to_path = 'D:/peimages/PEs/virusshare/'

threshold = 20

for i,label in enumerate(os.listdir(file_from_path)):
    file_amount = len(os.listdir(file_from_path+label+'/'))
    if file_amount >= threshold:
        print(label)
        shutil.move(file_from_path+label+'/', file_to_path+label+'/')


