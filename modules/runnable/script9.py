import json
import os
import shutil
import random as rd
from time import time

from modules.utils.imageUtils import convert

src_path = 'H:/DL/Drebin/'
dst_path = 'D:/peimages/drebin/samples/'
dict_path = 'D:/peimages/drebin/labels.json'

# with open(dict_path, 'r') as l_f:
#     label = json.load(l_f)
#
# count = 0
# for dir_ in os.listdir(src_path):
#     if not os.path.isdir(src_path+dir_+'/'):
#         continue
#
#     curPath = src_path+dir_+'/'
#     for item in os.listdir(curPath):
#         print(count)
#         count += 1
#
#         l = label[item].replace('/','-')
#         if not os.path.exists(dst_path+l+'/'):
#             os.mkdir(dst_path+l+'/')
#         if not os.path.exists(dst_path+l+'/'+item+'.jpg'):
#             img = convert(curPath+item, method='normal', padding=False)
#             img.save(dst_path+l+'/'+item+'.jpg', 'JPEG')

cnt = 0
num = 10
copy_path = 'D:/peimages/New/drebin_10/'

for dir_ in os.listdir(dst_path):
    files = os.listdir(dst_path+dir_+'/')
    if len(files) >= num:
        print(cnt, dir_)
        cnt += 1

        os.mkdir(copy_path+dir_+'/')

        rd.seed(time()%7355605)
        candidates = rd.sample(files, num)
        for item in candidates:
            shutil.copy(dst_path+dir_+'/'+item, copy_path+dir_+'/'+item)





