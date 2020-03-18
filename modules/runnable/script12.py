import csv
import os
import shutil
from PIL import Image

csv_file_path = 'H:/BaiduNetdiskDownload/mini-imagenet/val.csv'
src_file_path = 'H:/BaiduNetdiskDownload/mini-imagenet/images/'
dst_dir_path = 'H:/BaiduNetdiskDownload/mini-imagenet/validate/'


# ------------------将图片按类别分到文件夹中---------------------
# f = open(csv_file_path, 'r')
# reader = csv.reader(f)
#
# for i,line in enumerate(reader):
#     print(i)
#     if i == 0:
#         continue
#
#     # line[0]为文件名，line[1]为类别
#     src_file = src_file_path + line[0]
#     dst_file_dir = dst_dir_path+line[1]+'/'
#
#     # 对第一个遇到的类样本，先创建文件夹
#     if not os.path.exists(dst_file_dir):
#         os.mkdir(dst_file_dir)
#
#     shutil.copy(src=src_file,
#                 dst=dst_file_dir+line[0])
#
# f.close()
# -----------------------------------------------------------

dst_dir_path = 'D:/peimages/New/miniImageNet/validate/'
# ------------------将图片重新整理为84x84的大小---------------------
for i,cls in enumerate(os.listdir(dst_dir_path)):
    for j,item in enumerate(os.listdir(dst_dir_path+cls+'/')):
        print(i,j)
        img = Image.open(dst_dir_path+cls+'/'+item)
        img = img.resize((84,84), Image.LANCZOS)
        img.save(dst_dir_path+cls+'/'+item)