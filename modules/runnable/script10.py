import csv
import os
import random as rd
from time import time
import shutil

from modules.utils.imageUtils import convert

# label_path = 'H:/MicrosoftDataset/MalwareCla/trainLabels.csv'
# src_path = 'H:/MicrosoftDataset/MalwareCla/microsoftPE/train/'
# dst_path = 'H:/MicrosoftDataset/MalwareCla/microsoft/train/'
#
# label_file = open(label_path, 'r')
# reader = csv.reader(label_file)
# for i,line in enumerate(reader):
#     if i==0:
#         continue        # skip label line
#     print(i,line)
#     name,label = line
#     if not os.path.exists(dst_path+label+'/'):
#         os.mkdir(dst_path+label+'/')
#     try:
#         img = convert(src_path+name+'.pe', method='normal', padding=False)
#         img.save(dst_path+label+'/'+name+'.jpg', 'JPEG')
#     except FileNotFoundError:
#         print(i,line,'not exist!')
#         continue

s_p = 'D:/peimages/microsoft/train/'
d_p = 'D:/peimages/New/microsoft/train/'
for i,d in enumerate(os.listdir(s_p)):
    os.mkdir(d_p+d+'/')
    items = os.listdir(s_p+d+'/')
    print(i,len(items))

    rd.seed(time()%7355605)
    samples = rd.sample(items, 40)
    for j,sample in enumerate(samples):
        print(i,j,sample)
        shutil.copy(s_p+d+'/'+sample, d_p+d+'/'+sample)

