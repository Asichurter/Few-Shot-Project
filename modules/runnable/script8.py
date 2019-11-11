import os
import random as rd
from time import time
import shutil

src_path = 'D:/peimages/PEs/virusshare_origin/'
dst_path = 'D:/peimages/PEs/virusshare/'

sample_per_class = 20

for i,cls in enumerate(os.listdir(src_path)):
    print(i,cls)

    if not os.path.exists(dst_path+cls+'/'):
        os.mkdir(dst_path+cls+'/')

    rd.seed(time()%7365550)
    assert len(os.listdir(src_path+cls+'/')) >= sample_per_class, \
        "类：%s 的数量%d小于%d"%(cls, len(os.listdir(src_path+cls+'/')), sample_per_class)
    insts = rd.sample(os.listdir(src_path+cls+'/'), sample_per_class)

    for inst in insts:
        shutil.copy(src_path+cls+'/'+inst, dst_path+cls+'/'+inst)

