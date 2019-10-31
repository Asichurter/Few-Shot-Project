import cv2
import numpy as np
import pylab as pl
from PIL import Image
import os

"""
img:图片数据
blocks:划分成blocks*blocks个patch
direction:garbo滤波器的方向
scale:滤波器的尺度
返回gist特征
"""
def getGists(imgs,blocks,direction,scale):
    '''
    get the gist feature array of the images
    :param imgs: image list
    :param blocks: the amount of blocks to be splitted in each axis
    :param direction: the amount of intervals to be splitted between [0, Π]
    :param scale: kernel sizes
    :return: gist arrays of each input image
    '''
    #1.获取滤波器
    filters = buildFilters(direction,scale)

    img_array = []
    for img in imgs:
        #2.分割图片
        img_arr = cropImage(img,blocks)
        #3.gist向量
        gist_arr = process(img_arr,filters)
        img_array.append(gist_arr)

    # array shape: blocks * blocks * direction * |scale|
    return  img_array

#分割图片
def cropImage(img,blocks):
    img_array = []
    w,h=img.size
    patch_with = w / blocks
    patch_height = h / blocks
    for i in range(blocks):
        for j in range(blocks):
            crop_image = img.crop((j*patch_with,i*patch_height,patch_with*(j+1),patch_height*(i+1)))
            img_array.append(crop_image)
            # name = str(i)+'行'+str(j)+'列'+'.jpeg'
            # crop_image.save(name)
    return img_array


#构建Gabor滤波器
def buildFilters(direction,scale):
    filters = []
    lamda = np.pi/2.0

    for theta in np.arange(0,np.pi,np.pi/direction):
        for k in range(4):
            kern = cv2.getGaborKernel((scale[k],scale[k]),1.0,theta,lamda,0.5,0,ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

#gist向量
def process(img_array,filters):
    res = []
    for img in img_array:
        img_ndarray = np.asarray(img)
        for filter in filters:
            accum = np.zeros_like(img_ndarray)
            for kern in filter:
                fimg = cv2.filter2D(img_ndarray, cv2.CV_8UC3, kern)
                np.maximum(accum, fimg, accum)
            average = np.mean(accum)
            res.append(average)
    round_res = np.round(res, 4)
    return round_res

if __name__ == '__main__':
    path = 'D:/peimages/New/cluster/train/Backdoor.Win32.Agobot/'
    imgs = []
    for img in os.listdir(path):
        img_obj = Image.open(path+img)
        imgs.append(img_obj)

    gist = getGists(imgs, blocks=8, direction=4, scale=[3,5,7,9])
    print(gist)
