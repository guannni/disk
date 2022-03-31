#
# 勿轻易运行！！！输出文件易覆盖！！
#
# 输出椭圆位置方向的程序
import pims
import cv2
import func
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os.path
from skimage import morphology

warnings.filterwarnings('ignore')

def traversalDir_FirstDir(path):  # 返回一级子文件夹名字
    list = []
    if (os.path.exists(path)):    #获取该目录下的所有文件或文件夹目录路径
        files = glob.glob(path + '\\*' )
        # print(files)
        for file in files:            #判断该路径下是否是文件夹
            if (os.path.isdir(file)):                #分成路径和文件的二元元组
                h = os.path.split(file)
                # print(h[1] )
                list.append(h[1])
        return list

def predo(img, threshed = 180):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    h, w = img.shape
    # print(h,w)
    img = img.transpose()#[0]  # 取一维
    img = img.T  # 转置
    # add a mask for the container & Binarization
    r_m = 4.7 / 10. * h
    mask_index = func.create_circular_mask(h, w, center=[h / 2 -6, w / 2 - 5],
                                           radius=r_m-2)  # TODO: change the region of the stage#func.create_circular_mask(h, w, center=[h / 2 + 6.5, w / 2 - 2], radius=r_m - 1)  # TODO: change the region of the stage
    img_m = img.copy()
    img_m[~mask_index] = img.mean()  # image array with mask
    mask = np.zeros((h, w), np.uint8)
    mask[mask_index] = 255  # mask array
    th, img_t = cv2.threshold(img_m, threshed, 255, cv2.THRESH_BINARY)
    return img_t
#----------这一部分测试椭圆识别情况
# images = pims.ImageSequence('D:\\guan2019\\1_disk\\1.2_pics\\PeakFixedFreDiff\\1_1\\70_3\\*.jpg', process_func=predo)
# print(images)
# images = images[80000]  # We'll take just the first 10 frames for demo purposes.
#
#
# plt.figure()
# plt.imshow(images, cmap=plt.cm.gray)
# plt.show()
#
# th, threshed = cv2.threshold(images, 200, 255, cv2.THRESH_BINARY)
# cnts, hiers = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # cv2.RETR_LIST——无从属关系
# ellipse = cv2.fitEllipse(cnts[0])
#
# print(ellipse)
# a=cv2.ellipse(images,ellipse,(0,255,0),1)
#
# plt.figure()
# plt.imshow(a)
# plt.show()
path1 = 'D:\\guan2019\\1_disk\\1.2_pics\\PeakFixedFreDiff\\1_1\\'  # 读入路径
filename = traversalDir_FirstDir(path1)
file_n = [path1 + name for name in filename]
print(file_n)

path2 = 'C:\\Users\\guan\\Desktop\\'  # 输出路径
store_n = [path2 + name + '.h5' for name in filename]
print(store_n)

# print([i for i in range(1)])

for i in range(20,21):#len(file_n)):  #  !!千万别错了！！！，会覆盖，

    images = pims.ImageSequence(file_n[i], process_func=predo)
    # images = images[4180:]  # We'll take just the first 10 frames for demo purposes.

    # plt.figure()
    # plt.imshow(images, cmap=plt.cm.gray)
    # plt.show()

    store =pd.HDFStore(store_n[i], complib='blosc')
    c = 0

    for image in images:
        image = np.array(image)
        # print(image.shape)

        threshed0 = 180  #  TODO: change the highest thresheld

        th, threshed = cv2.threshold(image, threshed0, 255, cv2.THRESH_BINARY)
        cnts, hiers = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]  # cv2.RETR_LIST——无从属关系
        # print(len(cnts[0]), threshed0)  # 轮廓上的点的数目

        while True:  # 变化阈值
            if len(cnts[0]) <= 5:
                threshed0 -= 5
                image = pims.ImageSequence(file_n[i])[c]
                image = predo(image, threshed=threshed0)
                th, threshed = cv2.threshold(image, threshed0, 255, cv2.THRESH_BINARY)
                cnts, hiers = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]  # cv2.RETR_LIST——无从属关系
                print(len(cnts[0]), threshed0)  # 轮廓上的点的数目
                # plt.figure()
                # plt.imshow(image, cmap=plt.cm.gray)
                # plt.show()
            else:
                break




        names = ['center', '(2b,2a)', 'theta']
        ellipse = list(cv2.fitEllipse(cnts[0]))  # tuple
        # print(ellipse)
        ellipse = tuple(ellipse)
        ellipse_dict = dict(zip(names, ellipse))
        print(ellipse_dict)

        c += 1

        store.append(key='center', value=pd.DataFrame((ellipse_dict['center'],)))  # 参数输出
        store.append(key='(2b,2a)', value=pd.DataFrame((ellipse_dict['(2b,2a)'],)))
        store.append(key='theta', value=pd.DataFrame((ellipse_dict['theta'],)))
        print(c)

    store.close()
    print('--')
    print(i)
    del store,images
