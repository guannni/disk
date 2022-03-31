# 读取hdf文件
# 读取数据长度，计算active&inactive比例
 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
import seaborn as sns
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel,MoffatModel, VoigtModel,Pearson7Model, StudentsTModel,DampedOscillatorModel,ExponentialModel,ExponentialGaussianModel,ExpressionModel,SkewedGaussianModel
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator
from scipy import interpolate
import matplotlib

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0


# -----------------------------------------------

def traversalDir_FirstDir(path):  # 返回一级子文件夹名字
    list = []
    if (os.path.exists(path)):  # 获取该目录下的所有文件或文件夹目录路径
        files = glob.glob(path + '\\*')
        # print(files)
        for file in files:  # 判断该路径下是否是文件夹
            if (os.path.isdir(file)):  # 分成路径和文件的二元元组
                h = os.path.split(file)
                print(h[1] )
                list.append(h[1])
        return list

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def frac_reduc(n,m):
    n = int(n)
    m = int(m)
    for i in range(2, n):
        while (n % i == 0 and m % i == 0):
            n = n // i
            m = m // i
    return (n,m)


# 
# path2_a = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\active\\'  # 60Hz active
# path2_in = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\inactive\\'  # 60Hz inactive
path2_in = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\inactive_all0\\'  # 5g all
path2_a = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\active_all0\\'  # 5g all

filename_a = [name for name in os.listdir(path2_a)]  # active 文件夹
filename_ina = [name for name in os.listdir(path2_in)]  # inactive 文件夹
filename = filename_a if len(filename_a)>len(filename_ina) else filename_ina  # 选择长的为标准



delta_dict = {}
label = []
for p in range(len(filename)):
    if filename[p] in filename_a:
        path_3 = path2_a + filename[p] + '\\'
        filename_1 =[os.path.splitext(name)[0] for name in os.listdir(path_3) if name.endswith('.h5')]  # 每个fre下文件名
        file_n1 = [path_3 + name + '.h5' for name in filename_1]  # 每个fre下pdf_delta的存储文件
        print(filename_1, file_n1)

        for i in range(len(file_n1)):
            store = pd.HDFStore(file_n1[i], mode='r')
            print(store.keys())
            center_1 = store.get('center').values  # numpy array
            theta_1 = store.get('theta').values
            store.close()

            # todo : 改变delta间隔------------------
            # 这里改了要在存储路径改一下！
            step = 1
            center = center_1[::step]
            theta = theta_1[::step]
            # -------------------------------------


            N = len(theta)
            max_time = N / fps  # seconds
            frame_name = filename_1[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
            print(frame_name)

            x = center[:, 0]  # numpy array
            y = center[:, 1]

            # delatX单位是pixel和角度，dX单位是mm和rad
            deltax = center[1:, 0] - center[:-1, 0]  # numpy array
            deltay = center[1:, 1] - center[:-1, 1]
            THETA = theta.reshape(len(center))  # np.zeros(shape=(len(center), 3))
            deltar = np.sqrt(deltax ** 2 + deltay ** 2)
            deltatheta = THETA[1:] - THETA[:-1]

            index = []
            for k in range(len(deltatheta)):
                if deltatheta[k] > 160:  # 处理周期行导致的大deltatheta
                    deltatheta[k] -= 180
                elif deltatheta[k] < -160:
                    deltatheta[k] += 180

            index = []
            for k in range(len(deltatheta)):
                if abs(deltatheta[k]) > 20 or abs(deltax[k]) > 3 or abs(deltay[k]) > 3:  # 把明显由于识别错误产生的零星数据删掉
                    index.append(k)

            deltax = np.delete(deltax, index)
            deltay = np.delete(deltay, index)
            deltar = np.delete(deltar, index)
            deltatheta = np.delete(deltatheta, index)

            dtheta = deltatheta #/ 180 * math.pi
            dx = deltax / 480 * 260
            dy = deltay / 480 * 260
            dr = deltar / 480 * 260

            time = np.around(np.arange(0, len(dtheta) * 1 / 150, 1 / 150), decimals=2)
            deltatheta = dtheta /180.0*math.pi  #弧度
            deltar = (dx+dy)/2#/480*260

            delta_dict[filename[p]+'a_r'] = deltar
            delta_dict[filename[p]+'a_theta'] = deltatheta
    else:
        delta_dict[filename[p]+'a_r'] = None
        delta_dict[filename[p]+'a_theta'] = None


    if  filename[p] in filename_ina:
        path_3 = path2_in + filename[p] + '\\'
        filename_1 =[os.path.splitext(name)[0] for name in os.listdir(path_3) if name.endswith('.h5')]  # 每个fre下文件名
        file_n1 = [path_3 + name + '.h5' for name in filename_1]  # 每个fre下pdf_delta的存储文件
        print(filename_1, file_n1)

        for i in range(len(file_n1)):
            store = pd.HDFStore(file_n1[i], mode='r')
            print(store.keys())
            center_1 = store.get('center').values  # numpy array
            theta_1 = store.get('theta').values
            store.close()

            # todo : 改变delta间隔------------------
            # 这里改了要在存储路径改一下！
            step = 1
            center = center_1[::step]
            theta = theta_1[::step]
            # -------------------------------------


            N = len(theta)
            max_time = N / fps  # seconds
            frame_name = filename_1[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
            print(frame_name)

            x = center[:, 0]  # numpy array
            y = center[:, 1]

            # delatX单位是pixel和角度，dX单位是mm和rad
            deltax = center[1:, 0] - center[:-1, 0]  # numpy array
            deltay = center[1:, 1] - center[:-1, 1]
            THETA = theta.reshape(len(center))  # np.zeros(shape=(len(center), 3))
            deltar = np.sqrt(deltax ** 2 + deltay ** 2)
            deltatheta = THETA[1:] - THETA[:-1]

            index = []
            for k in range(len(deltatheta)):
                if deltatheta[k] > 160:  # 处理周期行导致的大deltatheta
                    deltatheta[k] -= 180
                elif deltatheta[k] < -160:
                    deltatheta[k] += 180

            index = []
            for k in range(len(deltatheta)):
                if abs(deltatheta[k]) > 20 or abs(deltax[k]) > 3 or abs(deltay[k]) > 3:  # 把明显由于识别错误产生的零星数据删掉
                    index.append(k)

            deltax = np.delete(deltax, index)
            deltay = np.delete(deltay, index)
            deltar = np.delete(deltar, index)
            deltatheta = np.delete(deltatheta, index)

            dtheta = deltatheta #/ 180 * math.pi
            dx = deltax / 480 * 260
            dy = deltay / 480 * 260
            dr = deltar / 480 * 260

            time = np.around(np.arange(0, len(dtheta) * 1 / 150, 1 / 150), decimals=2)
            deltatheta = dtheta /180.0*math.pi  #弧度
            deltar = (dx+dy)/2#/480*260

            delta_dict[filename[p]+'ina_r'] = deltar
            delta_dict[filename[p]+'ina_theta'] = deltatheta
    else:
        delta_dict[filename[p]+'ina_r'] = None
        delta_dict[filename[p]+'ina_theta'] = None


a_prop = [] # 计算active的比例
for p in range(len(filename)):
    if delta_dict[filename[p]+'a_theta'] is None:
        a_prop.append(0)
    else:
        a_prop.append(len(delta_dict[filename[p]+'a_theta'])/(len(delta_dict[filename[p]+'a_theta'])+len(delta_dict[filename[p]+'ina_theta'])))

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
f = [i for i in filename]
ax.scatter(f,a_prop,marker = 'o',s=25)
plt.plot(f,a_prop,'--')

ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1,decimals=0))
ax.set_xlabel('$a$ ($g$)')  #('$f$ (Hz)')  # 
ax.set_ylabel('Proportion of active mode')
ax.set_ylim(0,1)
ax.set_yscale('log')
# ax.xaxis.set_minor_locator(MultipleLocator(5))
# ax.xaxis.set_major_locator(MultipleLocator(10))
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
# ax.tick_params(which='minor', direction='in')
plt.show()




