# 读取hdf文件(msd)
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
from scipy.ndimage import filters
import matplotlib.cm as cm

def traversalDir_FirstDir(path):  # 返回一级子文件夹名字
    list = []
    if (os.path.exists(path)):  # 获取该目录下的所有文件或文件夹目录路径
        files = glob.glob(path + '\\*')
        # print(files)
        for file in files:  # 判断该路径下是否是文件夹
            if (os.path.isdir(file)):  # 分成路径和文件的二元元组
                h = os.path.split(file)
                print(h[1])
                list.append(h[1])
        return list

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(np.logspace(-2, 2, 5))
    y_vals = intercept + slope * x_vals
    plt.plot(np.logspace(-2, 2, 5), np.logspace(-2, 2, 5), '--')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis\\msd\\msd1\\' ###  # TLC
# path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis\\msd_rot\\msd_rot1\\'  # TLC
#
# path3 = 'C:\\Users\\guan\\Desktop\\2\\1_a\\msd\\delta0\\5\\'  # 5g不同频率
# path3 = 'C:\\Users\\guan\\Desktop\\2\\1_a\\msd_rot\\delta0\\5\\'  # 5g不同频率

path3 = 'C:\\Users\\guan\\Desktop\\2\\1_f\\msd_rot\\delta0\\60Hz\\'  # 60Hz不同加速度
# path3 = 'C:\\Users\\guan\\Desktop\\2\\1_f\\msd\\delta0\\60Hz\\'  # 60Hz不同加速度
filename = [os.path.splitext(name)[0] for name in os.listdir(path3) if name.endswith('.h5')]  # 只取.h5文件
file_n = [path3 + str(name) + '.h5' for name in filename]
# print(filename, file_n)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
label = []
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
marker = ['o', 'v', 'D','^','s','h',   'p', '*',  '+', 'x']
for i in range(len(file_n)):  #0,1):#
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    MSD_key = store.keys()[0]
    MSD = store.get(MSD_key).values[1:, 0]*26*26/48/48  # filters.gaussian_filter1d(store.get(MSD_key).values[:, 0], 3)  #
    # MSD = store.get(MSD_key).values[1:, 0]
    TAU = store.get(MSD_key).values[1:, 1]
    print(MSD[0:5],TAU[0:5])

    label.append(str(MSD_key[1:]) + ' Hz')

    a = np.logspace(-3,2,50)
    a_index = []
    TAU_new = []
    MSD_new = []
    for j in a:
        ind = find_nearest(TAU, j)
        a_index.append(ind)
        TAU_new.append(TAU[ind])
        MSD_new.append(MSD[ind])

    # if MSD_key[1:]=='80': # tianli rotational 专用
    #     MSD_new=np.array(MSD_new)
    #     MSD_new +=0.1

    # plt.plot(TAU_new, MSD_new, 'o', markerfacecolor="none",color=colors[i])
    plt.scatter(TAU_new, MSD_new,c='',  edgecolor=colors[i],label=label[i]) #marker=marker[i], c='', s=25, edgecolor=colors[i], label=label[i])

    # plt.plot(TAU, MSD, alpha=0.6,lw=3)
    store.close()


    plt.ylim(0.0001,1000000)
    # plt.xlim(0.01,80)
    plt.xscale('log')
    plt.yscale('log')

#
# ax.set_title('Translational MSD [60 Hz]', fontsize=10)
ax.set_ylabel(r'$\langle\Delta R^2 \rangle$ (mm$^2$)')
# ax.set_title('Rotational MSD [60 Hz]', fontsize=10)
# ax.set_ylabel(r'$\langle\Delta \Theta ^2 \rangle$ (rad$^2$)')

ax.set_xlabel(r'$t$ (s)')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')
# ax.xaxis.set_label_coords(0.5, -0.08)
# ax.yaxis.set_label_coords(-0.0005, 0.1)

# label = ['mode1','mode2','mode3']#,'mode4']
# label = ['1','2','3']#,'mode4']

# plt.legend(label)
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)

# # rotatioanl
# plt.plot(np.logspace(-0.5,0.5, 5), np.logspace(2.5,3.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.23,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.4, 5), np.logspace(-0.5,1.5, 5), 'b--')  # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.1,2), xycoords='data')

# # translational
# plt.plot(np.logspace(0, 1, 5), np.logspace(1.5,3.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(2,1800), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.4, 5), np.logspace(-0.3,0.8, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.1,5), xycoords='data')

# # 55 translational
# plt.plot(np.logspace(0, 1, 5), np.logspace(-1,1, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(4.8,0.5), xycoords='data')
# plt.plot(np.logspace(-1,1, 5), np.logspace(-0.1,1.8, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.1,5), xycoords='data')
# # plt.plot(np.logspace(-1.5,-0.5, 5), np.logspace(-2.3,-1.3, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.1,0.01), xycoords='data')

# # 55 rotational
# plt.plot(np.logspace(-1, 1, 5), np.logspace(-3,1, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(4.8,0.5), xycoords='data')
# plt.plot(np.logspace(0.4,1.6, 5), np.logspace(4,5.2, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(2.3,50000), xycoords='data')
#
# # 70 rotational
# plt.plot(np.logspace(-2,-0.2, 5), np.logspace(-2,1.6, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(0.17, 1), xytext=(0.17,1.15), xycoords='data')
# plt.plot(np.logspace(0.6,1.6, 5), np.logspace(3.7,4.7, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(2.3,15900), xycoords='data')

# # # 70 translational
# plt.plot(np.logspace(-1,0, 5), np.logspace(-0.5,0.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(0.17, 1), xytext=(0.6,1.1), xycoords='data')
# plt.plot(np.logspace(0.5,1.5, 5), np.logspace(1,2.5, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1.5$', xy=(2, 1), xytext=(8,30), xycoords='data')

# #################                   TLC----------
# # # step=1 rotatioanl
# plt.plot(np.logspace(0.1,1.1, 5), np.logspace(2.7,3.7, 5), 'b--')   # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.8,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.6, 5), np.logspace(-0.5,1.5, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.01,2), xycoords='data')

# # # step=1 translational
# plt.plot(np.logspace(0, 1, 5), np.logspace(1.5,3.5, 5), 'b--')   # k=1
# # plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(2,1800), xycoords='data')
# plt.plot(np.logspace(-1.1,-0.1, 5), np.logspace(-0.8,0.2, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.39,0.56), xycoords='data')
################################################################################

# # ####################                 60Hz----------------
# # # # step=1 translational
# # plt.plot(np.logspace(1.2,1.7, 5), np.logspace(1,2, 5), 'b--')   # k=1
# # plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(28,7.5), xycoords='data')
# plt.plot(np.logspace(-1.1,-0.1, 5), np.logspace(0,1, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.09,3.5), xycoords='data')

# # # step=1 rotatioanl
# plt.plot(np.logspace(0.1,1.1, 5), np.logspace(2.2,3.2, 5), 'b--')   # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.8,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.6, 5), np.logspace(-0.9,1.1, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.01,2), xycoords='data')

# # ####################                 5g----------------
# # # step=1 translational
# plt.plot(np.logspace(0,1, 5), np.logspace(-0.8,1.2, 5), 'b--')   # k=1
# # plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(1.8,0.23), xycoords='data')
# plt.plot(np.logspace(-1.1,-0.1, 5), np.logspace(0,1, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.09,3.5), xycoords='data')

# # # step=1 rotatioanl
# plt.plot(np.logspace(0.1,1.1, 5), np.logspace(2.1,3.1, 5), 'b--')   # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(1.5,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.6, 5), np.logspace(-0.8,1.2, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.02,2), xycoords='data')


plt.show()

print(label)