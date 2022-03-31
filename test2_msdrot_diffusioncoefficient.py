# 读取hdf文件(msd)
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy.ndimage import filters

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

# path3 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis_delta2\\msd\\'  # msd .h5文件
# path3 = 'D:\\guan2019\\1_disk\\TLc\\all\\analysis\\msd\\'
path3 = 'C:\\Users\\guan\\Desktop\\2\\1_a\\msd_rot\\delta0\\5g_new\\'  # 5g不同频率
# path3 = 'C:\\Users\\guan\\Desktop\\2\\1_f\\msd_rot\\delta2\\60Hz\\'  # 60Hz不同加速度
filename = [os.path.splitext(name)[0] for name in os.listdir(path3) if name.endswith('.h5')]  # 只取.h5文件
file_n = [path3 + str(name) + '.h5' for name in filename]
# print(filename, file_n)

fig = plt.figure()
ax = fig.add_subplot(111)
label = []
for i in range(len(file_n)):
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    MSD_key = store.keys()[0]
    MSD = store.get(MSD_key).values[:, 0]  # filters.gaussian_filter1d(store.get(MSD_key).values[:, 0], 3)  #
    TAU = store.get(MSD_key).values[:, 1]

# ----------------------------------------
    step = 1
    diff_coef = MSD/4/TAU
    label.append(str(MSD_key[1:]) + 'Hz, 5g')

    plt.plot(TAU, diff_coef*0.54**2, alpha=0.8)
    store.close()

    # plt.ylim(0,2)
    plt.xlim(0.01,60)
    plt.xscale('log')
    plt.yscale('log')

plt.axhline(y=1, c="r", ls="--", lw=0.3, alpha=0.8)
# plt.axvline(x=5.5, c="r", ls="--", lw=0.3, alpha=0.8)  # alpha < 1.01
ax.set_title('Diffusion coefficient', fontsize=10)
ax.set_xlabel('time(s)')
ax.set_ylabel('Dr(mm^2/s)')
plt.legend(label)
plt.show()

print(label)