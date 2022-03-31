# 读取hdf文件(msd)
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy.ndimage import filters
import matplotlib.cm as cm

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# def smooth(x, window_len=25, window='flat'):
#     """smooth the data using a window with requested size.
#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.
#     input:
#         x: the input signal
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.
#     output:
#         the smoothed signal
#     example:
#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)
#     """
#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")
#     if x.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")
#     if window_len < 3:
#         return x
#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
#     s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
#     # print(len(s))
#     if window == 'flat':  # moving average
#         w = np.ones(window_len, 'd')
#     else:
#         w = eval('np.' + window + '(window_len)')
#     y = np.convolve(w / w.sum(), s, mode='valid')
#     return y

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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis\\msd_rot\\msd_rot1\\'
path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis\\msd\\msd1\\' ### msd1\\'

# path3 = 'C:\\Users\\guan\\Desktop\\2\\1_a\\msd_rot\\delta0\\5g_new\\'  # 5g不同频率
# path3 = 'C:\\Users\\guan\\Desktop\\2\\1_f\\msd_rot\\delta0\\60Hz\\'  # 60Hz不同加速度
filename = [os.path.splitext(name)[0] for name in os.listdir(path3) if name.endswith('.h5')]  # 只取.h5文件
file_n = [path3 + str(name) + '.h5' for name in filename]
# print(filename, file_n)

fig = plt.figure()
ax = fig.add_subplot(111)
label = []
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for i in range(len(file_n)):
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    MSD_key = store.keys()[0]
    MSD = store.get(MSD_key).values[1:, 0] # filters.gaussian_filter1d(store.get(MSD_key).values[:, 0], 3)  #
    TAU = store.get(MSD_key).values[1:, 1]
    print(MSD[0:5],TAU[0:5])
# ----------------------------------------
#     TAU_new = TAU  # []
#     MSD_new = MSD  # []

    a = np.logspace(-2, 3, 50)
    a_index = []
    TAU_new = []
    MSD_new = []
    for j in a:
        ind = find_nearest(TAU, j)
        a_index.append(ind)
        TAU_new.append(TAU[ind])
        MSD_new.append(MSD[ind])

    MSD_new = np.array(MSD_new)
    TAU_new = np.array(TAU_new)
    TAU_new[np.isinf(MSD_new)] = 0
    MSD_new[np.isinf(MSD_new)] = 0

    tt = 0
    ind_r = []
    for T in TAU_new[1:]:
        if T == TAU_new[tt]:
            ind_r.append(tt+1)
        tt += 1

    TAU_new = np.delete(TAU_new, ind_r)
    MSD_new = np.delete(MSD_new, ind_r)

    step = 1
    slope = (np.log(MSD_new[1:])-np.log(MSD_new[:-1]))/(np.log(TAU_new[1:])-np.log(TAU_new[:-1]))#np.power(10, np.gradient(np.log(MSD)))

    label.append(str(MSD_key[1:]) + 'Hz')


    # plt.plot(TAU_new[:-1], smooth(slope, 4)[:len(TAU) - 1], 'o-', markerfacecolor='none', markersize=5, color=colors[i],alpha=0.6)
    # plt.plot(TAU_new[:-1], slope, 'o-', markerfacecolor='none', markersize=5, color=colors[i])

    slope1 = smooth(slope, 4)
    plt.plot(TAU_new[:-1], slope1, 'o-', markerfacecolor='none',markersize=5, color=colors[i])
    store.close()

    plt.ylim(-0.5,2.5)
    # plt.xlim(0.001,60)
    plt.xscale('log')

# label = ['mode1','mode2','mode3','mode4']

plt.axhline(y=1, c="r", ls="--", lw=0.3, alpha=0.8)
plt.axhline(y=2, c="r", ls="--", lw=0.3, alpha=0.8)
# plt.axhline(y=1.5, c="r", ls="--", lw=0.3, alpha=0.8)
# plt.annotate(r'$1.7$',c='r', xy=(2, 1), xytext=(0.01,1.7), xycoords='data')

# plt.axvline(x=5.5, c="r", ls="--", lw=0.3, alpha=0.8)  # alpha < 1.01
ax.set_title('Local Slope of Rotational MSD [0.6mm]', fontsize=10)
# ax.set_title('Local Slope of Translational MSD [0.6mm]', fontsize=10)
ax.set_xlabel('$t\ (s)$')
ax.set_ylabel('$\lambda $')
plt.legend(label)
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)
plt.show()

print(label)