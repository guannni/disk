# 用statsmodels 计算correlation代码
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.cm as cm
import os.path
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.ticker import FormatStrFormatter
from sklearn import preprocessing

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 1
FRE = 60  # todo eachF [50,85,5]没有80，手动排着输
# ACC = 5 # todo eachA [3,5,0.5]

# -----------------------------------------------

def compute_msd(trajectory, t_step, coords=['x', 'y']):
    tau = trajectory['t'].copy()
    shifts = np.floor(tau / t_step).astype(np.int)
    msds = np.zeros(shifts.size)
    msds_std = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = trajectory[coords] - trajectory[coords].shift(-shift)
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std()

    print(msds[0])
    msds = pd.DataFrame({'msds': msds, 'tau': tau, 'msds_std': msds_std})
    return msds


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


# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\data\\'  # TODO: eachF
# path2 = 'D:\\guan2019\\1_disk\\a\\'+str(ACC)+'\\data\\'  # TODO: eachA

path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # TLC
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\data\\'  # 60Hz
# path2 = 'D:\\guan2019\\1_disk\\a\\5\\data\\'  # 5g

filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)



fig = plt.figure()
ax = fig.add_subplot(111)
label = []
for i in range(len(file_n)): #2,3):#
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    center_1 = store.get('center').values  # numpy array
    theta_1 = store.get('theta').values
    store.close()

    # todo : 改变delta间隔------------------
    # 这里改了要在最后图片标题maxtime改一下！
    center = center_1[::step]
    theta = theta_1[::step]
    # -------------------------------------
    N = len(theta)
    max_time = N / fps  # seconds
    frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
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

    dtheta = deltatheta / 180 * math.pi
    dx = deltax / 480 * 260
    dy = deltay / 480 * 260
    dr = deltar / 480 * 260

    time = np.around(np.arange(0, len(dtheta) * 1 / 150, 1 / 150), decimals=2)


    if len(dx) > 10000:
        dx = dx[:10000]
        dr = dr[:10000]
        dtheta = dtheta[:10000]

    # v_t = dx # /(step/fps)  # 平动delta r
    v_t = dtheta #/(step/fps)  # 转动delta theta

    label.append(frame_name + 'Hz')

    plt.figure()
    plot_acf(v_t, alpha=0.05,lags=time,marker='+',markersize=0.5, use_vlines=False, zero=True)
    plt.show()

    time = np.around(np.arange(0, len(v_t) * step / fps, step / fps), decimals=3)
    # plot_pacf(v_t, ax=ax,alpha=0.05,lags=200,marker='+',markersize=8, title='Partial Auto-correlation Function of $\Delta \Theta$ [0.6mm]', use_vlines=False, zero=True)
    plot_pacf(v_t, ax=ax,lags=100,alpha=0.05,marker='+',linestyle='-',markersize=8, title='Auto-correlation Function of $\Delta X$ [0.6mm]', use_vlines=False, zero=True)
    # plot_acf(v_t, ax=ax,lags=100,alpha=0.05,marker='+',markersize=8, title='Auto-correlation Function of $\Delta R$ [0.6mm]', use_vlines=False, zero=True)
#  lags=time,
#  linestyle='-',
#  Partial

ax.set_xlabel('lags')#('time(s)') #

ax.set_ylabel('$C_{\Delta \Theta}(t)$')
# ax.set_ylabel('$C_{\Delta X}(t)$')
# ax.set_ylabel('$C_{\Delta R}(t)$')

# plt.xticks(np.arange(len(v_t)), time)

# ax.set_xlim((0.001, 0.15/(step/fps)))

# ax.set_xscale('log')
# ax.set_yscale('log')

ax.text(.18,.05, 'time step = 1/150s', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
plt.legend(label)
plt.show()

