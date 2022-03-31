# 读取hdf文件，计算v
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.ticker as ticker
from scipy import signal
from sklearn import preprocessing
import statsmodels.tsa.stattools as stattools
from matplotlib.ticker  import MultipleLocator

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


def smooth(x, window_len=51, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


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

# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\data\\'  # TODO: eachF
# path2 = 'D:\\guan2019\\1_disk\\a\\'+str(ACC)+'\\data\\'  # TODO: eachA

# path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # TLc
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\data\\'  # 60Hz
path2 = 'D:\\guan2019\\1_disk\\a\\5\\data\\'  # 5g


filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
marker = ['o', 'v', 'D', '^', 's', 'h', '2', 'p', '*', '+', 'x']
label = []
corr_valid = []
for i in range(len(file_n)):  # 3,4):#
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    center_1 = store.get('center').values  # numpy array
    theta_1 = store.get('theta').values
    store.close()

    # todo : 改变delta间隔------------------
    # 这里改了要在最后图片标题maxtime改一下！
    center = center_1
    theta = theta_1
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

    v_t = dx  # /(step/fps)  # 平动delta r
    v_r = dtheta  # /(step/fps)  # 转动delta theta

    if len(frame_name) > 3:
        frame_name = frame_name[1:]
    # label.append(frame_name + 'Hz')

    time = np.arange(0, len(v_t) * step / fps, step / fps)
    # corr = np.sum((v_t*v_r))/(np.sqrt((np.sum(v_t**2))*(np.sum(v_r**2))))
    # print(corr)

    # corr = np.correlate(v_t, v_r, mode='same')
    # corr = preprocessing.StandardScaler().fit_transform(corr.astype('float32').reshape(-1, 1))

    # --
    v_t = (v_t - np.mean(v_t)) / (np.std(v_t) * len(v_t))
    v_r = (v_r - np.mean(v_r)) / (np.std(v_r))
    corr = np.correlate(v_t, v_r, 'full')

    a = np.logspace(-3,2,50)
    a_index = []
    time_new = []
    corr_new = []
    for j in a:
        ind = find_nearest(time, j)
        a_index.append(ind)
        time_new.append(time[ind])
        corr_new.append(corr[ind])

    plt.scatter(time_new, corr_new[:len(time)], c='',marker=marker[i], s = 25, edgecolor=colors[i], label=frame_name + ' Hz')
    plt.plot(time_new, corr_new[:len(time)], alpha=0.65, color=colors[i])

    # plt.scatter(time, corr[:len(time)], c='', s=1, alpha=0.8, edgecolor=colors[i],label=frame_name + ' Hz')  # auto correlation
    corr_valid.append(np.correlate(v_t, v_r, 'valid'))
#
# # fourier transform of cross-correlation
#     ps = np.abs(np.fft.fft(corr[:len(time)])) ** 2
#     time_step = step/fps
#     freqs = np.fft.fftfreq(corr[:len(time)].size, time_step)
#     idx = np.argsort(freqs)
#     plt.plot(freqs[idx], ps[idx])
#     plt.show()


# font = {'family': 'Calibri', 'weight': 'bold', 'style': 'italic', 'size': 16 }
# font1 = {'family': 'Calibri', 'weight': 'bold', 'style': 'normal', 'size': 16 }
# ax.set_xlabel('$t (s)$', font1)
# ax.set_ylabel('$C_{\Delta x \Delta \Theta }(t)$', font1)  # 'I')#
# plt.title('Cross-correlation [60Hz]',font1)
# plt.tick_params(labelsize=15)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Calibri') for label in labels]

ax.set_xlim((0.003, 45))
ax.set_ylim((-0.1, 0.1))
leg = ax.legend(loc='upper left')
leg.get_frame().set_linewidth(0.0)
plt.legend()
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('$C_{\Delta x \Delta \Theta }(t)$')  # 'I')#
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')

ax.xaxis.set_label_coords(0.5, -0.07)
ax.yaxis.set_label_coords(-0.13, 0.5)

# ax.set_yscale('log')
ax.set_xscale('log')
# ax.xaxis.set_major_locator(ticker.LogLocator(10))
# label = ['mode1','mode2','mode3','mode4']
# label = ['1','2','3']
plt.legend(label)
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)

# for i in range(len(corr_valid)):
#     plt.axhline(y=corr_valid[i], c="b", alpha=0.5, lw=0.6, ls="--", label=str(corr_valid[i]))
#     plt.annotate("y=%.2f, %s" % (corr_valid[i],label[i]),  xy=(3**i-2**i+0.1, corr_valid[i]), textcoords='offset pixels',fontsize=8)


plt.show()
