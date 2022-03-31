# 读取hdf文件，计算v
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import scipy
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.ticker as ticker
from scipy.fftpack import fft,ifft
from matplotlib.ticker import MultipleLocator

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 1
FRE = 60  # todo eachF [50,85,5]没有80，手动排着输
# ACC = 5 # todo eachA [3,5,0.5]

# -----------------------------------------------
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


def smooth(x, window_len=35, window='flat'): #35 5
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


# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\data\\'  # TODO: eachF
# path2 = 'D:\\guan2019\\1_disk\\a\\'+str(ACC)+'\\data\\'  # TODO: eachA

# path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # TLC
path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\data1\\'  # 60Hz TOTAL
# path2 = 'D:\\guan2019\\1_disk\\a\\5\\data\\'  # 5g TOTAL
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\powerspectrum_inactive_select0\\'  # 60Hz MODE
# path2 = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\powerspectrum_active_select0\\'  # 5g MODE


filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)



fig = plt.figure() #figsize=(4,4))
ax = fig.add_subplot(111)
label = []
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
marker = ['o', 'v', 'D','^','h', 's', 'p', '*',  '+', 'x']
for i in range(len(file_n)):#3,4):#
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

    max_time = N / fps * step  # seconds
    frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    if len(frame_name)>3:
        frame_name = frame_name[1:]


    x = center[:, 0] # numpy array
    dx = (x[step::step] - x[:-step:step])/(step/fps)
    y = center[:, 1]
    dy = (y[step::step] - y[:-step:step])/(step/fps)
    N = len(dy)

    traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'v_x': dx, 'v_y': dy})
    print(type(frame_name))

    x = center[:, 0]  # numpy array
    dx = x[step::1] - x[:-step:1]
    y = center[:, 1]
    dy = y[step::1] - y[:-step:1]
    dr = np.sqrt(dx ** 2 + dy ** 2)

    THETA = theta.reshape(len(center))
    dtheta = THETA[step::1] - THETA[:-step:1]
    index = []
    for k in range(len(dtheta)):
        if dtheta[k] > 130:  # 处理周期行导致的大deltatheta
            dtheta[k] -= 180
        elif dtheta[k] < -130:
            dtheta[k] += 180
        if abs(dtheta[k]) > 60:  # 把明显由于识别错误产生的零星数据删掉
            index.append(k)
    dtheta = np.delete(dtheta, index)
    dr = np.delete(dr, index)

    v_t = dr / 480 * 0.26  # 平动delta r
    v_r = dtheta * math.pi / 180 # 转动delta theta

    time = np.arange(0, len(v_r) * step / fps, step / fps)

    # ps = np.abs(np.fft.fft(v_t)) ** 2
    # time_step = step/fps
    # freqs = np.fft.fftfreq(v_t.size, time_step)
    # idx = np.argsort(freqs)
    # plt.plot(freqs[idx], ps[idx])

    # rate = 1/(step/fps)
    # p = np.abs(np.fft.rfft(v_t))**2/(2*math.pi*len(v_t) * step / fps)
    # N = len(p)
    # p = p[0:round(N/2)]
    # fr = np.linspace(0,rate/2,N/2)
    # plt.plot(fr,abs(p)**2)

    f, Pxx_den = scipy.signal.periodogram(v_t, fps/step)  #todo 默认为功率谱密度，若要功率谱，加参数scaling='spectrum'


    print(Pxx_den.size)
    plt.plot(f, smooth(Pxx_den)[:len(f)],alpha=0.65,color=colors[i])
    # plt.scatter(f, smooth(Pxx_den)[:len(f)],marker=marker[i],alpha=0.75, c='', s=25, edgecolor=colors[i])
    # plt.plot(f, Pxx_den,alpha=0.65,color=colors[i])
# 
    # plt.show()

    label.append(frame_name + ' g')


    time = np.arange(0, len(v_r) * step / fps, step / fps)

# plt.xticks(np.arange(len(v_t)), time)
# ax.set_xlim((0.001, 1000))
# ax.set_ylim((1e-15, 1))

ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.set_xscale('log')
# ax.xaxis.set_major_locator(ticker.LogLocator(10))

ax.set_xlabel('$f$ (Hz)')
ax.set_yscale('log')

# ax.set_ylabel('$S_{\Delta \Theta}$ (rad$^2$/Hz)')
ax.set_ylabel('$S_{\Delta r}$ (m$^2$/Hz)')
ax.set_xlim(1/150.0, 80)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')
# ax.xaxis.set_label_coords(0.5, -0.08)
# ax.yaxis.set_label_coords(-0.14, 0.5)

# plt.title('Translational Power Spectrum Density [0.6mm]')#, step=%s]' % str(step))
# plt.title('Rotational Power Spectrum Density [0.6mm]')#, step=%s]' % str(step))

# label = ['1','2','3']
# plt.legend(label)
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)


# 60Hz translational
# plt.plot(np.logspace(0.9,1.9,5), np.logspace(-8.8,-7.8, 5), 'r--')   # k=1
# plt.annotate(r'$k=1$', xy=(7,10e-9), xytext=(7,10e-9), xycoords='data',color='r')

# # 60Hz rotational 5g rotational
# plt.plot(np.logspace(0.3,0.95,5), np.logspace(-2.8,-4.1, 5), 'r--')   # k=-2
# plt.annotate(r'$k=-2$', xy=(1,8.6e-5), xytext=(10,15e-5), xycoords='data',color='r')

# # TL rotational
# plt.plot(np.logspace(0.3,0.95,5), np.logspace(-2.3,-3.6, 5), 'brown--')   # k=-2
# plt.annotate(r'$k=-2$', xy=(1,8.6e-5), xytext=(10,50e-5), xycoords='data',color='r')

# # TL translational  5g Translational
# plt.plot(np.logspace(1.2,2,5), np.logspace(-9.3,-8.5, 5), 'b--')   # k=-2
# # plt.plot(np.logspace(1.3,1.8,5), np.logspace(-11,-10, 5), 'v--')   # k=-2
# # # plt.annotate(r'$k=1$', xy=(9,4e-9), xytext=(9,4e-9), xycoords='data',color='r')

# # # TL rotational
# plt.plot(np.logspace(0.8,1.3,5), np.logspace(-3,-4, 5), 'r--')   # k=-2
# plt.annotate(r'$k=-2$', xy=(10,0.0005), xytext=(10,0.0005), xycoords='data',color='r')

# plt.axvline(x=10, c="r", alpha=0.5, lw=0.6, ls="--")
# plt.annotate(r'$f=10$', xy=(14,4.2e-10), xytext=(9.5,5.3e-9), xycoords='data',color='r')
# # plt.axvline(x=20, c="r", alpha=0.5, lw=0.6, ls="--")
# # plt.axvline(x=30, c="r", alpha=0.5, lw=0.6, ls="--")
# # plt.axvline(x=40, c="r", alpha=0.5, lw=0.6, ls="--")
# # # plt.axvline(x=50, c="r", alpha=0.5, lw=0.6, ls="--")
# # plt.axvline(x=60, c="r", alpha=0.5, lw=0.6, ls="--")
# plt.axvline(x=70, c="r", alpha=0.5, lw=0.6, ls="--")
# plt.annotate(r'$f=70$', xy=(14,4.2e-10), xytext=(63,5.3e-9), xycoords='data',color='r')
plt.show()

