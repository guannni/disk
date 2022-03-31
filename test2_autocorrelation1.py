# 自己写的计算autocorrelation 的代码 + fft获得特征频率
# singlefile
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
from matplotlib.ticker import FormatStrFormatter
from sklearn import preprocessing
import scipy.fftpack
from matplotlib.ticker  import MultipleLocator

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 1
FRE = 60  # todo eachF [50,85,5]没有80，手动排着输
# ACC = 5 # todo eachA [3,5,0.5]

# -----------------------------------------------
def smooth(x, window_len=5, window='flat'):
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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def autocorr(array):  # array is deltax
    # array *= 10000
    ac = np.zeros(len(array))
    for i in range(len(array)):  # lags
        temp = []
        for j in range(len(array)):
            if j + i < len(array):
                temp.append(array[j]*array[j+i])
        # print(temp)
        temp /= np.var(array)  # np.square(array[:len(temp)])
        nan = np.isnan(np.array(temp))
        temp = np.delete(temp,nan)
        ac[i] = temp.sum()/len(temp)
    return ac

# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\data\\'  # TODO: eachF
# path2 = 'D:\\guan2019\\1_disk\\a\\'+str(ACC)+'\\data\\'  # TODO: eachA

# path2 = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\msd_trans_inactive_all0\\' # 5g mode
# path2 = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\msd_rot_inactive_all0\\' # 5g mode
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\msd_trans_inactive\\' # 60Hz mode
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\msd_rot_inactive\\' # 60Hz mode


# path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\C_w\\'  # TLC delta theta
# path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\C_v\\'  # TLC delta x
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\C_W\\'  # 60Hz delta theta
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\C_V\\'  # 60Hz delta x
path2 = 'D:\\guan2019\\1_disk\\a\\5\\data1\\'  # 5g delta theta delta x
#
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)



fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
label = []
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
marker = ['o', 'v', 'D','^','s', 'p','h', '2',  '*',  '+', 'x']
for i in range(len(file_n)): #0,1):#
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
    if len(frame_name) > 3:
        frame_name = frame_name[1:]

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




    if len(dx) > 5000: #10000:#
        dx = dx[:5000]
        #
        dy = dy[:3000]
        # dy1 = dy[:900*3:3]+dy[1:900*3+1:3]+dy[2:900*3+2:3]
        # dy = dy1

        dr = dr[:5000]
        dtheta = dtheta[:5000]

    # v_t = dr #dy#(dx+dy)/2 # /(step/fps)  # 平动delta r
    v_t = dtheta #/(step/fps)  # 转动delta theta

    label.append(frame_name + ' Hz')

# method 1 --my function-------
    ac = autocorr(v_t)
    nan = np.isnan(ac)
    ac = ac[~nan]

    # # rotational
    # ac = ac[2:]
    # # ac /= ac[0]
    # ac = np.insert(ac,0,1.0)


    # # ac = smooth(ac)
    # # ac = ac[5:]
    # # ac /= ac[0]



# # log-x axis
#     time = np.around(np.arange(0, (len(ac)+1) * 1 / 150, 1 / 150), decimals=3)[0:len(ac)]
#     a = np.logspace(-3,2,50)
#     a_index = []
#     time_new = []
#     ac_new = []
#     for j in a:
#         ind = find_nearest(time, j)
#         a_index.append(ind)
#         time_new.append(time[ind])
#         ac_new.append(ac[ind])

#     # plt.scatter(time_new[0:len(ac_new)], ac_new/ac_new[0],c='',alpha=0.65,edgecolor=colors[i])  # auto correlation
#     plt.plot(time_new[0:len(ac_new)], ac_new/ac_new[1], marker=marker[i],markerfacecolor="None", markersize=6, color=colors[i], label=label[i])# color=colors[i])  # auto correlation


 # linear
    time = np.around(np.arange(0, (len(ac)+1) *1 / 150, 1 / 150), decimals=3)[0:len(ac)]
    plt.plot(time[0:len(ac)], ac/ac[0],marker=marker[i],markerfacecolor="None", markersize=5, color=colors[i], label=label[i])# color=colors[i])  # auto correlation
     # plt.scatter(time[0:len(ac)], ac/ac[0],alpha=0.65,marker=marker[i],c='', s=25, edgecolor=colors[i], label=label[i])  # auto correlation


    # # # -------------提取频率
    # ac = autocorr(v_t)
    # nan = np.isnan(ac)
    # ac = ac[~nan]
    # ac_f = scipy.fftpack.fft(ac)
    # freqs = np.linspace(0.0, 150.0/2.0, int(len(ac) / 2.0))
    # # plt.figure()
    # # fig, ax = plt.subplots()
    # ax.plot(freqs, 2.0 / len(ac) * np.abs(ac_f[:len(ac) // 2]),alpha=0.65,color=colors[i])
    # # plt.show()

# method 2 --other's function-----------
    # time = np.around(np.arange(0, len(v_t) * step / fps, step / fps), decimals=3)
    # plot_acf(v_t, ax=ax,lags=200,alpha=0.65,marker='o',markersize=8,color=colors[i],  title='Auto-correlation Function of $\Delta \Theta$ [0.6mm]', use_vlines=False, zero=True)
    # plot_pacf(v_t, ax=ax,lags=100,alpha=0.05,marker='+',linestyle='-',markersize=8, title='Auto-correlation Function of $\Delta X$ [0.6mm]', use_vlines=False, zero=True)
    # plot_acf(v_t, ax=ax,lags=100,alpha=0.65,marker='o',markersize=8,color=colors[i], title='Auto-correlation Function of $\Delta R$ [0.6mm]', use_vlines=False, zero=True)
#  lags=time,
#  linestyle='-',
#  Partial
#-----------------------------------------------------------------------
#
#----------auto correlation---------------------
ax.set_xlabel('$t$ (s)') #('lags')#


#
# #----------Fourier------
# ax.set_xlabel('f(Hz)') #('lags')#
# # ax.set_ylabel('$fft$')



time = np.around(np.arange(0, (len(v_t) ) * 1 / 150, 1 / 150), decimals=3)
# plt.xticks(np.arange(len(v_t)), time)

# # rotational
ax.set_ylabel('$C_{\Delta \Theta}(t)$')
# ax.set_xlim((0.005, 17))
# ax.set_ylim((-0.5, 1.2))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.set_xscale('log')

# # translational
# ax.set_ylabel('$C_{\Delta r}(t)$')
# ax.set_xlim((-0.01, 0.065)) # tlc
# ax.xaxis.set_major_locator(MultipleLocator(0.02))
# ax.xaxis.set_minor_locator(MultipleLocator(0.01))
# ax.set_xlim((-0.01, 0.5)) # 60Hz
# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.xaxis.set_minor_locator(MultipleLocator(0.05))
# ax.set_xlim((-0.01, 0.25)) # 5g
# ax.xaxis.set_major_locator(MultipleLocator(0.05))
# ax.xaxis.set_minor_locator(MultipleLocator(0.01))
# ax.set_ylim((-0.6,1.15))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))


# # # # ffs
# ax.set_xlim((0,75))
# ax.xaxis.set_major_locator(MultipleLocator(15))
# ax.xaxis.set_minor_locator(MultipleLocator(5))
# ax.set_xlabel('$f$ (Hz)') #('lags')#


leg = ax.legend(label) #,loc='upper right') # right
leg.get_frame().set_linewidth(0.0)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')
# ax.xaxis.set_label_coords(0.5, -0.07)
# ax.yaxis.set_label_coords(-0.09, 0.5)




plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# ax.text(.18,.05, 'time step = 1/150s', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

plt.show()

