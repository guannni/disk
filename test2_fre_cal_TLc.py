# 读取hdf文件，计算pdf
# 出D:\guan2019\1_disk\f\下 按fre分类的数据的pdf图像


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
import seaborn as sns
from lmfit.models import GaussianModel, LorentzianModel, ExponentialModel

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

def smooth(x, window_len=25, window='flat'):
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

# 把pdf的delta数据存出去
path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # \tianli_f_all\
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]  # fre 文件夹
file_n = [path2 + name +'.h5' for name in filename]  # 每个fre下原始data的路径
# file_delta = 'D:\\guan2019\\1_disk\\TLc\\all\\analysis_delta2\\pdf\\'  # \all\ 每个fre下pdf——delta的存储路径
file_delta = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis\\pdf\\'  # \tianli_f_all\每个fre下pdf——delta的存储路径
print(filename, file_n, file_delta)

for p in range(len(file_n)):#2,3):# # TODO:60Hz——range(2,len(file_n))
    path_1 = file_n[p]
    filename_1 = filename  # 每个fre下文件名
    file_n1 = file_n  # 每个fre下pdf_delta的存储文件
    file_s1 = [file_delta + name + '.txt' for name in filename_1]  # 每个fre下pdf_delta的存储文件
    file_s2 = [file_delta + name + '.jpg' for name in filename]  # 每个fre下pdf_delta的存储文件

    print(filename_1, file_n1)
    counts = np.zeros((len(file_n1)))

    pdf_trans_dict = {}
    pdf_rot_dict = {}
    delta_dict = {}
    label = []
    for i in range(len(file_n1)):#2,3):#
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

        time = np.around(np.arange(0, len(dtheta) * 1 / 150, 1 / 150), decimals=2)


        deltatheta = dtheta
        deltar = dx

        weights_r = np.ones_like(deltar) / float(len(deltar))
        au, bu, cu = plt.hist(deltar, 31, histtype='bar', facecolor='yellowgreen',
                              weights=weights_r, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
        pdf_trans_dict[filename_1[i]+'y'] = au
        bu = (bu[:-1]+bu[1:])/2.
        pdf_trans_dict[filename_1[i]+'x'] = bu  # 存入dict

        weights_theta = np.ones_like(deltatheta) / float(len(deltatheta))
        AU, BU, CU = plt.hist(deltatheta, 31, histtype='bar',
                              facecolor='blue', weights=weights_theta, alpha=0.75, rwidth=0.2, density=True)
        pdf_rot_dict[filename_1[i] + 'y'] = AU
        BU = (BU[:-1]+BU[1:])/2.
        pdf_rot_dict[filename_1[i] + 'x'] = BU  # 存入dict

        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta

        label.append(frame_name + 'Hz')


    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}

    ys = [i + x + (i * x) ** 2 for i in range(len(filename_1))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'r'],norm_hist=True, bins=90, kde=True, hist=False, label=filename_1[i].split('_', 1)[0] + 'Hz')  #'mode'+str(i+1))#
        #
        plt.scatter(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], alpha=0.75,color=colors[i], label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.plot(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5)


    ax1.set_title('Translational PDF' + ' [0.6mm] ', fontsize=10)
    ax1.set_xlabel('$\Delta x(mm)$')
    ax1.set_ylabel('P')
    plt.yscale('log')
    ax1.set_ylim(0.0001, 100)
    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)

    ax2 = fig.add_subplot(122)

    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'theta'], norm_hist=True,bins=200, kde=True, hist=False, label=filename_1[i].split('_', 1)[0] + 'Hz')  #'mode'+str(i+1))#
        plt.scatter(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'], alpha=0.75, color=colors[i], cmap='hsv',label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.plot(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5)


    ax2.set_title('Rotational PDF', fontsize=10)
    ax2.set_xlabel('$\Delta theta (rad)$')
    ax2.set_ylabel('P')
    plt.yscale('log')
    ax2.set_ylim(0.0001, 100)
    # ---------------------------------------------------------------------------

    plt.legend(label)

    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)

    plt.show()
    # fig.savefig(file_s2[p])



