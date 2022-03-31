# 读取hdf文件，计算pdf
# 出D:\guan2019\1_disk\f\下 按fre分类的数据的pdf图像


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import seaborn as sns
from lmfit.models import GaussianModel, LorentzianModel, DampedOscillatorModel,ExponentialModel,ExponentialGaussianModel,ExpressionModel,SkewedGaussianModel

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
path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # TLC
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


        x = center[:, 0]  # numpy array
        y = center[:, 1]
        deltax = center[1:, 0] - center[:-1, 0]  # numpy array
        deltay = center[1:, 1] - center[:-1, 1]
        THETA = theta.reshape(len(center))#np.zeros(shape=(len(center), 3))

        index = []
        for k in range(len(deltax)):
            if abs(deltax[k]) > 5 or abs(deltay[k]) > 5:  # 把明显由于识别错误产生的零星数据删掉
                index.append(k+1)

        x = np.delete(x, index)
        y = np.delete(y, index)
        THETA = np.delete(THETA,index)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        dr = np.sqrt(dx ** 2 + dy ** 2)
        dtheta = THETA[1:] - THETA[:-1]
        print(np.min(dx), np.max(dx))
        print(len(x), len(dr))

        index = []
        for k in range(len(dtheta)):
            if dtheta[k] > 160:  # 处理周期行导致的大deltatheta
                dtheta[k] -= 180
            elif dtheta[k] < -160:
                dtheta[k] += 180

        deltatheta = dtheta/180*math.pi
        deltar = dx/480*260


        au, bu, cu = plt.hist(deltar, 201, histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        pdf_trans_dict[filename_1[i]+'y'] = au/len(deltar)
        bu = (bu[:-1]+bu[1:])/2.
        pdf_trans_dict[filename_1[i]+'x'] = bu  # 存入dict

        AU, BU, CU = plt.hist(deltatheta, 201, histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)
        pdf_rot_dict[filename_1[i] + 'y'] = AU/len(deltar)
        BU = (BU[:-1]+BU[1:])/2.
        pdf_rot_dict[filename_1[i] + 'x'] = BU  # 存入dict

        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta

    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}
    label = []


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'r'],norm_hist=True, bins=200, kde=True, hist=False, label=filename_1[i].split('_', 1)[0] + 'Hz')  #'mode'+str(i+1))#
        #
        plt.scatter(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], alpha=0.5,marker='o', s=3,label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.plot(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5)

########## gaussian fitting
        x = pdf_trans_dict[filename_1[i] + 'x']
        y = pdf_trans_dict[filename_1[i] + 'y']

        exp_mod = ExponentialModel(prefix='exp_')
        pars = exp_mod.guess(y, x=x)
        gauss1 = GaussianModel(prefix='g1_')
        pars.update(gauss1.make_params())

        pars['g1_center'].set(value=0, min=-0.1, max=0.1)
        pars['g1_sigma'].set(value=0.2, min=0.01)
        pars['g1_amplitude'].set(value=0.01, min=0.0001)

        mod = gauss1

        init = mod.eval(pars, x=x)
        out = mod.fit(y, pars, x=x)

        print(out.fit_report(min_correl=0.5))
        # ax1.plot(x,y, 'b',alpha=0.6,marker='.',markersize=3)
        ax1.plot(x, out.best_fit)
        ax1.legend(loc='best')


    ax1.set_title('Translational PDF' + ' [0.6mm] ', fontsize=10)
    ax1.set_xlabel('$\Delta x(mm)$')
    ax1.set_ylabel('P')
    # ax1.set_xlim(-2.4, 2.4)
    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)

    ax2 = fig.add_subplot(122)

    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'theta'], norm_hist=True,bins=200, kde=True, hist=False, label=filename_1[i].split('_', 1)[0] + 'Hz')  #'mode'+str(i+1))#
        plt.scatter(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'],marker='o', alpha=0.5, s=10,label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.plot(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5)

# ########## 整组数据拟合 2 （左右分开）#########################################################
        middle = np.where(pdf_rot_dict[filename_1[i] + 'x']==find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][0]  #pdf 0 的点
        x1 = pdf_rot_dict[filename_1[i] + 'x'][:middle-5]  # 越过中间的几个点
        y1 = pdf_rot_dict[filename_1[i] + 'y'][:middle-5]
        x2 = -np.flipud(pdf_rot_dict[filename_1[i] + 'x'][middle+5:])
        y2 = np.flipud(pdf_rot_dict[filename_1[i] + 'y'][middle+5:])

        exp_mod1 = ExponentialModel(prefix='exp_')
        pars1 = exp_mod1.guess(y1, x=x1)
        lorentz1 = ExponentialGaussianModel(prefix='l1_')
        pars1.update(lorentz1.make_params())
        pars1['l1_center'].set(value=-0.15, min=-0.5, max=-0.05)
        pars1['l1_sigma'].set(value=0.05, min=0)
        pars1['l1_amplitude'].set(value=0.01, min=0)
        pars1['l1_gamma'].set(value=5, min=1)

        mod1 = lorentz1 #+ lorentz2
        init = mod1.eval(pars1, x=x1)
        out1 = mod1.fit(y1, pars1, x=x1)

        exp_mod2 = ExponentialModel(prefix='exp_')
        pars2 = exp_mod2.guess(y2, x=x2)
        lorentz2 = ExponentialGaussianModel(prefix='l2_')
        pars2.update(lorentz2.make_params())
        pars2['l2_center'].set(value=-0.15, min=-0.5, max=-0.05)
        pars2['l2_sigma'].set(value=0.05, min=0)
        pars2['l2_amplitude'].set(value=0.01, min=0)
        pars2['l2_gamma'].set(value=5, min=1)

        mod2 = lorentz2
        init = mod2.eval(pars2, x=x2)
        out2 = mod2.fit(y2, pars2, x=x2)

        # print(out1.fit_report(min_correl=0.5))
        # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
        ax2.plot(x1, out1.best_fit,'grey',-np.flipud(x2), np.flipud(out2.best_fit),'grey')
        # ax2.plot(x2, out2.best_fit, 'grey', x1, out1.best_fit, 'grey')
        ax2.legend(loc='best')

##################整组数据拟合 1（左右一起)##############################
        # x = pdf_rot_dict[filename_1[i] + 'x']
        # y = pdf_rot_dict[filename_1[i] + 'y']
        #
        # exp_mod1 = ExponentialModel(prefix='exp_')
        # pars1 = exp_mod1.guess(y, x=x)
        # lorentz1 = ExponentialGaussianModel(prefix='l1_')
        # pars1.update(lorentz1.make_params())
        # pars1['l1_center'].set(value=0.15, max=0.5, min=0.05)
        # pars1['l1_sigma'].set(value=0.05, min=0)
        # pars1['l1_amplitude'].set(value=0.01, min=0)
        # pars1['l1_gamma'].set(value=5, min=1)
        # #
        #
        # lorentz2 = ExponentialGaussianModel(prefix='l2_')
        # pars1.update(lorentz2.make_params())
        # pars1['l2_center'].set(value=-0.15, min=-0.5, max=-0.05)
        # pars1['l2_sigma'].set(value=0.05, min=0)
        # pars1['l2_amplitude'].set(value=0.01, min=0)
        # pars1['l2_gamma'].set(value=5, min=1)
        #
        # mod1 = lorentz1  + lorentz2
        #
        #
        # init = mod1.eval(pars1, x=x)
        # out1 = mod1.fit(y, pars1, x=x)
        #
        # print(out1.fit_report(min_correl=0.5))
        # # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
        # ax2.plot(x, out1.best_fit)
        # ax2.legend(loc='best')
# # #########################################################################################################


#
#         # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
#         # axes[0].plot(x, y, 'bo-')
#         # axes[0].plot(x, init, 'k--', label='initial fit')
#         # axes[0].plot(x, out.best_fit, 'r-', label='best fit')
#         # axes[0].legend(loc='best')
#         #
#         # comps = out.eval_components(x=x)
#         # axes[1].plot(x, y, 'b')
#         # axes[1].plot(x, comps['g1_'], 'g--', label='Gaussian component 1')
#         # axes[1].plot(x, comps['g2_'], 'm--', label='Gaussian component 2')
#         # axes[1].plot(x, comps['g3_'], 'k--', label='Gaussian component 3')
#         # axes[1].legend(loc='best')
#         #
#         # plt.show()
#     # ------------------------------


    ax2.set_title('Rotational PDF', fontsize=10)
    ax2.set_xlabel('$\Delta theta (rad)$')
    ax2.set_ylabel('P')
    # ---------------------------------------------------------------------------
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim((0.0006, 0.03))
    ax2.set_ylim((0.0006, 0.03))

    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)

    plt.show()
    # fig.savefig(file_s2[p])



