# 读取hdf文件，计算pdf
# 出D:\guan2019\1_disk\f\下 按fre分类的数据的pdf图像


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
import seaborn as sns
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel,MoffatModel, VoigtModel,Pearson7Model, StudentsTModel,DampedOscillatorModel,ExponentialModel,ExponentialGaussianModel,ExpressionModel,SkewedGaussianModel
from scipy.optimize import curve_fit

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

def func(x, a, b, c):
    return (a*np.exp(-b*np.abs(x)**c))

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
path2 = 'D:\\guan2019\\1_disk\\a\\5\\data\\'  # 60Hz
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]  # fre 文件夹
file_n = [path2 + name +'.h5' for name in filename]  # 每个fre下原始data的路径
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
    pdf_transenergy_dict = {}
    pdf_rotenergy_dict = {}
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
        #-------------------
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
        deltatheta = dtheta#/180*math.pi
        deltar = dx#/480*260

# -------------------------------------
        # deltar/np.sqrt(2*np.sum(deltar ** 2) / len(deltar)),range=(-6, 6),
        au, bu, cu = plt.hist(deltar ,71,  histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        pdf_trans_dict[filename_1[i]+'y'] = au/len(deltar)
        bu = (bu[:-1]+bu[1:])/2.
        pdf_trans_dict[filename_1[i]+'x'] = bu  # 存入dict

        # deltatheta/(np.sqrt(2*np.sum(deltatheta**2)/len(deltatheta)))
        AU, BU, CU = plt.hist(deltatheta, 71, histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)
        pdf_rot_dict[filename_1[i] + 'y'] = AU/len(deltar)
        BU = (BU[:-1]+BU[1:])/2.
        pdf_rot_dict[filename_1[i] + 'x'] = BU  # 存入dict

        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta

# energy---------
        # range=(0, 0.00006),
        aue, bue, cue = plt.hist(0.001 * (dr / 1000 * 150) ** 2, 71, histtype='bar', facecolor='yellowgreen',
                                 alpha=0.75, rwidth=1)  # , density=True)  # au是counts，bu是deltar
        pdf_transenergy_dict[filename_1[i] + 'y'] = aue / len(dr)
        bue = (bue[:-1] + bue[1:]) / 2.
        pdf_transenergy_dict[filename_1[i] + 'x'] = bue  # 存入dict

        # (0, 0.003)
        AUE, BUE, CUE = plt.hist(0.0000002 * (deltatheta * 150) ** 2 / 16, 71, histtype='bar', facecolor='blue',
                                 alpha=0.75, rwidth=0.2)  # , density=True)
        pdf_rotenergy_dict[filename_1[i] + 'y'] = AUE / len(dr)
        BUE = (BUE[:-1] + BUE[1:]) / 2.
        pdf_rotenergy_dict[filename_1[i] + 'x'] = BUE  # 存入dict

        delta_dict[filename_1[i] + 'r'] = deltar
        delta_dict[filename_1[i] + 'theta'] = deltatheta

    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}
    label = []

    ys = [i + x + (i * x) ** 2 for i in range(len(filename_1))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'r'],norm_hist=True, bins=200, kde=True, hist=False, label=filename_1[i].split('_', 1)[0] + 'Hz')  #'mode'+str(i+1))#

        plt.scatter(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], alpha=0.65,color=colors[i], label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.plot(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5,label=filename_1[i].split('_', 1)[0] + 'Hz')
        label.append(filename_1[i].split('_', 1)[0] + 'Hz')




########## Translatioanl fitting
        middle = np.where(pdf_rot_dict[filename_1[i] + 'x'] == find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][
            0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
     #Gaussian   ----------------------
        x = pdf_trans_dict[filename_1[i] + 'x']
        y = pdf_trans_dict[filename_1[i] + 'y']
        mod = GaussianModel()  # ExponentialModel()  # LorentzianModel()  # VoigtModel()  # SkewedVoigtModel()  ####  PseudoVoigtModel()  # #SplitLorentzianModel()  #
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)
        # print(out.fit_report(min_correl=0.25))
        ax1.plot(x, out.best_fit, color=colors[i], alpha=0.5)

    # #exponential     ------------------------
    #     x = pdf_trans_dict[filename_1[i] + 'x'][middle:] # ExponentialModel()
    #     y = pdf_trans_dict[filename_1[i] + 'y'][middle:] # ExponentialModel()
    #     if int(filename_1[i])>70:
    #         x = pdf_trans_dict[filename_1[i] + 'x'][middle-1:]  # ExponentialModel()
    #         y = pdf_trans_dict[filename_1[i] + 'y'][middle-1:]  # ExponentialModel()
    #     mod = ExponentialModel()
    #     pars = mod.guess(y, x=x)
    #     out = mod.fit(y, pars, x=x)
    #     # print(out.fit_report(min_correl=0.25))
    #     ax1.plot(x, out.best_fit, color=colors[i], alpha=0.5)

# #function fitting-----------
#         x = pdf_trans_dict[filename_1[i] + 'x']
#         y = pdf_trans_dict[filename_1[i] + 'y']
#         popt, pcov = curve_fit(func, x, y, maxfev=1000)
#         y11 = [func(p, popt[0], popt[1], popt[2]) for p in x]
#         print(filename[i])
#         print(popt)
#         print(pcov)
#         ax1.plot(x,y11, color=colors[i], alpha=0.5)


    # plt.legend(label)
    leg = plt.legend(label)
    leg.get_frame().set_linewidth(0.0)

    ax1.set_title('PDF' + ' [5g] ', fontsize=10)
    ax1.set_xlabel('$\Delta x(mm)$')  # ('$\Delta x(pixel)$')  #
    ax1.set_ylabel('$P(\Delta x)$')
    # ax1.set_xlim(-2.4, 2.4)
    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)


    ax2 = fig.add_subplot(122)


    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'theta'], norm_hist=True,bins=100, kde=True, hist=False,label=filename_1[i].split('_', 1)[0] + 'g')  #'mode'+str(i+1))#
        plt.scatter(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'],alpha=0.65,color=colors[i], label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.plot(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5,label=filename_1[i].split('_', 1)[0] + 'g')

# ########## 整组数据拟合 2 （左右分开）#########################################################
#         middle = np.where(pdf_rot_dict[filename_1[i] + 'x']==find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][0]  #pdf 0 的点
#         x1 = pdf_rot_dict[filename_1[i] + 'x'][:middle-5]  # 越过中间的几个点
#         y1 = pdf_rot_dict[filename_1[i] + 'y'][:middle-5]
#         x2 = -np.flipud(pdf_rot_dict[filename_1[i] + 'x'][middle+5:])
#         y2 = np.flipud(pdf_rot_dict[filename_1[i] + 'y'][middle+5:])
#
#         exp_mod1 = ExponentialModel(prefix='exp_')
#         pars1 = exp_mod1.guess(y1, x=x1)
#         lorentz1 = ExponentialGaussianModel(prefix='l1_')
#         pars1.update(lorentz1.make_params())
#         pars1['l1_center'].set(value=-0.15, min=-0.5, max=-0.05)
#         pars1['l1_sigma'].set(value=0.05, min=0)
#         pars1['l1_amplitude'].set(value=0.01, min=0)
#         pars1['l1_gamma'].set(value=5, min=1)
#
#         mod1 = lorentz1 #+ lorentz2
#         init = mod1.eval(pars1, x=x1)
#         out1 = mod1.fit(y1, pars1, x=x1)
#
#         exp_mod2 = ExponentialModel(prefix='exp_')
#         pars2 = exp_mod2.guess(y2, x=x2)
#         lorentz2 = ExponentialGaussianModel(prefix='l2_')
#         pars2.update(lorentz2.make_params())
#         pars2['l2_center'].set(value=-0.15, min=-0.5, max=-0.05)
#         pars2['l2_sigma'].set(value=0.05, min=0)
#         pars2['l2_amplitude'].set(value=0.01, min=0)
#         pars2['l2_gamma'].set(value=5, min=1)
#
#         mod2 = lorentz2
#         init = mod2.eval(pars2, x=x2)
#         out2 = mod2.fit(y2, pars2, x=x2)
#
#         # print(out1.fit_report(min_correl=0.5))
#         # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
#         ax2.plot(x1, out1.best_fit,'grey',-np.flipud(x2), np.flipud(out2.best_fit),'grey')
#         # ax2.plot(x2, out2.best_fit, 'grey', x1, out1.best_fit, 'grey')
#         ax2.legend(loc='best')

##################整组数据拟合 1（左右一起)##############################

        x = pdf_rot_dict[filename_1[i] + 'x']
        y = pdf_rot_dict[filename_1[i] + 'y']

# # 1------------
#         exp_mod1 = ExponentialModel(prefix='exp_')
#         pars1 = exp_mod1.guess(y, x=x)
#         lorentz1 = VoigtModel(prefix='l1_')  # GaussianModel LorentzianModel SplitLorentzianModel  DampedHarmonicOscillatorModel   ExponentialGaussianModel VoigtModel
#         pars1.update(lorentz1.make_params())
#         pars1['l1_center'].set(value=0)
#         pars1['l1_sigma'].set(value=0.1, min=0.01)
#         pars1['l1_amplitude'].set(value=0.1, min=0.01)
#
#         lorentz2 =LorentzianModel(prefix='l2_')
#         pars1.update(lorentz2.make_params())
#         pars1['l2_center'].set(value=-0)
#         pars1['l2_sigma'].set(value=0.001, min=0.0001)
#         pars1['l2_amplitude'].set(value=0.001, min=0.0001)
#
#
#         mod1 = lorentz1 + lorentz2
#         init = mod1.eval(pars1, x=x)
#         out1 = mod1.fit(y, pars1, x=x)
#
#         print(out1.fit_report(min_correl=0.5))
#         # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
#         # ax2.plot(x, out1.best_fit, color=colors[i], alpha=0.5)
#         ax2.legend(loc='best')

# # 2-------------
#         mod = GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  # LorentzianModel()  #
#         pars = mod.guess(y, x=x)
#         out = mod.fit(y, pars, x=x)
#         print(out.fit_report(min_correl=0.25))
#
#         ax2.plot(x, out.best_fit, color=colors[i], alpha=0.5)
#         ax2.legend(loc='best')

# 3------------
        exp_mod1 = ExponentialModel(prefix='exp_')
        pars1 = exp_mod1.guess(y, x=x)
        lorentz1 = VoigtModel(prefix='l1_')  # GaussianModel LorentzianModel SplitLorentzianModel  DampedHarmonicOscillatorModel   ExponentialGaussianModel VoigtModel
        pars1.update(lorentz1.make_params())
        pars1['l1_center'].set(value=-0.1)
        pars1['l1_sigma'].set(value=0.03, min=0.01)
        pars1['l1_amplitude'].set(value=0.01, min=0.005)

        lorentz2 = LorentzianModel(prefix='l2_')
        pars1.update(lorentz2.make_params())
        pars1['l2_center'].set(value=-0)
        pars1['l2_sigma'].set(value=0.001, min=0.0001)
        pars1['l2_amplitude'].set(value=0.001, min=0.0001)

        lorentz3 =  VoigtModel(prefix='l3_')  # GaussianModel LorentzianModel SplitLorentzianModel  DampedHarmonicOscillatorModel   ExponentialGaussianModel VoigtModel
        pars1.update(lorentz3.make_params())
        pars1['l3_center'].set(value=0.1)
        pars1['l3_sigma'].set(value=0.03, min=0.01)
        pars1['l3_amplitude'].set(value=0.01, min=0.005)

        mod1 = lorentz1 + lorentz2 + lorentz3
        init = mod1.eval(pars1, x=x)
        out1 = mod1.fit(y, pars1, x=x)

        print(out1.fit_report(min_correl=0.5))
        # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
        # ax2.plot(x, out1.best_fit, color=colors[i], alpha=0.5)
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

    # ax2.set_title('Rotational PDF', fontsize=10)
    ax2.set_xlabel('$\Delta \Theta (rad)$')#('$\Delta theta (degree)$')  #
    ax2.set_ylabel('$P(\Delta \Theta )$')
    # ---------------------------------------------------------------------------
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim((0.0006, 1))
    ax2.set_ylim((0.0006, 1))


    # plt.legend(label)
    leg = plt.legend(label)
    leg.get_frame().set_linewidth(0.0)

    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)



    plt.show()
    # fig.savefig(file_s2[p])



#----energy distribution
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(len(filename_1)):#2,3):#
        ax1.scatter(pdf_transenergy_dict[filename_1[i] + 'x']*1000, np.clip(pdf_transenergy_dict[filename_1[i] + 'y'], 1e-18, None), alpha=0.75,color=colors[i], label=label[i])

    leg = ax1.legend(label)
    leg.get_frame().set_linewidth(0.0)

    for i in range(len(filename_1)):  # 2,3):#
        ax2.scatter(pdf_rotenergy_dict[filename_1[i] + 'x']*1000, np.clip(pdf_rotenergy_dict[filename_1[i] + 'y'], 1e-18, None), alpha=0.75, color=colors[i], label=label[i])

    ax1.set_xlabel('${E_t(mJ)}$')
    ax1.set_ylabel('${P(E_t)}$')
    ax2.set_xlabel('${E_r(mJ)}$')
    ax2.set_ylabel('${P(E_r)}$')
    # ---------------------------------------------------------------------------
    ax1.set_yscale('log')
    ax2.set_yscale('log')


    plt.show()





