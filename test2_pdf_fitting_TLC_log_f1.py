# 读取hdf文件，计算pdf
# 出D:\guan2019\1_disk\f\下 同一个f不同Δt的数据的pdf图像
# tranlational 是linear fit的

from scipy.special import xlogy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
import seaborn as sns
from lmfit.models import GaussianModel, LorentzianModel,LognormalModel, DampedOscillatorModel,ExponentialModel,VoigtModel,PseudoVoigtModel,SkewedVoigtModel,SplitLorentzianModel,DampedHarmonicOscillatorModel,ExponentialGaussianModel,ExpressionModel,SkewedGaussianModel
from scipy.optimize import curve_fit
from sympy import DiracDelta
from scipy import interpolate
from matplotlib.ticker  import MultipleLocator


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

def funce(x, a, b, c):
    return (a*np.exp(-b*x**c)) #(a*x**(c-1)*np.exp(-b*x**c)) #

def funcer(x, a, b, c,d,e): # rotational 整体fit
    return a**2 * np.exp(-(x - b)**2 / c**2) + a**2 * np.exp(-(x + b)**2 / c**2) + d**2 * np.exp(-x**2 / e**2)  # 3 Gaussian
    # return a**2 / ((x - c)**2 + b**2) + a**2  / ((x + c)**2 + b**2) + d**2 / (x**2 + e**2) # 3 Lorenzian
    # return  a**2 * np.exp(-(x - b)**2 / c**2) + a**2 * np.exp(-(x + b)**2 / c**2) + d**2 / (x**2 + e**2) # 2 Lorenzian +Gaussian
def funcer1(x, a, b, d,e,f): # rotational 分开fit
    return (a*np.exp(-b*x) + d* np.exp(-(x - e)**2 / f**2) )  # exp + Gaussian

def frac_reduc(n,m):
    n = int(n)
    m = int(m)
    for i in range(2, n):
        while (n % i == 0 and m % i == 0):
            n = n // i
            m = m // i
    return (n,m)

def smooth(x, window_len=3, window='flat'):
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
    pdf_transenergy_dict = {}
    pdf_rotenergy_dict = {}
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
        jump = np.array([1,2,5,10,15]) #[1,3,5,10,15,30,50,100,150] [1,30,50,100,150][1,2,5,10,15]

        for step in jump:
            center = center_1[::step]
            theta = theta_1[::step]
            deltax = list(center[1:, 0] - center[:-1, 0])
            deltax.extend(list(list(center[1:, 1] - center[:-1, 1])))
            deltatheta = list(theta[1:] - theta[:-1])
            for step1 in range(1,step):
                center = center_1[step1::step]
                theta = theta_1[step1::step]
                deltax.extend(list(center[1:, 0] - center[:-1, 0]))
                deltax.extend(list(center[1:, 1] - center[:-1, 1]))
                deltatheta.extend(list(theta[1:] - theta[:-1]))

            deltax = np.array(deltax)
            deltatheta = np.array(deltatheta)


            # -------------------------------------
            N = len(theta)
            max_time = N / fps  # seconds
            frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
            print(frame_name)
            if len(frame_name) > 3:
                frame_name = frame_name[1:]

            # x = center[:, 0]  # numpy array
            # y = center[:, 1]
            #
            # # delatX单位是pixel和角度，dX单位是mm和rad
            # deltax = center[1:, 0] - center[:-1, 0]  # numpy array
            # deltay = center[1:, 1] - center[:-1, 1]
            # THETA = theta.reshape(len(center))  # np.zeros(shape=(len(center), 3))
            # deltar = np.sqrt(deltax ** 2 + deltay ** 2)
            # deltatheta = THETA[1:] - THETA[:-1]
            #
            index = []
            for k in range(len(deltatheta)):
                if deltatheta[k] > 120:  # 处理周期行导致的大deltatheta
                    deltatheta[k] -= 180
                elif deltatheta[k] < -120:
                    deltatheta[k] += 180

            # index = []
            # for k in range(len(deltatheta)):
            #     if abs(deltatheta[k]) > 20 or abs(deltax[k]) > 3 or abs(deltay[k]) > 3:  # 把明显由于识别错误产生的零星数据删掉
            #         index.append(k)
            #
            # deltax = np.delete(deltax, index)
            # # deltay = np.delete(deltay, index)
            # deltar = np.delete(deltar, index)
            # deltatheta = np.delete(deltatheta, index)

            dtheta = deltatheta / 180 * math.pi
            dx = deltax / 480 * 260
            # dy = deltay / 480 * 260
            # dr = deltar / 480 * 260

            time = np.around(np.arange(0, len(dtheta) * 1 / 150*step, 1 / 150*step), decimals=2)

            deltatheta = dtheta  #弧度
            deltar = dx






    # 有两种 bins=71  /31(log)，bins=31(点平滑,但是太少了,作图时需要/2来与71结果一致)，拟合用bins=71（点多，略凌乱，获得的结果的概率接近TL）
            # translation-------------
            # [-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5]
            # [x*0.06+0.03 for x in range(-39,40)]
            # au, bu, cu = plt.hist(deltar,[x*0.12+0.06 for x in range(-59,60)], histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar 31
            au, bu, cu = plt.hist(deltar,[x*0.06+0.03 for x in range(-39,40)], histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar 31
            # au, bu, cu = plt.hist((deltar-np.mean(deltar)) / (np.sqrt(2 * np.sum(deltar ** 2) / len(deltar))), 41,range=(-3, 3),histtype='bar',facecolor='yellowgreen', alpha=0.75,rwidth=1)  # , density=True)  # au是counts，bu是deltar

            au /= len(deltar)
            au_ind = np.where(au == 0)
            au = np.delete(au,au_ind)
            au1 = np.array([math.log(x / 2) for x in au])
            # au1 = np.array([x/2 for x in au])
            bu = (bu[:-1]+bu[1:])/2.
            bu = np.delete(bu,au_ind)
            if filename_1[i]=='9100_2':
                if step ==10 or step ==2:
                    bu_ind = bu[np.where(au1 == np.max(au1))[0][0]-1]
                else:
                    bu_ind = bu[np.where(au1 == np.max(au1))[0][0]]
                # print(BU_ind,BU-BU_ind)
                pdf_trans_dict[filename_1[i]+'x'+str(step)] = bu -bu_ind # 存入dict
            else:
                pdf_trans_dict[filename_1[i] + 'x' + str(step)] = bu
            # pdf_trans_dict[filename_1[i]+'x'+str(step)] = bu # 存入dict
            pdf_trans_dict[filename_1[i] + 'y'+str(step)] = au1#*100 #- np.min(au1)

            # rotation-----------------------
            # [x*0.026+0.013 for x in range(-49,50)]
            # AU, BU, CU = plt.hist(deltatheta, [x*0.013+0.0065 for x in range(-149,150)],histtype='bar', facecolor='blue', alpha=0.75, rwidth=0.2)  # , density=True) # 无归一系数
            AU, BU, CU = plt.hist(deltatheta, 51, histtype='bar', facecolor='blue', alpha=0.75, rwidth=0.2 , density=True) # 无归一系数
            # AU, BU, CU = plt.hist((deltatheta-np.mean(deltatheta)) /(np.sqrt(2*np.sum(deltatheta**2)/len(deltatheta))), 61, range=(-3, 3), histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)  #归一系数，但是range很奇怪
            AU /= len(deltar)
            AU_ind = np.where(AU == 0)
            AU = np.delete(AU,AU_ind)
            AU1 = np.array([x/2 for x in AU])
            BU = (BU[:-1] + BU[1:]) / 2.
            BU = np.delete(BU,AU_ind)

            if filename_1[i]=='60_3':
                if step ==100:
                    BU_ind = BU[np.where(AU1 == np.max(AU1))[0][0]+1]
                else:
                    BU_ind = BU[np.where(AU1 == np.max(AU1))[0][0]]
                # print(BU_ind,BU-BU_ind)
                pdf_rot_dict[filename_1[i]+'x'+str(step)] = BU -BU_ind # 存入dict
            else:
                pdf_rot_dict[filename_1[i] + 'x' + str(step)] = BU
            # pdf_rot_dict[filename_1[i] + 'x' + str(step)] = BU
            pdf_rot_dict[filename_1[i] + 'y'+str(step)] = AU1*400 #- np.min(au1)

            delta_dict[filename_1[i]+'r'+str(step)] = deltar
            delta_dict[filename_1[i]+'theta'+str(step)] = deltatheta
            label.append(frame_name + ' Hz')



    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}


    ys = [i + deltax + (i * deltax) ** 2 for i in np.array(range(len(jump)))]# range(len(filename_1))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    marker = ['o', 'v', 'D', '^', 's','h', '2',  'p', '*', '+', 'x']

    # fig = plt.figure(figsize=(4,4))
    # ax1 = fig.add_subplot(111)

    for i in range(len(filename_1)):#2,3):#
        j =0
        fig = plt.figure(figsize=(4,4))
        ax1 = fig.add_subplot(111)
        for step in jump:
            if step < 150:
                # ax1.scatter(pdf_trans_dict[filename_1[i] + 'x'], np.clip(pdf_trans_dict[filename_1[i] + 'y'], 1e-18, None)/2, alpha=0.75,color=colors[i], label=label[i])
                ax1.scatter(0.8*pdf_trans_dict[filename_1[i] + 'x'+str(step)], 1000*np.exp(pdf_trans_dict[filename_1[i] + 'y'+str(step)]),c='',  marker=marker[j], s = 25, edgecolor=colors[j], label=str(frac_reduc(150.0,step)[1])+'/'+str(frac_reduc(150.0,step)[0])+' s')
                # marker[j], s = 25     'o',s=10

                # ax1.plot(pdf_trans_dict[filename_1[i] + 'x' + str(step)],pdf_trans_dict[filename_1[i] + 'y' + str(step)],  alpha=0.6,c=colors[j])
            else:
                # ax1.scatter(pdf_trans_dict[filename_1[i] + 'x'], np.clip(pdf_trans_dict[filename_1[i] + 'y'], 1e-18, None)/2, alpha=0.75,color=colors[i], label=label[i])
                ax1.scatter(0.8*pdf_trans_dict[filename_1[i] + 'x' + str(step)],1000*np.exp(pdf_trans_dict[filename_1[i] + 'y'+str(step)]), c='', alpha=0.65,marker=marker[j], s = 25,edgecolor=colors[j],label=str(np.around(step/150.0, decimals=1)) + ' s')
                # marker[j], s = 25     'o',s=10

                # ax1.plot(pdf_trans_dict[filename_1[i] + 'x' + str(step)],pdf_trans_dict[filename_1[i] + 'y' + str(step)],  alpha=0.6,c=colors[j])
            #
            # # # GaussianModel()------------
            # # only fit -1<x<1 part
            # # index0 = np.where(pdf_trans_dict[filename_1[i] + 'x' + str(step)] < 1)[0]
            # x = pdf_trans_dict[filename_1[i] + 'x' + str(step)]#[index0]
            # y = pdf_trans_dict[filename_1[i] + 'y' + str(step)]#[index0]
            # # index1 = np.where(x > -1)[0]
            # # x = x[index1]
            # # y = y[index1]
            # mod = GaussianModel()  # PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #ExponentialModel() #GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #
            # pars = mod.guess(y, x=x)
            # out = mod.fit(y, pars, x=x)
            # print(out.fit_report(min_correl=0.25))
            #
            # xnew = np.linspace(x.min(), x.max(), 1000)  # 平滑画曲线
            # funcax2 = interpolate.interp1d(x, out.best_fit, kind='cubic')
            # ynew = funcax2(xnew)
            #
            # ax1.plot(xnew, ynew, color=colors[j], alpha=0.5)
            # # ax1.legend()


            ########## Translatioanl fitting 1
            middle =np.where(pdf_trans_dict[filename_1[i] + 'x'+str(step)] == find_nearest(pdf_trans_dict[filename_1[i] + 'x'+str(step)], 0))[0][0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
            # x = pdf_trans_dict[filename_1[i] + 'x'][middle:] # ExponentialModel()
            # y = np.clip(pdf_trans_dict[filename_1[i] + 'y'], 1e-18, None) [middle:] # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())

            ########## Translatioanl fitting 1 # function fitting
            # fittin the -x axis
            x1 = pdf_trans_dict[filename_1[i] + 'x'+str(step)][middle:1:-1]  # ExponentialModel()
            y1 = pdf_trans_dict[filename_1[i] + 'y'+str(step)][middle:1:-1] - np.min(pdf_trans_dict[filename_1[i] + 'y'+str(step)])   # ExponentialModel()
            popt1, pcov1 = curve_fit(funce, -x1, y1, maxfev=10000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            # print(x1,y1,popt1,pcov1)
            # fittin the +x axis
            x2 = pdf_trans_dict[filename_1[i] + 'x'+str(step)][middle:-1]  # ExponentialModel()
            y2 = pdf_trans_dict[filename_1[i] + 'y'+str(step)][middle:-1]- np.min(pdf_trans_dict[filename_1[i] + 'y'+str(step)])    # ExponentialModel()
            popt2, pcov2 = curve_fit(funce, x2, y2, maxfev=10000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            popt = (popt1 + popt2) / 2.
            pcov = (pcov1 + pcov2) / 2.
            # print(popt,popt1,popt2)

            xnew = np.linspace(x2.min()-0.3, x2.max()+0.3, 100)  # 平滑画曲线
            y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
            print(filename[i])
            print(popt)  # 振幅要*1/2（第一个系数）
            print(pcov)
            print(len(xnew),len(y11),j)
            ax1.plot(-xnew[::-1]*0.8, 1000*np.exp(y11+np.min(pdf_trans_dict[filename_1[i] + 'y'+str(step)]))[::-1], color=colors[j],alpha=0.6)
            ax1.plot(0.8*xnew, 1000*np.exp(y11+np.min(pdf_trans_dict[filename_1[i] + 'y'+str(step)])), color=colors[j], alpha=0.6)

            j +=1

        if filename_1[i] == '9100_2':
            left, bottom, width, height = [0.7, 0.715, 0.19, 0.15]
            ax3 = fig.add_axes([left, bottom, width, height])
            expo1_y = [3.14544321, 3.02551633, 3.0016551, 3.04887906,2.97791085]
            expo1_x = [1,2,5,10,15]#[np.around(1/150.0, decimals=2), np.around(2/150.0, decimals=2), np.around(5/150.0, decimals=2),np.around(10/150.0, decimals=2), np.around(15/150.0, decimals=2)]
            ax3.scatter(expo1_x, expo1_y, s=12)
            ax3.xaxis.set_major_locator(MultipleLocator(5))
            ax3.xaxis.set_minor_locator(MultipleLocator(1))
            ax3.yaxis.set_major_locator(MultipleLocator(0.1))
            ax3.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax3.set_xlim(0,17)
            ax3.set_ylim(2.9,3.2)
            ax3.set_xlabel(r'$150*\Delta t$ (s)', fontsize=9)
            ax3.set_ylabel(r'$\beta$', fontsize=9)
            ax3.xaxis.set_label_coords(0.6, -0.32)
            ax3.yaxis.set_label_coords(-0.32,0.5)
            ax3.tick_params(axis="x", direction="in", labelsize=9)
            ax3.tick_params(which='minor', direction='in')
            ax3.tick_params(axis="y", direction="in", labelsize=9)


        leg = ax1.legend(loc='upper left')
        leg.get_frame().set_linewidth(0.0)
        # plt.legend()

        ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax1.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax1.tick_params(axis="x", direction="in")
        ax1.tick_params(axis="y", direction="in")
        ax1.tick_params(which='minor', direction='in')
        # ax1.set_xlim((-3,3))
        ax1.set_ylim((0.0001,0.3))
        ax1.set_yscale('log')


        ax1.set_xlabel('$\Delta x$ (mm)')  # ('$\Delta x(pixel)$')  #
        ax1.set_ylabel('$P(\Delta x)$')
        plt.show()






    # ax2 = fig.add_subplot(122)

    for i in range(len(filename_1)):#2,3):#
        j = 0
        fig = plt.figure(figsize=(4,4))
        ax2 = fig.add_subplot(111)
        for step in jump:
            middle = np.where(pdf_rot_dict[filename_1[i] + 'y' + str(step)] == np.max(pdf_rot_dict[filename_1[i] + 'y' + str(step)]))[0][0]  # pdf 0 的点
            print(middle)
            if step>=150:
                ax2.scatter(pdf_rot_dict[filename_1[i] + 'x'+str(step)], smooth(pdf_rot_dict[filename_1[i] + 'y'+str(step)])[1:len(pdf_rot_dict[filename_1[i] + 'x' + str(step)])+1], c='', alpha=0.65,marker=marker[j], s = 25 , edgecolor=colors[j], cmap='hsv', label= str(np.around(step/150.0, decimals=1))+' s')
                # marker[j], s = 25     'o',s=10
                # pdf_rot_dict[filename_1[i] + 'x'+str(step)]-pdf_rot_dict[filename_1[i] + 'x'+str(step)][middle]

                # ax2.plot(pdf_rot_dict[filename_1[i] + 'x'+str(step)], pdf_rot_dict[filename_1[i] + 'y'+str(step)], color=colors[j], alpha=0.6)
            else:
                print(frac_reduc(step,150.0))
                ax2.scatter(pdf_rot_dict[filename_1[i] + 'x' + str(step)],smooth(pdf_rot_dict[filename_1[i] + 'y' + str(step)])[1:len(pdf_rot_dict[filename_1[i] + 'x' + str(step)])+1], c='',marker=marker[j], s = 25, edgecolor=colors[j],cmap='hsv', label=str(frac_reduc(150.0,step)[1])+'/'+str(frac_reduc(150.0,step)[0])+' s')
                # marker[j], s = 25     'o',s=10
                # pdf_rot_dict[filename_1[i] + 'x'+str(step)]-pdf_rot_dict[filename_1[i] + 'x'+str(step)][middle]

                # ax2.plot(pdf_rot_dict[filename_1[i] + 'x' + str(step)], pdf_rot_dict[filename_1[i] + 'y' + str(step)],color=colors[j], alpha=0.6)
            if step ==15:
        # # GaussianModel()------------
                # only fit -1<x<1 part
                # # index0 = np.where(pdf_rot_dict[filename_1[i] + 'x'+str(step)]-pdf_rot_dict[filename_1[i] + 'x'+str(step)][middle]<1)[0]
                # # x = (pdf_rot_dict[filename_1[i] + 'x'+str(step)]-pdf_rot_dict[filename_1[i] + 'x'+str(step)][middle])[index0]
                # index0 = np.where(pdf_rot_dict[filename_1[i] + 'x'+str(step)]<1)[0]
                x = (pdf_rot_dict[filename_1[i] + 'x'+str(step)])#[index0]
                y = pdf_rot_dict[filename_1[i] + 'y'+str(step)]#[index0]
                # index1 = np.where(x>-1)[0]
                # x = x[index1]
                # y = y[index1]
                mod = GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #ExponentialModel() #GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #
                pars = mod.guess(y, x=x)
                out = mod.fit(y, pars, x=x)
                print(out.fit_report(min_correl=0.25))

                xnew = np.linspace(x.min(), x.max(), 1000)  # 平滑画曲线
                funcax2 = interpolate.interp1d(x, out.best_fit, kind='cubic')
                ynew= funcax2(xnew)

                ax2.plot(xnew, ynew,color=colors[j],alpha=0.5)

            j += 1

        print(filename[i])

        leg = ax2.legend(loc='upper left')
        leg.get_frame().set_linewidth(0.0)
        # plt.legend()

        ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.005))
        ax2.tick_params(axis="x", direction="in")
        ax2.tick_params(axis="y", direction="in")
        ax2.tick_params(which='minor', direction='in')
        ax2.set_xlim((-1.75,1.75))
        ax2.set_ylim((0.0001, 0.3))
        ax2.set_yscale('log')

        ax2.set_xlabel('$\Delta \Theta$ (rad)')  # ('$\Delta theta (degree)$')  #
        ax2.set_ylabel('$P(\Delta \Theta )$')

        plt.show()




    # # method 1     整体fit -------------------------------------
    #         x = pdf_rot_dict[filename_1[i] + 'x']*100
    #         y = pdf_rot_dict[filename_1[i] + 'y']-np.min(pdf_rot_dict[filename_1[i] + 'y'])
    #
    #         popt, pcov = curve_fit(funcer, x, y, maxfev=100000)#,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
    #         print(popt)
    #
    #         xnew = np.linspace(x.min(), x.max(), 10000) # 平滑画曲线
    #         y11 = [funcer(p, popt[0], popt[1], popt[2], popt[3],popt[4]) for p in xnew]
    #         print(filename[i])
    #         ax2.plot(xnew/100.0, np.exp(y11+np.min(pdf_rot_dict[filename_1[i] + 'y'])), color=colors[i],label=label[i],alpha=0.6)

    # # method 2    fit一半   -------------------------------------------
    #         middle = np.where(pdf_rot_dict[filename_1[i] + 'x'] == find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
    #         # x = pdf_trans_dict[filename_1[i] + 'x'][middle:] # ExponentialModel()
    #         # y = np.clip(pdf_trans_dict[filename_1[i] + 'y'], 1e-18, None) [middle:] # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
    #
    #         # fittin the -x axis
    #         x1 = pdf_rot_dict[filename_1[i] + 'x'][middle::-1]  # ExponentialModel()
    #         y1 = pdf_rot_dict[filename_1[i] + 'y'][middle::-1] - np.min(pdf_rot_dict[filename_1[i] + 'y'])  # ExponentialModel()
    #         print(type(filename[i]))
    #         if int(filename_1[i].split('_', 1)[0]) == 50:
    #             popt1, pcov1 = curve_fit(funce, -x1, y1, maxfev=1000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
    #         else:
    #             popt1, pcov1 = curve_fit(funcer1, -x1, y1, maxfev=1000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
    #         # print(x1,y1,popt1,pcov1)
    #         # fittin the +x axis
    #         x2 = pdf_rot_dict[filename_1[i] + 'x'][middle:-1]  # ExponentialModel()
    #         y2 = pdf_rot_dict[filename_1[i] + 'y'][middle:-1] - np.min(pdf_rot_dict[filename_1[i] + 'y'])  # ExponentialModel()
    #         if int(filename_1[i].split('_', 1)[0]) == 50:
    #             popt1, pcov1 = curve_fit(funce, x2, y2, maxfev=10000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
    #         else:
    #             popt2, pcov2 = curve_fit(funcer1, x2, y2, maxfev=10000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
    #         popt = (popt1 + popt2) / 2.
    #         pcov = (pcov1 + pcov2) / 2.
    #         # print(popt,popt1,popt2)
    #
    #         xnew = np.linspace(x2.min(), x2.max(), 100)  # 平滑画曲线
    #         y11 = [funcer1(p, popt[0], popt[1], popt[2],popt[3], popt[4]) for p in xnew]
    #         print(filename[i])
    #         print(popt)  # 振幅要*1/2（第一个系数）
    #         print(pcov)
    #         ax1.plot(-xnew[::-1], np.exp(y11 + np.min(pdf_rot_dict[filename_1[i] + 'y']))[::-1], color=colors[i],alpha=0.6)
    #         ax1.plot(xnew, np.exp(y11 + np.min(pdf_rot_dict[filename_1[i] + 'y'])), color=colors[i], label=label[i],alpha=0.6)
    #


            # if int(filename_1[i].split('_', 1)[0]) == 50:
            #     Gau_mod1 = GaussianModel(prefix='gau_')
            #     pars1 = Gau_mod1.guess(y, x=x)
            #
            #     all = Gau_mod1
            #
            #     init = all.eval(pars1, x=x)
            #     out1 = all.fit(y, pars1, x=x)
            #
            # else:
            #     Gau_mod1 = GaussianModel(prefix='gau_')
            #     pars1 = Gau_mod1.guess(y, x=x)
            #     mod1 = GaussianModel(prefix='l1_')  # GaussianModel LorentzianModel SplitLorentzianModel  DampedHarmonicOscillatorModel   ExponentialGaussianModel VoigtModel
            #     pars1.update(mod1.make_params())
            #
            #     pars1['l1_center'].set(value=0.2, min=0.05, max=0.5)  # ExponentialGaussianModel
            #     pars1['l1_sigma'].set(value=0.05, min=0)
            #     pars1['l1_amplitude'].set(value=2, min=0)
            #
            #     mod2 = GaussianModel(prefix='l2_')
            #     pars1.update(mod2.make_params())
            #
            #     pars1['l2_center'].set(value=-0.2, max=-0.05, min=-0.5)  # Exponential GaussianModel
            #     pars1['l2_sigma'].set(value=0.05, min=0)
            #     pars1['l2_amplitude'].set(value=2, min=0)
            #
            #     mod3 = GaussianModel(prefix='l3_')
            #     pars1.update(mod3.make_params())
            #
            #     pars1['l3_center'].set(value=0, max=-0.0025, min=0.0025)  # Exponential GaussianModel
            #     pars1['l3_sigma'].set(value=0.015, min=0)
            #     pars1['l3_amplitude'].set(value=10, min=0)
            #
            #
            #     all = mod1 + mod2 + mod3
            #
            #     init = all.eval(pars1, x=x)
            #     out1 = all.fit(y, pars1, x=x)
            #
            # print(filename_1[i])
            # print(out1.fit_report(min_correl=0.5))
            #
            # xnew = np.linspace(x.min(), x.max(), 100)  # 平滑画曲线
            # # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
            # ax2.plot(xnew, np.exp(out1.best_fit+np.min(pdf_rot_dict[filename_1[i] + 'y'])),color=colors[i],alpha=0.5,label=label[i])
            # # ax2.plot(x, out1.best_fit, color=colors[i], alpha=0.5,label=label[i])
            #
            # # plt.show()
            # # ax2.legend(loc='best')

        # # ##########整组数据拟合 2 （左右分开）#########################################################
        # #         middle = np.where(pdf_rot_dict[filename_1[i] + 'x']==find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][0]  #pdf 0 的点
        # #         x1 = pdf_rot_dict[filename_1[i] + 'x'][:middle-2]  # 越过中间的几个点
        # #         y1 = pdf_rot_dict[filename_1[i] + 'y'][:middle-2]
        # #         x2 = -np.flipud(pdf_rot_dict[filename_1[i] + 'x'][middle+2:])
        # #         y2 = np.flipud(pdf_rot_dict[filename_1[i] + 'y'][middle+2:])
        # #
        # #         exp_mod1 = ExponentialModel(prefix='exp_')
        # #         pars1 = exp_mod1.guess(y1, x=x1)
        # #         lorentz1 = ExponentialGaussianModel(prefix='l1_')
        # #         pars1.update(lorentz1.make_params())
        # #         pars1['l1_center'].set(value=-0.15, min=-0.0, max=-0.05)
        # #         pars1['l1_sigma'].set(value=0.05, min=0)
        # #         pars1['l1_amplitude'].set(value=0.01, min=0)
        # #         pars1['l1_gamma'].set(value=5, min=1)
        # #
        # #         mod1 = lorentz1 #+ lorentz2
        # #         init = mod1.eval(pars1, x=x1)
        # #         out1 = mod1.fit(y1, pars1, x=x1)
        # #
        # #         exp_mod2 = ExponentialModel(prefix='exp_')
        # #         pars2 = exp_mod2.guess(y2, x=x2)
        # #         lorentz2 = ExponentialGaussianModel(prefix='l2_')
        # #         pars2.update(lorentz2.make_params())
        # #         pars2['l2_center'].set(value=-0.15, min=-0.5, max=-0.05)
        # #         pars2['l2_sigma'].set(value=0.05, min=0)
        # #         pars2['l2_amplitude'].set(value=0.01, min=0)
        # #         pars2['l2_gamma'].set(value=5, min=1)
        # #
        # #         mod2 = lorentz2
        # #         init = mod2.eval(pars2, x=x2)
        # #         out2 = mod2.fit(y2, pars2, x=x2)
        # #
        # #         # print(out1.fit_report(min_correl=0.5))
        # #         # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
        # #         ax2.plot(x1, out1.best_fit,'grey',-np.flipud(x2), np.flipud(out2.best_fit),'grey')
        # #         # ax2.plot(x2, out2.best_fit, 'grey', x1, out1.best_fit, 'grey')
        # #         ax2.legend(loc='best')
        #
        # ##################整组数据拟合 1（左右一起) >=65Hz ##############################
        #     if int(filename_1[i].split('_', 1)[0])>=65:
        #         middle = np.where(pdf_rot_dict[filename_1[i] + 'x'] == find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
        #
        #         x = np.delete(pdf_rot_dict[filename_1[i] + 'x'], middle-1)
        #         y = np.delete(pdf_rot_dict[filename_1[i] + 'y'], middle-1)-np.min(pdf_rot_dict[filename_1[i] + 'y'])
        #
        #         exp_mod1 = LorentzianModel(prefix='exp_')
        #         pars1 = exp_mod1.guess(y, x=x)
        #         lorentz1 =GaussianModel(prefix='l1_')  # GaussianModel LorentzianModel SplitLorentzianModel  DampedHarmonicOscillatorModel   ExponentialGaussianModel VoigtModel
        #         pars1.update(lorentz1.make_params())
        #         # 归一系数 参数
        #         # pars1['l1_center'].set(value=0.8, max=0.9, min=0.6)
        #         # pars1['l1_sigma'].set(value=0.1, min=0.0001)
        #         # pars1['l1_amplitude'].set(value=0.09, min=0.05)
        #
        #         # 无归一系数参数
        #         pars1['l1_center'].set(value=0.2, min=0.05, max=0.5)  # ExponentialGaussianModel
        #         pars1['l1_sigma'].set(value=0.05, min=0)
        #         pars1['l1_amplitude'].set(value=2, min=0)
        #
        #
        #         lorentz2 = GaussianModel(prefix='l2_')
        #         pars1.update(lorentz2.make_params())
        #         # 归一
        #         # pars1['l2_center'].set(value=-0.8, min=-0.9, max=-0.6)
        #         # pars1['l2_sigma'].set(value=0.1, min=0.0001)
        #         # pars1['l2_amplitude'].set(value=0.09, min=0.05)
        #
        #         # 无
        #         pars1['l2_center'].set(value=-0.2, max=-0.05, min=-0.5)  # Exponential GaussianModel
        #         pars1['l2_sigma'].set(value=0.05, min=0)
        #         pars1['l2_amplitude'].set(value=2, min=0)
        #
        #
        #         mod1 = lorentz1  + lorentz2
        #
        #
        #         init = mod1.eval(pars1, x=x)
        #         out1 = mod1.fit(y, pars1, x=x)
        #
        #         print(filename_1[i])
        #         print(out1.fit_report(min_correl=0.5))
        #         # ax2.plot(x, y, 'bo-',alpha=0.6,marker='o',markersize=3)
        #         ax2.plot(x, np.exp(out1.best_fit+np.min(pdf_rot_dict[filename_1[i] + 'y'])),color=colors[i],alpha=0.5,label=label[i])
        #         # ax2.plot(x, out1.best_fit, color=colors[i], alpha=0.5,label=label[i])
        #
        #         # plt.show()
        #         # ax2.legend(loc='best')
        #
        # ##################整组数据拟合 1（左右一起) <=60Hz ##############################
        #     elif int(filename_1[i].split('_', 1)[0])<=60:
        #         middle = np.where(pdf_rot_dict[filename_1[i] + 'x'] == find_nearest(pdf_rot_dict[filename_1[i] + 'x'], 0))[0][
        #             0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
        #
        #         x = np.delete(pdf_rot_dict[filename_1[i] + 'x'], middle-1)
        #         y = np.delete(pdf_rot_dict[filename_1[i] + 'y'], middle-1)-np.min(pdf_rot_dict[filename_1[i] + 'y'])
        #
        #
        #         exp_mod1 = LorentzianModel(prefix='exp_')
        #         pars1 = exp_mod1.guess(y, x=x)
        #         lorentz1 = LorentzianModel(prefix='l1_')  # GaussianModel LorentzianModel SplitLorentzianModel  DampedHarmonicOscillatorModel   ExponentialGaussianModel VoigtModel
        #         pars1.update(lorentz1.make_params())
        #         pars1['l1_center'].set(value=0.15, max=0.5, min=0)
        #         pars1['l1_sigma'].set(value=0.1, min=0.0001)
        #         pars1['l1_amplitude'].set(value=0.1, min=0.0001)
        #         # pars1['l1_center'].set(value=0.15, min=0.05, max=0.5)  # ExponentialGaussianModel
        #         # pars1['l1_sigma'].set(value=0.05, min=0)
        #         # pars1['l1_amplitude'].set(value=0.01, min=0)
        #         # pars1['l1_gamma'].set(value=5, min=1)
        #
        #         lorentz2 = LorentzianModel(prefix='l2_')
        #         pars1.update(lorentz2.make_params())
        #         pars1['l2_center'].set(value=-0.15, min=-0.5, max=0)
        #         pars1['l2_sigma'].set(value=0.1, min=0.0001)
        #         pars1['l2_amplitude'].set(value=0.1, min=0.0001)
        #         # pars1['l2_center'].set(value=-0.15, max=-0.05, min=-0.5)  # ExponentialGaussianModel
        #         # pars1['l2_sigma'].set(value=0.05, min=0)
        #         # pars1['l2_amplitude'].set(value=0.01, min=0)
        #         # pars1['l2_gamma'].set(value=5, min=1)
        #
        #         mod1 = lorentz1  + lorentz2
        #
        #
        #         init = mod1.eval(pars1, x=x)
        #         out1 = mod1.fit(y, pars1, x=x)
        #
        #         print(filename_1[i])
        #         print(out1.fit_report(min_correl=0.5))
        #
        #         ax2.plot(x, np.exp(out1.best_fit+np.min(pdf_rot_dict[filename_1[i] + 'y'])),color=colors[i],alpha=0.5,label=label[i])
        #         # ax2.plot(x, out1.best_fit, color=colors[i], alpha=0.5,label=label[i])
        #         # plt.show()
        #         # ax2.legend(loc='best')
    # # #########################################################################################################
    #
    #     #
    #     # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    #     # axes[0].plot(x, y, 'bo-')
    #     # axes[0].plot(x, init, 'k--', label='initial fit')
    #     # axes[0].plot(x, out.best_fit, 'r-', label='best fit')
    #     # axes[0].legend(loc='best')
    #     #
    #     # comps = out.eval_components(x=x)
    #     # axes[1].plot(x, y, 'b')
    #     # axes[1].plot(x, comps['g1_'], 'g--', label='Gaussian component 1')
    #     # axes[1].plot(x, comps['g2_'], 'm--', label='Gaussian component 2')
    #     # axes[1].plot(x, comps['g3_'], 'k--', label='Gaussian component 3')
    #     # axes[1].legend(loc='best')
    #     #
    #     # plt.show()
    #     # # ------------------------------


    # ax2.set_title('Rotational PDF', fontsize=10)


