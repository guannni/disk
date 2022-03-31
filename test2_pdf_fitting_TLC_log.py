# 读取hdf文件，计算pdf
# 出D:\guan2019\1_disk\f\下 按fre分类的数据的pdf图像
# 能量用这个，v_distribution用test2_pdf_fitting_TLC.py

from scipy.special import xlogy
import pandas as pd
import matplotlib.pyplot  as plt
from matplotlib.ticker  import MultipleLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import numpy as np
import os.path
import math
import matplotlib.cm as cm
import seaborn as sns
from lmfit.models import LinearModel, GaussianModel,Pearson7Model, LorentzianModel,LognormalModel, DampedOscillatorModel,ExponentialModel,VoigtModel,PseudoVoigtModel,SkewedVoigtModel,SplitLorentzianModel,DampedHarmonicOscillatorModel,ExponentialGaussianModel,ExpressionModel,SkewedGaussianModel
from scipy.optimize import curve_fit
from sympy import DiracDelta
from scipy.interpolate import spline
from scipy import interpolate


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


def funcer0(x, a, b,c, d,e): # rotational 整体fit
    return 2.08**2* np.exp(-(x+b)**2 / c**2) +2.08**2 * np.exp(-(x-b )**2 / c**2) +0.000006** 2  * np.exp(-x**2 / e**2)  # 3 Gaussian 2 * np.exp(-(x+0.1 )**2 / c**2) +2 * np.exp(-(x-0.1 )**2 / c**2) +
def funcer1(x, a, b,c, d,e): # rotational 整体fit
    return 5.4 * np.exp(-(x - 7.9)**2 /9**2) + 5.4 * np.exp(-(x + 7.9)**2 / 9**2) + d ** 2 *np.exp(-x**2 / 2.5**2)  # 3 Gaussian
def funcer2(x, a, b, c, d, e):  # rotational 整体fit
    return 6.9* np.exp(-(x - 13) ** 2 / 13** 2) +6.9* np.exp(-(x + 13) ** 2 /13** 2) + 1.6 ** 2 * np.exp(-x ** 2 / 3.4** 2)  # 3 Gaussian
def funcer3(x, a, b, c, d, e):  # rotational 整体fit
    return 4.9* np.exp(-(x - 22) ** 2 / 13 ** 2) + 4.9* np.exp(-(x + 22) ** 2 / 13** 2) + d ** 2 * np.exp(-x ** 2 / 20 ** 2)  # 3 Gaussian
    # return a**2 / ((x - c)**2 + b**2) + a**2  / ((x + c)**2 + b**2) + d**2 / (x**2 + e**2) # 3 Lorenzian
    # return  a**2 * np.exp(-(x - b)**2 / c**2) + a**2 * np.exp(-(x + b)**2 / c**2) + d**2 / (x**2 + e**2) # 2 Lorenzian +Gaussian

# def funcer4(x, a, b, d,e,f): # rotational 分开fit
#     return (a*np.exp(-b*x) + d* np.exp(-(x - e)**2 / f**2) )  # exp + Gaussian

def to_percent(temp, position):
    return '%1.0f'%(10*temp) + '%'

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
path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # 100_originaldata\\' # TLC
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]  # fre 文件夹
file_n = [path2 + name +'.h5' for name in filename]  # 每个fre下原始data的路径
# file_delta = 'D:\\guan2019\\1_disk\\TLc\\all\\analysis_delta2\\pdf\\'  # \all\ 每个fre下pdf——delta的存储路径
file_delta = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis\\pdf\\'  # \tianli_f_all\每个fre下pdf——delta的存储路径
print(filename, file_n, file_delta)

err_trans_dict = {}
# x
# err_trans_dict['50_6']=[0,8.80712E-06,6.41633E-05,6.76425E-05,0.000562486,0.001594619,0.002634837,0.003785677,0.009267697,0.031315922,0.008135959,0.003715839,0.002190533,0.001474232,0.00062098,0.000198104,2.49544E-05,5.49302E-05,1.35951E-06,0]
# err_trans_dict['60_3']=[3.67755E-05,5.79157E-05,4.98678E-05,0.000208582,0.000173013,0.00030228,0.000414875,0.000768761,0.001090653,0.002646823,0.003896144,0.003187257,0.007752724,0.015264715,0.031267337,0.01238704,0.005187959,0.002972394,0.005571766,0.006723043,0.002297628,0.001100057,0.000613667,0.00026112,0.000171344,8.39661E-05,0.000138097,3.47217E-05,8.50286E-05]
# err_trans_dict['80_3']=[3.78943E-05,0.000252062,2.54992E-05,5.68874E-06,2.01402E-06,1.61358E-05,0.000341383,7.9266E-06,0.0003521,0.000950149,0.001101943,0.001713168,0.001506518,0.005437606,0.008316287,0.000570866,0.001495292,0.003864163,0.003456765,0.000418954,8.5024E-05,0.001476916,0.00092068,0.000840626,0.000558721,0.000299901,0.000298088,1.87033E-05,4.58161E-06,1.52289E-05]
# err_trans_dict['9100_2']=[0,5.91531E-05,1.31187E-05,0.000148551,0.000265521,0.000245724,0.000127656,0.000278402,0.000279354,0.000540538,0.000518253,0.002128359,0.000754352,0.000670519,0.00101711,0.00114894,0.000496039,0.000177329,0.000242063,0.001177205,0.00092068,0.000840626,0.000558721,0.000299901,0.000298088,1.87033E-05,4.58161E-06,1.52289E-05,2.56759E-06,0,6.31601E-05,4.60344E-05,3.158E-05,1.44544E-05,1.44544E-05]
# xy average
err_trans_dict['50_6']=[6.41633E-05,6.41633E-05,6.76425E-05,0.0000562486,0.001094619,0.002634837,0.003785677,0.009267697,0.031315922,0.008135959,0.003715839,0.002190533,0.0001474232,0.00062098,0.000198104,2.49544E-05,5.49302E-05]
err_trans_dict['60_3']=[5.79157E-05,4.98678E-05,0.000208582,0.000173013,0.00030228,0.000414875,0.000768761,0.001090653,0.002646823,0.003896144,0.003187257,0.007752724,0.015264715,0.031267337,0.01238704,0.005187959,0.002972394,0.0005571766,0.0006723043,0.0002297628,0.0001100057,0.0000613667,8.39661E-05]
err_trans_dict['80_3']=[2.01402E-06,1.61358E-05,0.000341383,7.9266E-06,0.0003521,0.000950149,0.001101943,0.001713168,0.001506518,0.005437606,0.008316287,0.000570866,0.001495292,0.003864163,0.003456765,0.000418954,8.5024E-05,0.001476916,0.00092068,0.000840626,0.000558721,0.000299901]
err_trans_dict['9100_2']=[0.000265521,0.000245724,0.000127656,0.000278402,0.000279354,0.000540538,0.000518253,0.002128359,0.000754352,0.000670519,0.00101711,0.00114894,0.000496039,0.000177329,0.000242063,0.001177205,0.00092068,0.000840626,0.000558721,0.000299901,0.000298088,1.87033E-05,0.000458161,0.000652289,2.56759E-06,0]

err_rot_dict = {}
err_rot_dict['50_6']=[6.60919E-05,0.012950076,0.022579436,0.009968916,4.05926E-05]
err_rot_dict['60_3']=[5.77943E-05,0.001167186,0.005412882,0.01504157,0.015474087,0.008228788,0.014999849,0.061884904,0.011667102,0.009491892,0.008521982,0.002394878,0.002199991,1.75099E-05,0.00005686356]
err_rot_dict['80_3']=[9.54012E-06,5.77943E-05,0.000312563,0.000311714,0.001596329,0.007503571,0.015354618,0.016447834,0.013646895,0.009716606,0.008696631,0.007454373,0.004237557,0.001120306,0.002357844,0.00492629,0.008487855,0.008066856,0.00444568,0.001024031,6.63804E-05,2.78901E-05,0]
err_rot_dict['9100_2']=[0.000131425,0.000526797,0.000286562,0.00107052,0.001678555,0.004038951,0.001629849,0.001547752,0.002116063,0.000678825,0.000730878,0.000219633,0.000812737,0.002178601,0.001380464,0.000224061,0.000876903,0.000803041,0.000217329,0.000544053,0.000234287,0.000270826,0.001089327,0.00089617,6.04887E-05,0]

err_trans_energy_dict = {}
err_trans_energy_dict['50_6']= [0.004055314,0.003722701,0.000729181,0.000292482,0.000121682,9.18813E-05,1.35951E-06,0,0]
err_trans_energy_dict['60_3']= [0.047276766,0.021306414,0.01205779,0.005780361,0.003668206,0.001932219,0.001052976,0.000615676,0.000351888,0.000213297,0.000244095,0.000216225,8.60844E-05,5.691E-05,3.42467E-05,1.63992E-05,8.65701E-06]
err_trans_energy_dict['80_3']= [0.000677318,0.001989101,0.001095207,0.000776508,4.9234E-06,0.000270315,0.000230647,0.000151558,7.90301E-06,6.79588E-06,3.74303E-05,3.94443E-05,4.40495E-06,5.28358E-05,1.65127E-05,2.54992E-05,1.28379E-06,1.28379E-06]
    #[0.080943054,0.003369317,0.013300414,0.015408005,0.014650296,0.010666091,0.008100496,0.005394406,0.004252422,0.002927033,0.002474117,0.001861418,0.00143055,0.000738074,0.000804658,0.000525039,0.000375759,0.000243761]
err_trans_energy_dict['9100_2']= [0.003774631,0.001897414,0.000659963,0.003365507,0.000495893,0.001479011,0.000286984,0.001091085,0.000497651,0.00019678,3.31536E-05,0.000154991,5.38104E-05,0.000215718,1.84613E-05,4.33631E-05,4.73701E-05,0.000136768,1.71257E-04,1.44544E-04,0.71531E-04,0.31531E-04,1.44544E-05]

err_rot_energy_dict = {}
err_rot_energy_dict['50_6']= [0]
err_rot_energy_dict['60_3']= 0.2*np.array([0.058596604,0.007314032,0.009097928,0.009376256,0.010718058,0.010142613,0.009311247,0.006831826,0.004233948,0.002641268,0.002121639,0.001145794])
err_rot_energy_dict['80_3']= 0.2*np.array([0.032566476,0.014602398,0.013195248,0.01098392,0.008981212,0.00602125,0.002099258,0.001392805,0.004399054,0.005693942,0.007858105,0.008456212,0.009846367,0.009030973,0.008516843,0.007296462,0.006101219,0.005610767,0.004200908,0.003406696,0.002289303,0.001382227,0.001029522,0.000794974,0.000459236,0.000299188])
err_rot_energy_dict['9100_2']= [0.003940275,0.001221374,9.46861E-05,0.001898533,5.80012E-05,0.001354502,0.001290482,0.00011278,0.000733825,0.000531464,0.002733723,0.000565477,0.001095422,0.002738589,0.000157954,0.001145809,0.000630595,0.000376619,4.57045E-05,6.97465E-05,0.00015671,3.73066E-05,0.00037133,0.000775193,0.000231032,9.23067E-05,0.000720284,0.000170781,0.000428526,0.000206368,3.62374E-04,0.000327345,0.000103852]


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
    pdf_energy_dict = {}
    delta_dict = {}
    label = []

    x_0 = [0.041, 0.048, 0]  # central peak 两侧的最小值
    area_per = [0]
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


        deltatheta = dtheta  #弧度
        deltar = (dx+dy)/2





# 有两种 bins=71  /31(log)，bins=31(点平滑,但是太少了,作图时需要/2来与71结果一致)，拟合用bins=71（点多，略凌乱，获得的结果的概率接近TL）
        # translation-------------
        # [-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5]
        # [x*0.06+0.03 for x in range(-39,40)]
        au, bu, cu = plt.hist(deltar,[x*0.06+0.03 for x in range(-39,40)],histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar 31
        # au, bu, cu = plt.hist((deltar-np.mean(deltar)) / (np.sqrt(2 * np.sum(deltar ** 2) / len(deltar))), histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        au /= len(deltar)
        au_ind = np.where(au == 0)
        au = np.delete(au,au_ind)
        au1 = np.array([math.log(x/2) for x in au])
        bu = (bu[:-1]+bu[1:])/2.
        bu = np.delete(bu,au_ind)
        pdf_trans_dict[filename_1[i]+'x'] = bu  # 存入dict
        pdf_trans_dict[filename_1[i] + 'y'] = au1 #- np.min(au1)

        # rotation-----------------------
        AU, BU, CU = plt.hist(deltatheta,[x*0.026+0.013 for x in range(-49,50)], histtype='bar', facecolor='blue', alpha=0.75)#, rwidth=0.2)  # , density=True) # 无归一系数
        # AU, BU, CU = plt.hist((deltatheta-np.mean(deltatheta)) /(np.sqrt(2*np.sum(deltatheta**2)/len(deltatheta))), 71, range=(-3, 3), histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)  #归一系数，但是range很奇怪
        AU /= len(deltar)
        AU_ind = np.where(AU == 0)
        AU = np.delete(AU,AU_ind)
        AU1 = np.array([math.log(x/2) for x in AU])
        BU = (BU[:-1] + BU[1:]) / 2.
        BU = np.delete(BU,AU_ind)
        pdf_rot_dict[filename_1[i]+'x'] = BU  # 存入dict
        pdf_rot_dict[filename_1[i] + 'y'] = AU1 #- np.min(au1)


        cdf = np.cumsum(AU) / np.sum(AU)
        BU_new = np.linspace(min(BU), max(BU), 100)
        funcc = interpolate.interp1d(BU,cdf)#, kind='zero') # cubic, slinear, nearest, zero
        cdf_smooth = funcc(BU_new)
        if i>=1:
            area_per.append(cdf_smooth[np.where(BU_new == find_nearest(BU_new, -x_0[i-1]))[0][0]]+1-cdf_smooth[np.where(BU_new == find_nearest(BU_new, x_0[i-1]))[0][0]])

        # plt.figure()
        # plt.hist(deltatheta,cumulative=True, density=True) # 无归一系数 , [x*0.026+0.013 for x in range(-49,50)]
        # plt.show()



        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta
        label.append(frame_name + ' Hz')

        #trnaslation energy---------
        # [x*0.00001 for x in range(0,6)]
        aue, bue, cue = plt.hist(0.001*(dr/1000*150)**2,45, range=(0, 0.00006), histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        aue /= len(dr)
        aue_ind = np.where(aue == 0)
        aue = np.delete(aue, aue_ind)
        aue1 = np.array([math.log(x / 2) for x in aue])
        bue = (bue[:-1] + bue[1:]) / 2.
        bue = np.delete(bue, aue_ind)
        pdf_transenergy_dict[filename_1[i]+'x'] = bue  # 存入dict
        pdf_transenergy_dict[filename_1[i] + 'y'] = aue1  # - np.min(au1)

       # rotational energy -----
        # (0, 0.000003)
        #
        AUE, BUE, CUE = plt.hist(0.0000002*(dtheta*150)**2/16,35, range=(0, 0.00003), histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True) #35
        AUE /= len(dr)
        AUE_ind = np.where(AUE == 0)
        AUE = np.delete(AUE, AUE_ind)
        AUE1 = np.array([math.log(x / 2) for x in AUE])
        BUE = (BUE[:-1] + BUE[1:]) / 2.
        BUE = np.delete(BUE, AUE_ind)
        pdf_rotenergy_dict[filename_1[i]+'x'] = BUE  # 存入dict
        pdf_rotenergy_dict[filename_1[i] + 'y'] = AUE1  # - np.min(au1)

        # total energy ---
        AUTE, BUTE, CUTE = plt.hist(0.001*(dr/1000*150)**2+0.0000002*(dtheta*150)**2/16,45, range=(0, 0.00006), histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)
        AUTE /= len(dr)
        AUTE_ind = np.where(AUTE == 0)
        AUTE = np.delete(AUTE, AUTE_ind)
        AUTE1 = np.array([math.log(x / 2) for x in AUTE])
        BUTE = (BUTE[:-1] + BUTE[1:]) / 2.
        BUTE = np.delete(BUTE, AUTE_ind)
        pdf_energy_dict[filename_1[i]+'x'] = BUTE  # 存入dict
        pdf_energy_dict[filename_1[i] + 'y'] = AUTE1  # - np.min(au1)


# # percentage of active mode
#     print(area_per)
#     fig = plt.figure()
#     ax = fig.add_subplot(121)
#     ax.scatter([50,60,80,100],100*np.array(area_per))
#     ax.set_ylim(0,100)
#     ax.set_xlim(0,110)
#     ax.xaxis.set_major_locator(MultipleLocator(20))
#     ax.xaxis.set_minor_locator(MultipleLocator(10))
#     ax.yaxis.set_minor_locator(MultipleLocator(10))
#     ax.tick_params(axis="x", direction="in")
#     ax.tick_params(axis="y", direction="in")
#     ax.tick_params(which='minor', direction='in')
#     fmt = '%.0f%%'
#     yticks = mtick.FormatStrFormatter(fmt)
#     ax.yaxis.set_major_formatter(yticks)
#     ax.plot([50, 110], [40, 109.93], 'black', alpha=0.3)  # y = 0.0116x - 0.1767
#     ax.plot([50,50],[0,40],'black',alpha=0.3)
#     ax.plot([0,50], [40, 40], '-', alpha=0.3)
#     # plt.plot(BU_new,cdf_smooth)
#     # ax.set_yscale('log')
#     plt.show()


    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}


    ys = [i + x + (i * x) ** 2 for i in range(len(filename_1))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    marker = ['o', 'v', 'D','^','s', 'h', '2', 'p', '*',  '+', 'x']

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    left,bottom, width,height=[0.3575,0.715,0.1,0.15]
    ax3 = fig.add_axes([left,bottom, width,height])
    expo1_y= []
    expo1_x=[]
    for i in range(len(filename_1)):#2,3):#
        ax1.scatter(pdf_trans_dict[filename_1[i] + 'x'], np.exp(pdf_trans_dict[filename_1[i] + 'y']), marker=marker[i], c='', s=25, edgecolor=colors[i], label=label[i])
        # ax1.errorbar(pdf_trans_dict[filename_1[i] + 'x'], np.exp(pdf_trans_dict[filename_1[i] + 'y']),yerr=err_trans_dict[filename_1[i]],fmt='none',elinewidth=1,ms=1,ecolor=colors[i])
        # ax1.legend()


    # ########## Translatioanl fitting 1
        middle = np.where(pdf_trans_dict[filename_1[i] + 'x'] == find_nearest(pdf_trans_dict[filename_1[i] + 'x'], 0))[0][0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
        # x = pdf_trans_dict[filename_1[i] + 'x'][middle:] # ExponentialModel()
        # y = np.clip(pdf_trans_dict[filename_1[i] + 'y'], 1e-18, None) [middle:] # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())

    # fittin the -x axis
        x1 = pdf_trans_dict[filename_1[i] + 'x'][middle:1:-1]  # ExponentialModel()
        y1 = pdf_trans_dict[filename_1[i] + 'y'][middle:1:-1]- np.min(pdf_trans_dict[filename_1[i] + 'y'])  # ExponentialModel()
        popt1, pcov1 = curve_fit(funce, -x1, y1, maxfev=10000)#,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
        # print(x1,y1,popt1,pcov1)
     # fittin the +x axis
        x2 = pdf_trans_dict[filename_1[i] + 'x'][middle:-1]  # ExponentialModel()
        y2 = pdf_trans_dict[filename_1[i] + 'y'][middle:-1]- np.min(pdf_trans_dict[filename_1[i] + 'y'])  # ExponentialModel()
        popt2, pcov2 = curve_fit(funce, x2, y2, maxfev=10000)#,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
        popt = (popt1+popt2)/2.
        pcov = (pcov1+pcov2)/2.
        # print(popt,popt1,popt2)

        xnew = np.linspace(x2.min()-0.3, x2.max()+0.3, 100) # 平滑画曲线
        y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
        print(filename[i])
        print(popt) # 振幅要*1/2（第一个系数）
        print(pcov)
        ax1.plot(-xnew[::-1], np.exp(y11+np.min(pdf_trans_dict[filename_1[i] + 'y']))[::-1], color=colors[i],alpha=0.6)
        ax1.plot(xnew, np.exp(y11+np.min(pdf_trans_dict[filename_1[i] + 'y'])), color=colors[i],alpha=0.6)

        expo1_y.append(popt[2])
        expo1_x.append(int(label[i][:-2]))



    # # # GaussianModel()------------
    #     x = pdf_trans_dict[filename_1[i] + 'x']
    #     y = pdf_trans_dict[filename_1[i] + 'y']-np.min(pdf_trans_dict[filename_1[i] + 'y'])
    #     mod = Pearson7Model()  # Pearson7Model() #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #ExponentialModel() #GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #
    #     pars = mod.guess(y, x=x)
    #     out = mod.fit(y, pars, x=x)
    #     # print(out.fit_report(min_correl=0.25))
    #     ax1.plot(x, np.exp(out.best_fit+np.min(pdf_trans_dict[filename_1[i] + 'y'])),':',color=colors[i],alpha=0.5)
    #     # ax1.legend()
    #     # ax1.legend(loc='best')

    # expo1_x = [6.0, 8.7, 15.5, 24.0]
    # ax3.set_xlim(5,25)
    # ax3.xaxis.set_major_locator(MultipleLocator(5))
    # ax3.xaxis.set_minor_locator(MultipleLocator(5))
    # ax3.set_xlabel(r'$\Gamma$',fontsize=9)
    ax3.xaxis.set_major_locator(MultipleLocator(20))
    ax3.xaxis.set_minor_locator(MultipleLocator(10))
    ax3.set_xlim(40,110)
    ax3.set_xlabel(r'$f$ (Hz)',fontsize=9)
    ax3.scatter(expo1_x,expo1_y,s=11)
    ax3.yaxis.set_major_locator(MultipleLocator(1))
    ax3.set_ylim(1,3.5)
    ax3.set_ylabel(r'$\beta$',fontsize=9)
    ax3.xaxis.set_label_coords(0.65, -0.35)
    ax3.yaxis.set_label_coords(-0.15,0.5)
    ax3.tick_params(axis="x",direction="in", labelsize=9)
    ax3.tick_params(which='minor', direction='in')
    ax3.tick_params(axis="y",direction="in", labelsize=9)


    # plt.legend(label)
    leg1 = ax1.legend(loc='upper left')
    # leg = plt.legend(label,loc='bottom')
    leg1.get_frame().set_linewidth(0.0)

    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(which='minor', direction='in')

    # ax1.set_title('PDF' + ' [0.6mm] ', fontsize=10)
    ax1.set_xlabel('$\Delta x$ (mm)') #('$\Delta x(pixel)$')  #
    ax1.set_ylabel('$P(\Delta x)$')
    # ax1.set_xlim(-2.4, 2.4)
    ax1.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    ax1.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)

    ax2 = fig.add_subplot(122)

    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'theta'], norm_hist=True,bins=200, kde=True, hist=False, label=filename_1[i].split('_', 1)[0] + 'Hz')  #'mode'+str(i+1))#
        # ax2.scatter(pdf_rot_dict[filename_1[i] + 'x'], np.exp(pdf_rot_dict[filename_1[i] + 'y']), alpha=0.75, color=colors[i], cmap='hsv', label=label[i])
        ax2.scatter(pdf_rot_dict[filename_1[i] + 'x'], np.exp(pdf_rot_dict[filename_1[i] + 'y']), marker=marker[i],  c='', s=25, edgecolor=colors[i], cmap='hsv', label=label[i])
        ax2.errorbar(pdf_rot_dict[filename_1[i] + 'x'], np.exp(pdf_rot_dict[filename_1[i] + 'y']),yerr=err_rot_dict[filename_1[i]],fmt='none',elinewidth=1,ms=1,ecolor=colors[i])
        # ax2.scatter(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y']-np.min(pdf_rot_dict[filename_1[i] + 'y']), alpha=0.75,color=colors[i], label=label[i])


# method 1     整体fit -------------------------------------
        x = pdf_rot_dict[filename_1[i] + 'x']*100.0
        y = pdf_rot_dict[filename_1[i] + 'y']-np.min(pdf_rot_dict[filename_1[i] + 'y'])

        if int(filename_1[i].split('_', 1)[0]) == 50:
            popt, pcov = curve_fit(funcer0, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min(), x.max(), 10000)  # 平滑画曲线
            y11 = [funcer0(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

        elif int(filename_1[i].split('_', 1)[0]) == 60:
            popt, pcov = curve_fit(funcer1, x, y, maxfev=100000)#,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min(), x.max(), 10000) # 平滑画曲线
            y11 = [funcer1(p, popt[0], popt[1], popt[2], popt[3],popt[4]) for p in xnew]
            print(filename[i])
        elif int(filename_1[i].split('_', 1)[0]) == 80:
            popt, pcov = curve_fit(funcer2, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min(), x.max(), 10000)  # 平滑画曲线
            y11 = [funcer2(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

            x_80_1 = np.arange(-0.35,0.2,0.01)*100.0
            y_80_1 = [6.9 * np.exp(-(x_i + 13) ** 2 / 13 ** 2) for x_i in x_80_1 ]
            ax2.scatter(x_80_1/100.0, np.exp(y_80_1 + np.min(pdf_rot_dict[filename_1[i] + 'y'])),color='orange', alpha=0.4,s=3)
            x_80_2 = np.arange(-0.2,0.35,0.01)*100.0
            y_80_2 = [6.9 * np.exp(-(x_i - 13) ** 2 / 13 ** 2) for x_i in x_80_2 ]
            ax2.scatter(x_80_2/100.0, np.exp(y_80_2 + np.min(pdf_rot_dict[filename_1[i] + 'y'])),color='orange', alpha=0.4,s=3)
            x_80_3 = np.arange(-0.15,0.15,0.001)*100.0
            y_80_3 = [1.6 ** 2 * np.exp(-x_i ** 2 / 3.4** 2) for x_i in x_80_3 ]
            ax2.plot(x_80_3/100.0, np.exp(y_80_3 + np.min(pdf_rot_dict[filename_1[i] + 'y'])),'--',color='orange', alpha=0.75)


        elif int(filename_1[i].split('_', 1)[0]) == 9100:
            popt, pcov = curve_fit(funcer3, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-0.2, x.max()+0.2, 10000)  # 平滑画曲线
            y11 = [funcer3(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

        ax2.plot(xnew / 100.0, np.exp(y11 + np.min(pdf_rot_dict[filename_1[i] + 'y'])), color=colors[i],alpha=0.6)

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
#             popt1, pcov1 = curve_fit(funcer4, -x1, y1, maxfev=1000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
#         # print(x1,y1,popt1,pcov1)
#         # fittin the +x axis
#         x2 = pdf_rot_dict[filename_1[i] + 'x'][middle:-1]  # ExponentialModel()
#         y2 = pdf_rot_dict[filename_1[i] + 'y'][middle:-1] - np.min(pdf_rot_dict[filename_1[i] + 'y'])  # ExponentialModel()
#         if int(filename_1[i].split('_', 1)[0]) == 50:
#             popt1, pcov1 = curve_fit(funce, x2, y2, maxfev=10000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
#         else:
#             popt2, pcov2 = curve_fit(funcer4, x2, y2, maxfev=10000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
#         popt = (popt1 + popt2) / 2.
#         pcov = (pcov1 + pcov2) / 2.
#         # print(popt,popt1,popt2)
#
#         xnew = np.linspace(x2.min(), x2.max(), 100)  # 平滑画曲线
#         y11 = [funcer4(p, popt[0], popt[1], popt[2],popt[3], popt[4]) for p in xnew]
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

    ax4 = fig.add_axes([left+0.434,bottom, width,height])
    expo2_x= [50,60,80,100]
    ax4.set_xlabel(r'$f$ (Hz)', fontsize=9)
    ax4.xaxis.set_major_locator(MultipleLocator(20))
    ax4.xaxis.set_minor_locator(MultipleLocator(10))
    ax4.set_xlim(40,110)
    # expo2_x = [6.0, 8.7, 15.5, 24.0]
    # ax4.set_xlabel(r'$\Gamma$', fontsize=9)
    # ax4.xaxis.set_major_locator(MultipleLocator(5))
    # ax4.xaxis.set_minor_locator(MultipleLocator(5))
    # ax4.set_xlim(5,25)
    expo2_y= [2.21E-05,0.079,0.13,0.22]
    ax4.scatter(expo2_x,expo2_y,s=11)
    ax4.yaxis.set_major_locator(MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.set_ylim(-0.05,0.3)
    ax4.set_ylabel(r'$\Delta \Theta_0$ (rad)',fontsize=9)
    ax4.xaxis.set_label_coords(0.65, -0.35)
    ax4.yaxis.set_label_coords(-0.29,0.5)
    ax4.tick_params(axis="x",direction="in", labelsize=9)
    ax4.tick_params(which='minor', direction='in')
    ax4.tick_params(axis="y",direction="in", labelsize=9)



    ax5 = fig.add_axes([left+0.2155,bottom-0.001, width,height])
    ax5.scatter([50, 60, 80, 100], 100*np.array(area_per), s=11)  # [50,60,80,100][6.0, 8.7, 15.5,24.0]
    ax5.set_xlim(40,110)
    ax5.xaxis.set_major_locator(MultipleLocator(20))
    ax5.xaxis.set_minor_locator(MultipleLocator(10))
    ax5.set_xlabel(r'$f$ (Hz)',fontsize=9)
    print(area_per)
    # ax5.scatter([6.0, 8.7, 15.5,24.0], 100*np.array(area_per), s=11)  # [50,60,80,100][6.0, 8.7, 15.5,24.0]
    # ax5.set_xlim(5,25)
    # ax5.xaxis.set_major_locator(MultipleLocator(5))
    # ax5.xaxis.set_minor_locator(MultipleLocator(5))
    # ax5.set_xlabel(r'$\Gamma$',fontsize=9)
    # ax5.set_ylim(0,1)
    ax5.yaxis.tick_right()
    ax5.yaxis.set_minor_locator(MultipleLocator(10))
    ax5.yaxis.set_major_locator(MultipleLocator(50))
    ax5.tick_params(axis="x", direction="in")
    ax5.tick_params(axis="y", direction="in")
    ax5.tick_params(which='minor', direction='in')
    ax5.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax5.set_ylabel(r'Active mode',fontsize=9)
    ax5.xaxis.set_label_coords(0.35, -0.32)
    ax5.yaxis.set_label_coords(1.4,0.5)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)




    ax2.set_xlabel('$\Delta \Theta$ (rad)')#('$\Delta theta (degree)$')  #
    ax2.set_ylabel('$P(\Delta \Theta )$')
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.tick_params(axis="x", direction="in")
    ax2.tick_params(which='minor', direction='in')
    ax2.tick_params(axis="y", direction="in")

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.set_ylim((0.00005, 5))
    ax1.set_xlim((-1.05,1.05))
    ax2.set_ylim((0.00005, 5))
    ax2.set_xlim((-0.4,0.4))
    # leg = plt.legend(label,loc='bottom center')
    # leg.get_frame().set_linewidth(0.0)



    # arr_lena = mpimg.imread('C:\\Users\\guan\\Desktop\\report_d\\Template for submissions to Scientific Reports (4)\\insert3_1.PNG')
    # imagebox = OffsetImage(arr_lena, zoom=0.3)
    # ab = AnnotationBbox(imagebox, (0.6,0.08), xycoords='data', boxcoords='offset points', frameon=False)
    # ax1.add_artist(ab)
    plt.subplots_adjust(wspace=0.27, hspace=0,bottom=0.22)
    # plt.draw()

    ax2.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    ax2.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)
    # ax1.set_title('(a)', loc='left', fontstyle='normal',fontweight=800)
    # ax2.set_title('(b)', loc='left', fontstyle='normal',fontweight=800)
    # plt.savefig('C:\\Users\\guan\\Desktop\\report_d\\Template for submissions to Scientific Reports (4)\\fig3_1.eps',  dpi=500,format='eps')
    plt.show()


#----energy distribution
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for i in range(len(filename_1)):#2,3):#
        ax1.scatter(pdf_transenergy_dict[filename_1[i] + 'x']*1000,np.exp(pdf_transenergy_dict[filename_1[i] + 'y']), marker=marker[i],s=25,c='',edgecolors=colors[i], label=label[i])
        ax1.errorbar(pdf_transenergy_dict[filename_1[i] + 'x']*1000, np.exp(pdf_transenergy_dict[filename_1[i] + 'y']), yerr=err_trans_energy_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i])

        ########## Translatioanl fitting 1 # function fitting
        if float(filename[i].split('_', 1)[0]) == 9100:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][:-1]*1000  # ExponentialModel()
            y = pdf_transenergy_dict[filename_1[i] + 'y'][:-1]-np.min(pdf_transenergy_dict[filename_1[i] + 'y']) # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
        elif float(filename[i].split('_', 1)[0]) == 80:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][:-2] * 1000  # ExponentialModel()
            y = pdf_transenergy_dict[filename_1[i] + 'y'][:-2] - np.min(pdf_transenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
        elif float(filename[i].split('_', 1)[0]) == 60:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][:-3] * 1000  # ExponentialModel()
            y = pdf_transenergy_dict[filename_1[i] + 'y'][:-3] - np.min(pdf_transenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
        else:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][:-1] * 1000  # ExponentialModel()
            y = pdf_transenergy_dict[filename_1[i] + 'y'][:-1] - np.min(pdf_transenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())

        popt, pcov = curve_fit(funce, x, y, maxfev=10000)

        xnew = np.linspace(x.min(), np.array(pdf_transenergy_dict[filename_1[i] + 'x']*1000).max(), 100)  # 平滑画曲线
        y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
        print(filename[i])
        print(popt)  # 振幅要*1/2（第一个系数）
        print(pcov)


        ax1.plot(xnew, np.exp(y11+np.min(pdf_transenergy_dict[filename_1[i] + 'y'])), color=colors[i], alpha=0.6)


    left,bottom, width,height=[0.245,0.71,0.08,0.15]
    ax4 = fig.add_axes([left,bottom, width,height])
    expo4_y= [1.10423094,1.21964384,1.51685809,1.76550091]
    expo4_x=[50,60,80,100]
    ax4.scatter(expo4_x,expo4_y,s=11)
    ax4.plot([40,50,60,80,100,110],[0.975,1.1,1.235,1.485,1.735,1.86],':',alpha=0.6)
    ax4.xaxis.set_major_locator(MultipleLocator(20))
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.xaxis.set_minor_locator(MultipleLocator(10))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax4.set_xlim(40,110)
    ax4.set_ylim(1,2)
    ax4.set_xlabel(r'$f$ (Hz)',fontsize=9)
    ax4.set_ylabel(r'$\beta$',fontsize=9)
    ax4.xaxis.set_label_coords(0.5, -0.35)
    ax4.yaxis.set_label_coords(-0.25,0.5)
    ax4.tick_params(axis="x", direction="in", labelsize=9)
    ax4.tick_params(which='minor',direction='in')
    ax4.tick_params(axis="y", direction="in", labelsize=9)


    # # # ax1.legend(label)
    # leg = ax1.legend(label,loc='center right')
    # leg.get_frame().set_linewidth(0.0)

    ax1.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(which='minor', direction='in')
    ax1.tick_params(axis="y", direction="in")

    ax1.set_xlabel('$E_t$ (mJ)')
    ax1.set_ylabel('$P(E_t)$')
    ax1.set_ylim((0.00005, 1))
    ax1.set_xlim((-0.0002,0.034))




    for i in range(len(filename_1)):  # 2,3):#
        ax2.scatter(pdf_rotenergy_dict[filename_1[i] + 'x']*1000, np.exp(pdf_rotenergy_dict[filename_1[i] + 'y']), marker=marker[i],s=25,c='',edgecolors=colors[i], label=label[i])
        ax2.errorbar(pdf_rotenergy_dict[filename_1[i] + 'x'] * 1000, np.exp(pdf_rotenergy_dict[filename_1[i] + 'y']),yerr=err_rot_energy_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i],label=label[i])

# ########## Rotational fitting 1
#
# # # # ExponentialModel()---------
# #         mod = ExponentialModel()
# #         pars = mod.guess(y, x=x)
# #         out = mod.fit(y, pars, x=x)
# #         # print(out.fit_report(min_correl=0.25))
# #         ax2.plot(x, out.best_fit / 2, color=colors[i], alpha=0.5)
# #         # ax2.legend(loc='best')
#
########## Translatioanl fitting 1 # function fitting
        if float(filename[i].split('_', 1)[0]) == 50:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'] * 100000  # ExponentialModel()
            y = pdf_rotenergy_dict[filename_1[i] + 'y'] - np.min(pdf_rotenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
            ax2.plot(x, np.exp(y + np.min(pdf_rotenergy_dict[filename_1[i] + 'y'])), color=colors[i], alpha=0.5)
        elif float(filename[i].split('_', 1)[0]) == 80:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'] * 100000  # ExponentialModel()
            y = pdf_rotenergy_dict[filename_1[i] + 'y'] - np.min(pdf_rotenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
        elif float(filename[i].split('_', 1)[0]) == 60:
            x = pdf_rotenergy_dict[filename_1[i] + 'x']* 100000  # ExponentialModel()
            y = pdf_rotenergy_dict[filename_1[i] + 'y']- np.min(pdf_rotenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())
        else:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'] * 100000  # ExponentialModel()
            y = pdf_rotenergy_dict[filename_1[i] + 'y'] - np.min(pdf_rotenergy_dict[filename_1[i] + 'y'])  # ExponentialModel()  ## # xlogy(np.sign(pdf_trans_dict[filename_1[i] + 'y']), pdf_trans_dict[filename_1[i] + 'y']) / np.log(math.e())

        if float(filename[i].split('_', 1)[0])!= 50:
            popt, pcov = curve_fit(funce, x, y,p0=[5,1,2],maxfev=10000)

            xnew = np.linspace(x.min(), x.max(), 100)  # 平滑画曲线  np.array(pdf_rotenergy_dict[filename_1[i] + 'x'] * 1000)
            y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
            print(filename[i],'----',popt,pcov)

            ax2.plot(xnew*0.01, np.exp(y11 + np.min(pdf_rotenergy_dict[filename_1[i] + 'y'])), color=colors[i],alpha=0.6)


# # GaussianModel()------------
#         mod = GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #ExponentialModel() #GaussianModel()  #PseudoVoigtModel()  #SplitLorentzianModel()  # SkewedVoigtModel()  #VoigtModel()  #LorentzianModel()  #
#         pars = mod.guess(y, x=x)
#         out = mod.fit(y, pars, x=x)
#         # print(out.fit_report(min_correl=0.25))
#         ax2.plot(x, out.best_fit/2,color=colors[i],alpha=0.5)
#         # ax2.legend(loc='best')

    left,bottom, width,height=[0.53,0.71,0.08,0.15]
    ax5 = fig.add_axes([left,bottom, width,height])
    expo5_y= [2.53007354,3.38405904,5.96788378e+00]
    expo5_x=[60,80,100]
    ax5.scatter(expo5_x,expo5_y,s=11)
    # ax5.plot([40,50,60,80,100,110],[0.975,1.1,1.235,1.485,1.735,1.86],':',alpha=0.6)
    ax5.xaxis.set_major_locator(MultipleLocator(20))
    ax5.xaxis.set_minor_locator(MultipleLocator(10))
    ax5.yaxis.set_major_locator(MultipleLocator(2))
    ax5.yaxis.set_minor_locator(MultipleLocator(1))
    ax5.set_xlim(50,110)
    ax5.set_ylim(1,7)
    ax5.set_xlabel(r'$f$ (Hz)',fontsize=9)
    ax5.set_ylabel(r'$\beta$',fontsize=9)
    ax5.xaxis.set_label_coords(0.5, -0.35)
    ax5.yaxis.set_label_coords(-0.15,0.5)
    ax5.tick_params(axis="x", direction="in", labelsize=9)
    ax5.tick_params(which='minor', direction='in')
    ax5.tick_params(axis="y", direction="in", labelsize=9)


    for i in range(len(filename_1)):#2,3):#
        # ax3.plot(pdf_energy_dict[filename_1[i] + 'x'] * 1000,np.exp(pdf_energy_dict[filename_1[i] + 'y']) + np.exp(pdf_energy_dict[filename_1[i] + 'y']),alpha=0.3, color=colors[i])
        ax3.scatter(pdf_energy_dict[filename_1[i] + 'x']*1000,np.exp(pdf_energy_dict[filename_1[i] + 'y'])+np.exp(pdf_energy_dict[filename_1[i] + 'y']), marker=marker[i],s=25,c='',edgecolors=colors[i], label=label[i])



    leg = ax3.legend(label,loc='upper right')
    leg.get_frame().set_linewidth(0.0)

    ax2.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax2.tick_params(axis="x", direction="in")
    ax2.tick_params(which='minor', direction='in')
    ax2.tick_params(axis="y", direction="in")
    ax3.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax3.tick_params(axis="x", direction="in")
    ax3.tick_params(which='minor', direction='in')
    ax3.tick_params(axis="y", direction="in")
    ax2.set_xlabel('$E_r$ (mJ)')
    ax2.set_ylabel('$P(E_r)$')
    ax3.set_xlabel('$E$ (mJ)')
    ax3.set_ylabel('$P(E)$')
    ax2.set_ylim((0.00005, 1))
    ax2.set_xlim((-0.0002,0.027))
    ax3.set_ylim((0.00005, 1))
    ax3.set_xlim((-0.0002,0.04))
    # ax1.set_title('(a)', loc='left', fontstyle='normal',fontweight=800)
    # ax2.set_title('(b)', loc='left', fontstyle='normal',fontweight=800)
    plt.subplots_adjust(wspace=0.37, hspace=0, bottom=0.22)
    # ---------------------------------------------------------------------------
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')





    # ax1.set_ylim((0.0005, 1))
    # ax2.set_ylim((0.0005, 1))

    plt.show()