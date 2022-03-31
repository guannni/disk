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
from matplotlib.ticker import MultipleLocator
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
    return (np.abs(a)*np.exp(-np.abs(b)*x**np.abs(c)))

def funcer0(x, a, b,c, d,e): # rotational 整体fit
    # return a**2* np.exp(-(x+b)**2 / 8**2) +a**2 * np.exp(-(x-b)**2 / 8**2) +0.9 * np.exp(-x**2 / 2**2)
    return 0.063**2* np.exp(-(x+9.3)**2 / 7.2**2) +0.063**2 * np.exp(-(x-9.3)**2 / 7.2**2) +0.9 * np.exp(-x**2 / e**2)  # 3 Gaussian 2 * np.exp(-(x+0.1 )**2 / c**2) +2 * np.exp(-(x-0.1 )**2 / c**2) +
def funcer1(x, a, b,c, d,e): # rotational 整体fit
    return 0.204**2 * np.exp(-(x - 8.4)**2 /6.18**2) + 0.204**2* np.exp(-(x + 8.4)**2 / 6.18**2) + d ** 2 *np.exp(-x**2 / 1.801**2)  # 3 Gaussian
def funcer2(x, a, b, c, d, e):  # rotational 整体fit
    return 0.3**2 * np.exp(-(x - 8.4) ** 2 / 5.9** 2) +0.3**2 * np.exp(-(x + 8.4) ** 2 /5.9** 2) + 0.13  * np.exp(-x ** 2 / 3** 2)  # 3 Gaussian
def funcer3(x, a, b, c, d, e):  # rotational 整体fit
    return a**2 * np.exp(-(x - 8.9) ** 2 / 6.7** 2) + a**2 * np.exp(-(x + 8.9) ** 2 / 6.7** 2) +0.12 * np.exp(-x ** 2 / 2.2** 2)  # 3 Gaussian

def frac_reduc(n,m):
    n = int(n)
    m = int(m)
    for i in range(2, n):
        while (n % i == 0 and m % i == 0):
            n = n // i
            m = m // i
    return (n,m)

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
path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\data\\'  # 60Hz
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]  # fre 文件夹
file_n = [path2 + name +'.h5' for name in filename]  # 每个fre下原始data的路径
file_delta = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis\\pdf\\'  # \tianli_f_all\每个fre下pdf——delta的存储路径
print(filename, file_n, file_delta)


err_trans_dict = {}
err_trans_dict['3.5_0']=[0,	0.000191905,0.000329888,0.000364172	,0.000558638,0.000142885,0.000826359,0.001685453,0.004432394,0.002105049,0.023319422,0.00212039,0.002856216,0.002214522,0.001008502,0.00143813,0.00083085,0.00195995,0.0000355338,0.000043715,0.00083085,0.00095995,0.0000355338,0.000043715]
err_trans_dict['4.0_1']=[	0.0000171459,	0.0000293191	,0.0000315532,	0.000361162,	0.000589424,0.0009112175	,0.0009872102,	0.002156579	,0.0030920027,	0.0025066153,	0.00490682618,	0.0046179804	,0.0037799709,	0.0026208911	,0.004613756	,0.003832046,	0.001487754,	0.001269521,	0.0008063877,	0.000466977,	0.000143173,	5.07501E-05]
err_trans_dict['4.5_0']=[	0.0000113311,	0.0000418351	,0.0001779947	,0.000220808	,0.0003342615	,0.0005756073,	0.0009956977,	0.00083596333	,0.0087210951	,0.0082548273,	0.0081922626	,0.0076525527	,0.0013923648	,0.0013378765	,0.0010293837	,0.0011135577,	0.001146065,	0.0008172688,	0.001100165,	0.0009419355	,0.000902162,	0.000547211,	0.000145064]
err_trans_dict['5.0_3']=[1.93068E-05	,9.93068E-05,	9.93068E-05	,9.93068E-05,	6.93068E-04,	5.93068E-04	,1.93068E-03,	1.93068E-03	,1.93068E-03	,1.93068E-03,	1.93068E-03,	1.93068E-02	,1.93068E-02,	1.93068E-02,	1.93068E-02	,1.93068E-03,	1.93068E-03	,1.93068E-03,	1.93068E-03,	5.93068E-04	,1.93068E-04	,1.93068E-04	,1.93068E-05,	1.93068E-05	,1.93068E-05,	1.93068E-05]#[1.93068E-05,	2.79033E-05,	2.00416E-05	,3.64049E-05,	4.64619E-05,	1.03646E-05	,5.03822E-05	,0.000164732	,0.000304205	,0.001056519,	0.0024153,	0.005005281,	0.009754874	,0.018695484,	0.028609289,	0.038746935	,0.039405337	,0.030833667	,0.016081313	,0.02732266	,0.036084084	,0.037883752,	0.031568576,	0.022771378,	0.01435797,	0.008444092,	0.005065643	,0.002990728	,0.001984661,	0.001181254	,0.000787675,	0.000370781	,0.000264128	,8.55665E-05,	7.62715E-05,	1.88289E-05	,1.859E-05,	1.93068E-05,	1.07286E-05	,2.90797E-05]


err_rot_dict = {}
err_rot_dict['3.5_0']=[4.03915E-05,	0.000205346,	0.000422369,	0.0005273013,	0.0007705745,	0.0010532396,	0.00094967826,	0.0011632137,	0.0028598373,	0.0269479773	,0.00295312,	0.0008985954	,0.0010775049	,0.0008628379	,0.0014007773	,0.0009473026	,0.0009462128	,0.000386908	,9.20863E-05,0]
err_rot_dict['4.0_1']=[4.03915E-05,	0.000205346,	0.000422369,	0.001273013,	0.005705745,	0.010532396,	0.014967826,	0.017632137,	0.028598373,	0.0269479773	,0.0295312,	0.008985954	,0.010775049	,0.008628379	,0.006007773	,0.003473026	,0.001462128	,0.000386908	,9.20863E-05]
err_rot_dict['4.5_0']=[0.000386926,	0.003129407,	0.01107847,	0.015875026	,0.008104365,	0.000527854	,0.009556399,	0.004429341	,0.057499722,	0.007159627,	0.01051032,	0.008705251	,0.012044375,	0.016445992,	0.020195098	,0.011661372	,0.002684825	,0.000177555]
err_rot_dict['5.0_3']=[3.622E-08,	9.46685E-05,	0.001831157,	0.00626493,	0.010301045,	0.006446761,	0.009142572,	0.009810801,	0.007377384,	0.004363645,	0.028790512,	0.0077747,	0.009894652,	0.004876543,	0.008394273	,0.007692261	,0.004319464	,0.003186258	,0.001440024,	0.000162307,0]

err_trans_energy_dict = {}
err_trans_energy_dict['3.5_0']=[0.011159609	,0.002620465,	0.003147667,	0.002093799,	0.001353292,	0.000818909,	0.000217161,	0.000175515	,0.00017791	,0.000178981	,0.000118505,	9.55953E-05,	7.27511E-05,	8.34077E-05	,8.47806E-05	,8.51021E-05	,7.32684E-05	,6.18218E-05	,5.05679E-05]
err_trans_energy_dict['4.0_1']=[0.053145207	,0.032912551	,0.011262642	,0.004944968,	0.00117161,	0.000875515	,0.00052791	,0.000218981	,0.000218505,	5.55953E-05,	5.27511E-05,	6.34077E-05	,3.47806E-05	,3.51021E-05	,2.32684E-05	,1.18218E-05	,1.05679E-05]
err_trans_energy_dict['4.5_0']=[0.277754243	,0.002412341	,0.001678803,	3.38193E-05	,0.000888148	,0.000854592,	0.000298366	,0.000417304	,0.000325903,	0.000120838,	0.000116283	,0.000112342,	2.07317E-05,	3.9784E-05	,4.64299E-05,	2.63381E-05,	4.11275E-05,	3.92874E-05,	3.99443E-05,	6.00579E-05	,1.58472E-05,	2.13425E-05,	2.13425E-05	,0,0]
err_trans_energy_dict['5.0_3']=[0.014984042,	0.0123072,	0.004764505,	0.001325274,	0.000919792,	0.000711294,	0.000818631	,0.000795617,	0.000506092,	0.00043257	,0.000234469,	0.000247468,	0.000277757	,5.94784E-05,	0.000112852,	5.14755E-05	,6.01217E-05	,7.18742E-05	,7.74662E-05,	9.55743E-07,	2.88408E-05,	1.90679E-05,	9.29501E-06	,9.53394E-06,	4.77871E-07	,4.77871E-07]


err_rot_energy_dict = {}
err_rot_energy_dict['3.5_0']=[0.014763836,	0.0017394834	,0.0014739213,	0.001156179	,0.001082646,	0.00075353,	0.0003214043	,0.000185646,	0.0000968291,	0.000114612	,0.0000905361	,5.56298E-06]
err_rot_energy_dict['4.0_1']=[0.054763836,	0.027394834	,0.014739213,	0.008156179	,0.004082646,	0.002075353,	0.001214043	,0.000685646,	0.000368291,	0.000214612	,0.000105361	,5.56298E-05]
err_rot_energy_dict['4.5_0']=[0.030288192,	0.017390646,	0.009547397,	0.004925206,	0.004553888	,0.004643739,	0.002159115,	0.001377202,	0.000451311,	0.00021684,	6.26258E-05]
err_rot_energy_dict['5.0_3']=[0.025389526	,0.011981767,	0.006597613,	0.002215154	,0.006643302,	0.006694171,	0.005706622,0.003681291,	0.00218931,	0.000953543	,0.000373115,	0.000135858	,3.11251E-04,	5.06929E-05]

err_energy_dict = {}
err_energy_dict['3.5_0']=[0.450589197	,0.011589791,	0.002713035,	0.002845615,	0.002446888	,0.001972664,1.23E-03	,1.03E-03,	5.41E-04	,5.90E-04	,1.47E-04,	2.46E-04	,1.47E-04	,9.83E-05	,1.47E-04	,9.83E-05	,9.83E-05	,9.83E-05	,9.83E-05,	4.92E-05,	4.92E-05	,4.92E-06,4.92E-06]
err_energy_dict['4.0_1']=[0.038776762,	0.019444987,	0.011561389	,0.008073482	,0.004491201	,0.002531621,	0.001688061,	0.001706698,	0.001536979,		0.000175174,	0.000107113,	0.000119721	,4.95175E-05,	3.10881E-05,	2.13889E-05	,5.02355E-06	,3.68298E-06	,1.75575E-06]
err_energy_dict['4.5_0']=[0.037795806	,0.017263919,	0.011662734,	0.007474903	,0.002605744,	0.003619378,	0.003426069,	0.002304364,	0.001192752,	0.000452369,	0.000195963	,0.000264168,	0.00014365,	4.91094E-05,	9.39693E-05,	1.79394E-05,	2.46792E-05,	1.5097E-05,	7.21203E-05,	3.313E-05,	2.50389E-05,5.25091E-05,	5.15499E-05	,6.68552E-06	,5.36757E-06	,2.13425E-05]
err_energy_dict['5.0_3']=[0.033341502	,0.014350681,	0.013052188,	0.005505531	,0.001612455	,0.004656282	,0.004771256,	0.003022292	,0.002300134	,0.002111604,	0.000799769	,0.000491813,	5.43306E-05,	0.000178777,	0.000128646,	0.00010347	,7.40124E-05,	5.93504E-05,	9.97864E-05,	9.16798E-05,	9.65521E-05,	5.18726E-05,	5.41123E-05,	7.07404E-05	,9.53394E-06,	7.16807E-07	,2.38936E-07,	2.38936E-07]


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
    x_0 = [0.043,0.041, 0.043, 0.036]  # central peak 两侧的最小值
    area_per = []
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

        dtheta = deltatheta #/ 180 * math.pi
        dx = deltax / 480 * 260
        dy = deltay / 480 * 260
        dr = deltar / 480 * 260

        time = np.around(np.arange(0, len(dtheta) * 1 / 150, 1 / 150), decimals=2)
        deltatheta = dtheta /180.0*math.pi  #弧度
        deltar = (dx+dy)/2#/480*260


        # /np.sqrt(2*np.sum(deltar ** 2) / len(deltar)),range=(-6, 6),
        au, bu, cu = plt.hist(deltar,[x*0.06+0.03 for x in range(-39,40)], histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        au_ind = np.where(au == 0)
        au = np.delete(au,au_ind)
        # au1 = np.array([math.log(x) for x in au])
        bu = (bu[:-1]+bu[1:])/2.
        bu = np.delete(bu,au_ind)
        pdf_trans_dict[filename_1[i]+'x'] = bu  # 存入dict

        pdf_trans_dict[filename_1[i] + 'y'] = au/len(deltar) #- np.min(au1)


        # /np.sqrt(np.sum(deltatheta ** 2)/len(deltar))/0.001/1.4/10**(-23)
        AU, BU, CU = plt.hist(deltatheta,[x*0.026+0.013 for x in range(-29,30)],   histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)  /(np.sqrt(2*np.sum(deltatheta**2)/len(deltatheta)))
        AU_ind = np.where(AU == 0)
        AU = np.delete(AU, AU_ind)
        pdf_rot_dict[filename_1[i] + 'y'] = AU/len(deltar)
        BU = (BU[:-1]+BU[1:])/2.
        BU = np.delete(BU, AU_ind)
        pdf_rot_dict[filename_1[i] + 'x'] = BU  # 存入dict

        cdf = np.cumsum(AU) / np.sum(AU)
        BU_new = np.linspace(min(BU), max(BU), 100)
        funcc = interpolate.interp1d(BU,cdf)#, kind='zero') # cubic, slinear, nearest, zero
        cdf_smooth = funcc(BU_new)

        area_per.append(cdf_smooth[np.where(BU_new == find_nearest(BU_new, -x_0[i - 1]))[0][0]] + 1 - cdf_smooth[np.where(BU_new == find_nearest(BU_new, x_0[i - 1]))[0][0]])

        #energy---------
        # range=(0, 0.00006),
        aue, bue, cue = plt.hist(0.001*(dr/1000*150)**2,45, range=(0, 0.00006),  histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        aue_ind = np.where(aue == 0)
        aue = np.delete(aue, aue_ind)
        pdf_transenergy_dict[filename_1[i]+'y'] = aue/len(dr)
        bue = (bue[:-1]+bue[1:])/2.
        bue = np.delete(bue, aue_ind)
        pdf_transenergy_dict[filename_1[i]+'x'] = bue  # 存入dict

        #(0, 0.003)
        AUE, BUE, CUE = plt.hist(0.0000002*(deltatheta*150)**2/16, 45, range=(0, 0.00006), histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)
        AUE_ind = np.where(AUE == 0)
        AUE = np.delete(AUE, AUE_ind)
        pdf_rotenergy_dict[filename_1[i] + 'y'] = AUE/len(dr)
        BUE = (BUE[:-1]+BUE[1:])/2.
        BUE = np.delete(BUE, AUE_ind)
        pdf_rotenergy_dict[filename_1[i] + 'x'] = BUE  # 存入dict

        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta
        label.append(frame_name + r' g')

        # total energy ---
        AUTE, BUTE, CUTE = plt.hist(0.001*(dr/1000*150)**2+0.0000002*(deltatheta*150)**2/16,45,range=(0,0.00006), histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)
        AUTE /= len(dr)
        AUTE_ind = np.where(AUTE == 0)
        AUTE = np.delete(AUTE, AUTE_ind)
        BUTE = (BUTE[:-1] + BUTE[1:]) / 2.
        BUTE = np.delete(BUTE, AUTE_ind)
        pdf_energy_dict[filename_1[i]+'x'] = BUTE  # 存入dict
        pdf_energy_dict[filename_1[i] + 'y'] = AUTE  # - np.min(au1)
        print(filename_1[i])
        print(((dr/1000*150)**2).mean(),(0.0002*(deltatheta*150)**2/16).mean())

    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}


    ys = [i + x + (i * x) ** 2 for i in range(len(filename_1))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    marker = ['o', 'v', 'D', '^', 's', 'h', '2', 'p', '*', '+', 'x']

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    left,bottom, width,height=[0.3575,0.715,0.1,0.15]
    ax3 = fig.add_axes([left,bottom, width,height])
    expo1_y= []
    expo1_x=[]

    for i in range(len(filename_1)):#2,3):#
        ax1.scatter(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], marker=marker[i],  c='', s=25, edgecolor=colors[i], cmap='hsv', label=label[i])#, label=label[i])
        ax1.errorbar(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'],yerr=err_trans_dict[filename_1[i]],fmt='none',elinewidth=1,ms=1,ecolor=colors[i])
        middle = np.where(pdf_trans_dict[filename_1[i] + 'x'] == find_nearest(pdf_trans_dict[filename_1[i] + 'x'], 0))[0][0]  # pdf 0 的点    # x = np.delete(pdf_rot_dict[filename_1[i] + 'x'],middle-1)
        # if float(filename_1[i].split('_', 1)[0]) == 4.0:
        #     x = pdf_trans_dict[filename_1[i] + 'x'][middle+1:]
        #     y = pdf_trans_dict[filename_1[i] + 'y'][middle+1:]
        # else:
        #     x = pdf_trans_dict[filename_1[i] + 'x'][middle:]
        #     y = pdf_trans_dict[filename_1[i] + 'y'][middle:]
        x = pdf_trans_dict[filename_1[i] + 'x'][middle-1:]
        sig = err_trans_dict[filename_1[i]][middle-1:]
        y = pdf_trans_dict[filename_1[i] + 'y'][middle-1:]
        popt, pcov = curve_fit(func, x, y, maxfev=10000,sigma=1./np.array(sig), absolute_sigma=True)
        x1 = pdf_trans_dict[filename_1[i] + 'x'][middle+1::-1]
        sig = err_trans_dict[filename_1[i]][middle+1::-1]
        y1 = pdf_trans_dict[filename_1[i] + 'y'][middle+1::-1]
        popt1, pcov1 = curve_fit(func, x1, y1, maxfev=10000,sigma=1./np.array(sig), absolute_sigma=True)
        xnew = np.linspace(0, x.max()+0.3, 100) # 平滑画曲线
        y11 = [funce(p, 0.5*(popt[0]+popt1[0]), 0.5*(popt[1]+popt1[1]), 0.5*(popt[2]+popt1[2])) for p in xnew]
        ax1.plot(-xnew[::-1], np.array(y11[::-1]), color=colors[i],alpha=0.6)
        ax1.plot(xnew, np.array(y11), color=colors[i],alpha=0.6)

        expo1_y.append(0.5*(popt[2]+popt1[2]))
        # print(type(label[i][:-1]))
        expo1_x.append(float(label[i][:-1]))
    print(expo1_y,expo1_x)

    # plt.legend(label)
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)

    # plt.legend(label)
    leg1 = ax1.legend(loc='upper left')
    # leg = plt.legend(label,loc='bottom')
    leg1.get_frame().set_linewidth(0.0)
    # ax1.set_title('PDF' + ' [0.6mm] ', fontsize=10)
    ax1.set_xlabel('$\Delta x$ (mm)')  # ('$\Delta x(pixel)$')  #
    ax1.set_ylabel('$P(\Delta x)$')
    # ax1.set_xlim(-2.4, 2.4)
    ax1.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    ax1.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(which='minor', direction='in')

    ax3.xaxis.set_major_locator(MultipleLocator(1))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax3.set_xlim(3,5.5)
    ax3.set_xlabel(r'$a$ ($g$)', fontsize=9)
    ax3.scatter(expo1_x, expo1_y, s=11)
    ax3.yaxis.set_major_locator(MultipleLocator(0.5))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax3.set_ylim(0.5,2)
    ax3.set_ylabel(r'$\beta$', fontsize=9)
    ax3.xaxis.set_label_coords(0.65, -0.16)
    ax3.yaxis.set_label_coords(-0.25, 0.5)
    ax3.tick_params(axis="x", direction="in", labelsize=9,pad=1)
    ax3.tick_params(which='minor', direction='in')
    ax3.tick_params(axis="y", direction="in", labelsize=9,pad=1)




    ax2 = fig.add_subplot(122)


    for i in range(len(filename_1)):#2,3):#
        # sns.distplot(delta_dict[filename_1[i] + 'theta'], norm_hist=True,bins=100, kde=True, hist=False,label=filename_1[i].split('_', 1)[0] + 'g')  #'mode'+str(i+1))#
        ax2.scatter(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'],marker=marker[i],  c='', s=25, edgecolor=colors[i], cmap='hsv', label=label[i])
        ax2.errorbar(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'],yerr=err_rot_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i])
        x = pdf_rot_dict[filename_1[i] + 'x'] * 100.0
        y = pdf_rot_dict[filename_1[i] + 'y']

        if float(filename_1[i].split('_', 1)[0]) == 3.5:
            popt, pcov = curve_fit(funcer0, x, y, maxfev=100000,sigma=1./np.array(err_rot_dict[filename_1[i]]), absolute_sigma=True)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min(), x.max(), 10000)  # 平滑画曲线
            y11 = [funcer0(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

        elif float(filename_1[i].split('_', 1)[0]) == 4.0:
            popt, pcov = curve_fit(funcer1, x, y, maxfev=100000,sigma=1./np.array(err_rot_dict[filename_1[i]]), absolute_sigma=True)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-3, x.max()+3, 10000)  # 平滑画曲线
            y11 = [funcer1(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

            x_4_1 = np.arange(-0.25,0.2,0.01)*100.0
            y_4_1 = [0.3 ** 2 * np.exp( -(x_i + 8.4) ** 2 / 5.9 ** 2) for x_i in x_4_1 ]
            ax2.scatter(x_4_1/100.0, np.array(y_4_1)/2.0,color='DarkTurquoise', alpha=0.3,s=3)
            x_4_2 = np.arange(-0.2,0.35,0.01)*100.0
            y_4_2 = [0.3 ** 2 * np.exp(-(x_i - 8.4) ** 2 / 5.9 ** 2) for x_i in x_4_2]
            ax2.scatter(x_4_2 / 100.0, np.array(y_4_2) / 2.0, color='DarkTurquoise', alpha=0.3, s=3)
            x_4_3 = np.arange(-0.15, 0.15, 0.0025) * 100.0
            y_4_3 = [0.13 * np.exp(-x_i ** 2 / 3 ** 2)  for x_i in x_4_3]
            ax2.scatter(x_4_3 / 100.0, np.array(y_4_3) / 2.0, color='DarkTurquoise', alpha=0.4, s=3)


        elif float(filename_1[i].split('_', 1)[0]) == 4.5:
            popt, pcov = curve_fit(funcer2, x, y, maxfev=100000,sigma=1./np.array(err_rot_dict[filename_1[i]]), absolute_sigma=True)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-3, x.max()+3, 10000)  # 平滑画曲线
            y11 = [funcer2(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])
        elif float(filename_1[i].split('_', 1)[0]) == 5.0:
            popt, pcov = curve_fit(funcer3, x, y, maxfev=100000,sigma=1./np.array(err_rot_dict[filename_1[i]]), absolute_sigma=True) # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-2, x.max()+2, 10000)  # 平滑画曲线
            y11 = [funcer3(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

        ax2.plot(xnew / 100.0, np.array(y11), color=colors[i], alpha=0.6)

    ax4 = fig.add_axes([left+0.434,bottom, width,height])
    ax4.scatter([3.5, 4, 4.5, 5], np.array(area_per), s=11)
    print(area_per)
    ax4.set_xlabel(r'$a$ ($g$)', fontsize=9)
    ax4.xaxis.set_major_locator(MultipleLocator(1))
    ax4.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax4.set_xlim(3,5.5)
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.set_ylim(0,1)
    ax4.set_ylabel('fraction',fontsize=9)
    ax4.xaxis.set_label_coords(0.65, -0.16)
    ax4.yaxis.set_label_coords(-0.27,0.5)
    ax4.tick_params(axis="x",direction="in", labelsize=9,pad=1)
    ax4.tick_params(which='minor', direction='in')
    ax4.tick_params(axis="y",direction="in", labelsize=9,pad=1)


    ax5 = fig.add_axes([left+0.21,bottom-0.001, width,height])
    ax5.scatter([3.5,4,4.5,5], [0.093,0.084,0.084,0.089], s=11)
    ax5.set_xlim(3,5.5)
    ax5.xaxis.set_major_locator(MultipleLocator(1))
    ax5.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax5.set_xlabel(r'$a$ ($g$)',fontsize=9)
    ax5.yaxis.set_major_locator(MultipleLocator(0.01))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax5.set_ylim(0.075,0.1)
    ax5.yaxis.tick_right()
    ax5.set_ylabel(r'$\Delta \Theta_0$ (rad)',fontsize=8)
    ax5.xaxis.set_label_coords(0.35, -0.16)
    ax5.yaxis.set_label_coords(1.55,0.5)
    ax5.tick_params(axis="x", direction="in", labelsize=9,pad=1)
    ax5.tick_params(which='minor', direction='in')
    ax5.tick_params(axis="y", direction="in", labelsize=9,pad=1)


    ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.tick_params(axis="x", direction="in")
    ax2.tick_params(which='minor', direction='in')
    ax2.tick_params(axis="y", direction="in")

    # ax2.set_title('Rotational PDF', fontsize=10)
    ax2.set_xlabel('$\Delta \Theta$ (rad)')#('$\Delta theta (degree)$')  #
    ax2.set_ylabel('$P(\Delta \Theta )$')
    # # ---------------------------------------------------------------------------
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim((0.0006, 8))
    ax2.set_ylim((0.0006, 8))
    ax1.set_xlim((-0.75,0.75))
    ax2.set_xlim((-0.3,0.3))

    # plt.legend(label)

    # plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    ax2.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)


    plt.subplots_adjust(wspace=0.27, hspace=0, bottom=0.22)
    plt.show()
    # fig.savefig(file_s2[p])

#----energy distribution
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for i in range(len(filename_1)):#2,3):#
        ax1.scatter(pdf_transenergy_dict[filename_1[i] + 'x']*1000, np.clip(pdf_transenergy_dict[filename_1[i] + 'y'], 1e-18, None), marker=marker[i], s=25, c='', edgecolors=colors[i], label=label[i])
        ax1.errorbar(pdf_transenergy_dict[filename_1[i] + 'x'] * 1000, pdf_transenergy_dict[filename_1[i] + 'y'],yerr=err_trans_energy_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i])
        if float(filename[i].split('_', 1)[0]) == 3.5:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_transenergy_dict[filename_1[i] + 'y'][1:] * 10
        elif float(filename[i].split('_', 1)[0]) == 4.0:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_transenergy_dict[filename_1[i] + 'y'][1:] * 10
        elif float(filename[i].split('_', 1)[0]) == 4.5:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_transenergy_dict[filename_1[i] + 'y'][1:] * 10
        elif float(filename[i].split('_', 1)[0]) == 5.0:
            x = pdf_transenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_transenergy_dict[filename_1[i] + 'y'][1:] * 10
        popt, pcov = curve_fit(funce, x, y, maxfev=10000)
        xnew = np.linspace(0.0001, np.array(pdf_transenergy_dict[filename_1[i] + 'x'] * 10000).max(), 100)  # 平滑画曲线
        y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
        print(filename[i])
        print(popt)  # 振幅要*1/2（第一个系数）
        print(pcov)
        ax1.plot(xnew * 0.1, 0.1 * np.array(y11), color=colors[i], alpha=0.5)



    for i in range(len(filename_1)):  # 2,3):#
        ax2.scatter(pdf_rotenergy_dict[filename_1[i] + 'x']*1000, np.clip(pdf_rotenergy_dict[filename_1[i] + 'y'], 1e-18, None),  marker=marker[i], s=25, c='', edgecolors=colors[i], label=label[i])
        ax2.errorbar(pdf_rotenergy_dict[filename_1[i] + 'x'] * 1000, pdf_rotenergy_dict[filename_1[i] + 'y'],yerr=err_rot_energy_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i])
        if float(filename[i].split('_', 1)[0]) == 3.5:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:] *10
        elif float(filename[i].split('_', 1)[0]) == 4.0:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:]*10
        elif float(filename[i].split('_', 1)[0]) == 4.5:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:] * 10000
            y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:]*10
        elif float(filename[i].split('_', 1)[0]) == 5.0:
            x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:]* 10000
            y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:]*10
        popt, pcov = curve_fit(funce, x, y, maxfev=100000)
        xnew = np.linspace(x.min(), np.array(pdf_rotenergy_dict[filename_1[i] + 'x'] * 10000).max(), 100)  # 平滑画曲线
        y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
        ax2.plot(xnew * 0.1, np.array(y11) * 0.1, color=colors[i], alpha=0.6)
        print(filename[i])
        print(popt)  # 振幅要*1/2（第一个系数）
        print(pcov)

    for i in range(len(filename_1)):#2,3):#
        ax3.scatter(pdf_energy_dict[filename_1[i] + 'x']*1000,np.clip(pdf_energy_dict[filename_1[i] + 'y'], 1e-18, None),  marker=marker[i], s=25, c='', edgecolors=colors[i])
        ax3.errorbar(pdf_energy_dict[filename_1[i] + 'x'] * 1000, pdf_energy_dict[filename_1[i] + 'y'],yerr=err_energy_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i])
        # if float(filename[i].split('_', 1)[0]) == 3.5:
        #     x = pdf_energy_dict[filename_1[i] + 'x'][2:-3] * 10000
        #     y = pdf_energy_dict[filename_1[i] + 'y'][2:-3] *10
        # elif float(filename[i].split('_', 1)[0]) == 4.0:
        #     x = pdf_energy_dict[filename_1[i] + 'x'][2:-3] * 10000
        #     y = pdf_energy_dict[filename_1[i] + 'y'][2:-3]*10
        # elif float(filename[i].split('_', 1)[0]) == 4.5:
        #     x = pdf_energy_dict[filename_1[i] + 'x'][2:-3] * 10000
        #     y = pdf_energy_dict[filename_1[i] + 'y'][2:-3]*10
        # elif float(filename[i].split('_', 1)[0]) == 5.0:
        #     x = pdf_energy_dict[filename_1[i] + 'x'][2:-7]* 10000
        #     y = pdf_energy_dict[filename_1[i] + 'y'][2:-7]*10
        # popt, pcov = curve_fit(funce, x, y, maxfev=100000)
        # xnew = np.linspace(x.min(), np.array(pdf_energy_dict[filename_1[i] + 'x'] * 10000).max(), 100)  # 平滑画曲线
        # y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
        # ax3.plot(xnew * 0.1, np.array(y11) * 0.1, color=colors[i], alpha=0.6)
        # print(filename[i])
        # print(popt)  # 振幅要*1/2（第一个系数）
        # print(pcov)
        leg = ax3.legend(label,loc='upper right')
        leg.get_frame().set_linewidth(0.0)


    ax1.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(which='minor', direction='in')
    ax1.tick_params(axis="y", direction="in")

    ax1.set_xlabel('$E_t$ (mJ)')
    ax1.set_ylabel('$P(E_t)$')
    ax1.set_ylim((0.00005, 4))
    ax1.set_xlim((-0.0002,0.034))
    # leg = ax1.legend(label,loc='center right')
    # leg.get_frame().set_linewidth(0.0)

    ax2.xaxis.set_minor_locator(MultipleLocator(0.0025))
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
    ax2.set_ylim((0.00005, 4))
    ax2.set_xlim((-0.0002,0.02))
    ax3.set_ylim((0.00005, 4))
    ax3.set_xlim((-0.0002,0.034))
    # ax1.set_title('(a)', loc='left', fontstyle='normal',fontweight=800)
    # ax2.set_title('(b)', loc='left', fontstyle='normal',fontweight=800)
    plt.subplots_adjust(wspace=0.37, hspace=0, bottom=0.22)
    # ---------------------------------------------------------------------------
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    left, bottom, width, height = [0.245, 0.71, 0.08, 0.15]
    ax4 = fig.add_axes([left, bottom, width, height])
    expo4_y = [1.88883870e-01, 0.49120944, 0.58853435, 0.62601007]
    expo4_x = [3.5, 4, 4.5, 5]
    ax4.scatter(expo4_x, expo4_y, s=11)
    # ax4.plot([40,50,60,80,100,110],[0.975,1.1,1.235,1.485,1.735,1.86],':',alpha=0.6)
    ax4.xaxis.set_major_locator(MultipleLocator(1))
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.set_xlim(3, 5.5)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel(r'$a$ ($g$)', fontsize=9)
    ax4.set_ylabel(r'$\beta$', fontsize=9)
    ax4.xaxis.set_label_coords(0.5, -0.26)
    ax4.yaxis.set_label_coords(-0.25, 0.5)
    ax4.tick_params(axis="x", direction="in", labelsize=9)
    ax4.tick_params(which='minor', direction='in')
    ax4.tick_params(axis="y", direction="in", labelsize=9)

    left,bottom, width,height=[0.245+0.283,0.71,0.08,0.15]
    ax5 = fig.add_axes([left,bottom, width,height])
    expo5_y= [2.83102915e+00,1.31492533,1.89951679,2.00267644]
    expo5_x=[3.5,4,4.5,5]
    ax5.scatter(expo5_x,expo5_y,s=11)
    ax5.xaxis.set_major_locator(MultipleLocator(1))
    ax5.yaxis.set_major_locator(MultipleLocator(0.5))
    ax5.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax5.set_xlim(3,5.5)
    ax5.set_ylim(1,3)
    ax5.set_xlabel(r'$a$ ($g$)',fontsize=9)
    ax5.set_ylabel(r'$\beta$',fontsize=9)
    ax5.xaxis.set_label_coords(0.5, -0.26)
    ax5.yaxis.set_label_coords(-0.25,0.5)
    ax5.tick_params(axis="x", direction="in", labelsize=9)
    ax5.tick_params(which='minor',direction='in')
    ax5.tick_params(axis="y", direction="in", labelsize=9)

    # left,bottom, width,height=[0.245+0.283+0.283,0.71,0.08,0.15]
    # ax6 = fig.add_axes([left,bottom, width,height])
    # expo6_y= [1.12547645,1.48041679,1.94834752,2.25914094]
    # expo6_x=[3.5,4,4.5,5]
    # ax6.scatter(expo6_x,expo6_y,s=11)
    # ax6.plot([3.5,5],[1.12547645,2.25914094],':',alpha=0.6)
    # ax6.xaxis.set_major_locator(MultipleLocator(1))
    # ax6.yaxis.set_major_locator(MultipleLocator(0.5))
    # ax6.xaxis.set_minor_locator(MultipleLocator(0.5))
    # ax6.yaxis.set_minor_locator(MultipleLocator(0.25))
    # ax6.set_xlim(3,5.5)
    # ax6.set_ylim(1,2.5)
    # ax6.set_xlabel(r'$a$ ($g$)',fontsize=9)
    # ax6.set_ylabel(r'$\beta$',fontsize=9)
    # ax6.xaxis.set_label_coords(0.5, -0.26)
    # ax6.yaxis.set_label_coords(-0.25,0.5)
    # ax6.tick_params(axis="x", direction="in", labelsize=9)
    # ax6.tick_params(which='minor',direction='in')
    # ax6.tick_params(axis="y", direction="in", labelsize=9)



    plt.show()


