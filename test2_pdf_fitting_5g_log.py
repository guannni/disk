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
    return a**2* np.exp(-(x+8.6)**2 / 6**2) +a**2 * np.exp(-(x-8.6)**2 / 6**2) +d**2 * np.exp(-x**2 / 3**2)  # 3 Gaussian 2 * np.exp(-(x+0.1 )**2 / c**2) +2 * np.exp(-(x-0.1 )**2 / c**2) +
def funcer1(x, a, b,c, d,e): # rotational 整体fit
    return a**2 * np.exp(-(x - 9.4)**2 /6.7**2) + a**2* np.exp(-(x + 9.4)**2 / 6.7**2) + d ** 2 *np.exp(-x**2 / e**2)  # 3 Gaussian
def funcer2(x, a, b, c, d, e):  # rotational 整体fit
    return 0.3**2 * np.exp(-(x - 9.4) ** 2 / 6.5** 2) +0.3**2 * np.exp(-(x + 9.4) ** 2 /6.5** 2) + d**2 * np.exp(-x ** 2 / e** 2)  # 3 Gaussian
def funcer3(x, a, b, c, d, e):  # rotational 整体fit
    return 0.08**2 * np.exp(-(x - 0.3) ** 2 / 9.5** 2) + 0.08**2 * np.exp(-(x + 0.3) ** 2 / 9.5** 2) +0.7 * np.exp(-x ** 2 / e** 2)  # 3 Gaussian
def funcer4(x, a, b, c, d, e):  # rotational 整体fit
    return  0.07**2 * np.exp(-(x - 0.3) ** 2 / 9.5** 2) + 0.07**2 * np.exp(-(x + 0.3) ** 2 / 9.5** 2) +0.7 * np.exp(-x ** 2 / e** 2)  # 3 Gaussian


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
path2 = 'D:\\guan2019\\1_disk\\a\\5\\data1\\'  # 60Hz  data1 _copy
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]  # fre 文件夹
file_n = [path2 + name +'.h5' for name in filename]  # 每个fre下原始data的路径
file_delta = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis\\pdf\\'  # \tianli_f_all\每个fre下pdf——delta的存储路径
print(filename, file_n, file_delta)

err_trans_dict = {}
err_trans_dict['50_0']=[0,	0,	1.50268E-05	,0.000147193,	0.000456513,	0.00075783,	0.001600528,	0.003870124	,0.008441252	,0.014546108,	0.024949231	,0.023704092,	0.021642408	,0.024563981,	0.027979834,	0.022918733,	0.015703946,	0.010518343,	0.004845759	,0.002301553	,0.000834729,	0.000411258,	9.52913E-05	,8.49472E-05,	1.90782E-05,	1.00299E-05]
err_trans_dict['60_2']=[0,	0,	1.50268E-05	,0.000147193,	0.000456513,	0.00075783,	0.001600528,	0.003870124	,0.008441252	,0.014546108,	0.024949231	,0.023704092,	0.021642408	,0.024563981,	0.027979834,	0.022918733,	0.015703946,	0.010518343,	0.004845759	,0.002301553	,0.000834729,	0.000411258,	9.52913E-05	,8.49472E-05,	1.90782E-05,	1.00299E-05]
err_trans_dict['70_2']=[3.95216E-06,	0.00013891	,0.000220014	,0.000351799,	0.000171121	,0.000579317,	0.001478835,	0.000238367,	0.002191361	,0.002120037	,0.012580589	,0.002498194,	0.005835844	,0.002085073	,0.00071111,	0.000329591	,0.000260655,	0.000103165	,0.000336708,	0.000107149	,3.07434E-05	,1.52361E-05]
err_trans_dict['75_1']=[0,	2.56514E-05	,7.01518E-05	,0.000166625,	0.000205011,	0.0009350622,	0.001006834	,0.0021445839	,0.0016046731	,0.0022406929,	0.006381295,	0.109163259,	0.007674604	,0.0025091165	,0.0013220252,	0.00274557,	0.002672294	,0.000976749,	0.000371972,	0.000227582,	8.58986E-05]
err_trans_dict['85_1']=[0,	3.17133E-06	,3.17133E-06	,2.74455E-05,	0.000112347,	0.000382715,	0.000335607,	0.0008868709	,0.001708524	,0.0082923319,	0.054249074,	0.122367623,	0.049185229	,0.007940412,	0.00177588	,0.001256783,	0.00180672	,8.86195E-05	,3.61918E-05,	2.57322E-05]

err_rot_dict = {}
err_rot_dict['50_0']=[0.000375287	,0.002967105,	0.0022366825,	0.003828517	,0.0061281989,	0.008902197	,0.005409019	,0.006305676,	0.005865408	,0.003866279	,0.020575581	,0.006973399,	0.008983936,	0.002321192	,0.0208122	,0.029287477,	0.018444671	,0.003092321,0.00101442	,9.81657E-06	,0]
err_rot_dict['60_2']=[0.000075287	,0.00167105,	0.0042366825,	0.0094828517	,0.0091281989,	0.008902197	,0.005409019	,0.006305676,	0.005865408	,0.003866279	,0.020575581	,0.006973399,	0.008983936,	0.002321192	,0.0208122	,0.029287477,	0.018444671	,0.005092321,	0.000261442	,9.81657E-06	,0]
err_rot_dict['70_2']=[0	,1.45609E-05	,0.000164957,	0.001926383,	0.002546006,	0.003417222,	0.005621313,	0.003590807	,0.004010797,	0.004826182	,0.001657104,	0.046845684	,0.001689134,	0.006759145,	0.005223487,	0.004976893	,0.00372221,	0.002910603,	0.000332417,	0.000223977,	5.09681E-05]
err_rot_dict['75_1']=[2.28781E-05	,0.000136024,	0.000947244,	0.00222341,	0.0014181086	,0.0021199877	,0.0022768008,	0.0152064386,	0.0028558891	,0.0026928174,	0.002043617	,0.0009038532	,0.002008111,	0.00191091,	8.75148E-06]
err_rot_dict['85_1']=[3.19953E-05,	0.000224655,	0.000541893	,0.00116789,	0.001484133	,0.0010945	,0.003547986,	0.012387099	,0.00265393	,0.002023395,	0.002029619	,0.001434789,	0.000603174	,0.000231759,	2.03879E-05]

err_trans_energy_dict = {}
err_trans_energy_dict['50_0']=[0.014984042	,0.0153072	,0.003764505,	0.001525274,	0.0007919792	,0.000711294,	0.0008718631	,0.000795617	,0.000506092	,0.00043257	,0.000234469,	0.000247468,	0.000277757	,5.94784E-05,	0.000112852	,5.14755E-05,	6.01217E-05,	7.18742E-05,	8.72676E-05,	9.55743E-07	,2.88408E-05,	1.90679E-05	,9.29501E-06	,9.53394E-06	,4.77871E-07,	4.77871E-07,0]
err_trans_energy_dict['60_2']=[0.014984042	,0.0123072	,0.004764505,	0.001325274,	0.000919792	,0.000711294,	0.000818631	,0.000795617	,0.000506092	,0.00043257	,0.000234469,	0.000247468,	0.000277757	,5.94784E-05,	0.000112852	,5.14755E-05,	6.01217E-05,	7.18742E-05,	8.72676E-05,	9.55743E-07	,2.88408E-05,	1.90679E-05	,9.29501E-06	,9.53394E-06	,4.77871E-07,	4.77871E-07]
err_trans_energy_dict['70_2']=[0.019169381	,0.0084252	,0.004893009	,0.001291436,	0.001273355,	0.001459136,	0.000532239,	0.000563662,	0.000258746,	0.000148124	,0.000216975,	0.000205365	,0.000121816,	4.46948E-05	,3.65302E-05	,7.8805E-06,	0.000109828	,1.11103E-05,	5.34039E-05,	1.52361E-05	,1.83781E-05]
err_trans_energy_dict['75_1']=[0.0183860028	,0.0148647491,	0.0118824624,	0.0019452232,	0.0013194727,	0.0011774226	,0.000180038,	0.000155205,	0.000189035	,0.00015986	,0.000308666,	0.00014981	,7.85575E-05	,2.60536E-06,	8.75148E-06,	8.48581E-06	,8.48581E-06	,0]
err_trans_energy_dict['85_1']=[0.003761868	,0.002589137	,0.000722138	,0.000392399	,0.000265549	,0.000391067	,9.50516E-05,	3.48753E-05	,7.27207E-05	,2.69548E-05,	3.04887E-05,	6.34267E-06,	1.14698E-05	,1.85841E-05,	1.68734E-05,	0,	0,	0,	0	,0]

err_rot_energy_dict = {}
err_rot_energy_dict['50_0']=[0.023389526,	0.010981767,	0.006597613	,0.002215154	,0.006643302,	0.006694171	,0.005706622,	0.001681291,	0.00118931	,0.000353543,	0.000173115	,0.000035858	,3.11251E-05	,5.06929E-06]
err_rot_energy_dict['60_2']=[0.025389526,	0.011981767,	0.006597613	,0.002215154	,0.006643302,	0.006694171	,0.005706622,	0.003681291,	0.00218931	,0.000953543,	0.000373115	,0.000135858	,3.11251E-05	,5.06929E-05]
err_rot_energy_dict['70_2']=[0.033490588,	0.010353895,	0.006106509	,0.006579068,	0.004250737,	0.003551072,	0.002190792	,0.00023765	,0.000775799	,0.000108657,	0.000164808	,0.000419859,	2.22207E-05	,1.45609E-05,	1.19889E-05]
err_rot_energy_dict['75_1']=[0.047248573,	0.0053757316,	0.0021040337,	0.002163046,	0.000484848,	0.000104078,	1.97264E-05]
err_rot_energy_dict['85_1']=[0.007202942,	0.003694546,	0.002018086,	0.000845857,	0.000441784	,0.000267416,	6.19953E-05,	1.11957E-05]

err_energy_dict = {}
err_energy_dict['50_0']=[0.033341502	,0.014350681	,0.013052188,	0.005505531,	0.001612455,	0.004656282	,0.004771256,	0.003022292	,0.002300134	,0.002111604,	0.000799769	,0.000491813	,5.43306E-05,	0.000178777,	0.000128646	,0.00010347,	7.40124E-05,	5.93504E-05,	7.97864E-05,	9.16798E-05	,4.65521E-05,	5.18726E-05,	5.41123E-05,	2.07404E-05	,9.53394E-06	,7.16807E-07	,2.38936E-07,	2.38936E-07,	6.38926E-06,	6.38926E-06]
err_energy_dict['60_2']=[0.033341502	,0.014350681	,0.013052188,	0.005505531,	0.001612455,	0.004656282	,0.004771256,	0.003022292	,0.002300134	,0.002111604,	0.000799769	,0.000491813	,5.43306E-05,	0.000178777,	0.000128646	,0.00010347,	7.40124E-05,	0.93504E-04,	1.07864E-04,	9.16798E-05	,4.65521E-05,	5.18726E-05,	5.41123E-05,	7.07404E-05	,9.53394E-06	,7.16807E-06	,2.38936E-07,	2.38936E-07]
err_energy_dict['70_2']=[0.044683441	,0.011311988	,0.008481457,	0.006445227,	0.005282398	,0.004254573,	0.002517419	,0.002353447,	0.001971897,	0.000208224,	0.000775625,	0.000576979,	0.000322209,	0.00024386	,0.000300168	,0.000325297,	3.06191E-05,	0.000100179	,6.26965E-05,	4.31663E-05,	3.25782E-05	,3.04722E-05,	3.95216E-06]
err_energy_dict['75_1']=[0.133582307	,0.014294585	,0.007182965	,0.007427163,	0.002824264	,0.000983454,	0.000733076,	0.000330329,	0.000137527,	0.000202755,	0.000108667,	0.0000905399,	6.0029E-05,	2.23455E-05,	3.36118E-05,	2.26568E-05,	2.56514E-05,	0,	0	,0]
err_energy_dict['85_1']=[0.007943997	,0.002285777	,0.002146693,	0.001630649,	0.001064538,	0.000536525,	0.000244646,	0.000122059,	0.000196207	,9.02943E-05	,7.55129E-05,	2.75543E-05,	3.30778E-05,	2.12883E-05	,2.55461E-05,	2.42685E-05	,0,	0	,0,	0]

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
    x_0 = [0.0423,0.0406,0.0366, 0.0447, 0.0425]  # central peak 两侧的最小值
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
        deltar = (dx+dy)/2#/480*260

# -------------------------------------
        # deltar/np.sqrt(2*np.sum(deltar ** 2) / len(deltar)),range=(-6, 6),
        au, bu, cu = plt.hist(deltar ,[x*0.06+0.03 for x in range(-39,40)],  histtype='bar', facecolor='yellowgreen', alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
        au_ind = np.where(au == 0)
        au = np.delete(au,au_ind)
        # au1 = np.array([math.log(x) for x in au])
        bu = (bu[:-1]+bu[1:])/2.
        bu = np.delete(bu,au_ind)
        pdf_trans_dict[filename_1[i]+'x'] = bu  # 存入dict

        pdf_trans_dict[filename_1[i] + 'y'] = au/len(deltar) #- np.min(au1)



        # deltatheta/(np.sqrt(2*np.sum(deltatheta**2)/len(deltatheta)))
        AU, BU, CU = plt.hist(deltatheta, [x*0.026+0.013 for x in range(-29,30)], histtype='bar', facecolor='blue',  alpha=0.75, rwidth=0.2)#, density=True)
        AU_ind = np.where(AU == 0)
        AU = np.delete(AU, AU_ind)
        pdf_rot_dict[filename_1[i] + 'y'] = AU/len(deltar)
        BU = (BU[:-1]+BU[1:])/2.
        BU = np.delete(BU, AU_ind)
        pdf_rot_dict[filename_1[i] + 'x'] = BU  # 存入dict

        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta
        cdf = np.cumsum(AU) / np.sum(AU)
        BU_new = np.linspace(min(BU), max(BU), 100)
        funcc = interpolate.interp1d(BU, cdf)  # , kind='zero') # cubic, slinear, nearest, zero
        cdf_smooth = funcc(BU_new)

        area_per.append(cdf_smooth[np.where(BU_new == find_nearest(BU_new, -x_0[i - 1]))[0][0]] + 1 - cdf_smooth[np.where(BU_new == find_nearest(BU_new, x_0[i - 1]))[0][0]])

        # energy---------
        # range=(0, 0.00006),
        aue, bue, cue = plt.hist(0.001 * (dr / 1000 * 150) ** 2, 45, range=(0, 0.00006), facecolor='yellowgreen',alpha=0.75, rwidth=1)  # , density=True)  # au是counts，bu是deltar
        aue/=len(dr)
        aue_ind = np.where(aue == 0)
        aue = np.delete(aue, aue_ind)
        pdf_transenergy_dict[filename_1[i]+'y'] = aue
        bue = (bue[:-1]+bue[1:])/2.
        bue = np.delete(bue, aue_ind)
        pdf_transenergy_dict[filename_1[i]+'x'] = bue  # 存入dict

        # (0, 0.003)
        AUE, BUE, CUE = plt.hist(0.0000002 * (deltatheta * 150) ** 2 / 16,  45, range=(0, 0.00006), histtype='bar', facecolor='blue',alpha=0.75, rwidth=0.2)  # , density=True)
        AUE /= len(dr)
        AUE_ind = np.where(AUE == 0)
        AUE = np.delete(AUE, AUE_ind)
        pdf_rotenergy_dict[filename_1[i] + 'y'] = AUE
        BUE = (BUE[:-1] + BUE[1:]) / 2.
        BUE = np.delete(BUE, AUE_ind)
        pdf_rotenergy_dict[filename_1[i] + 'x'] = BUE  # 存入dict

        delta_dict[filename_1[i] + 'r'] = deltar
        delta_dict[filename_1[i] + 'theta'] = deltatheta
        label.append(frame_name + ' Hz')

        # total energy ---
        AUTE, BUTE, CUTE = plt.hist(0.001 * (dr / 1000 * 150) ** 2 + 0.0000002 * (deltatheta * 150) ** 2 / 16, 45,range=(0, 0.00006), histtype='bar', facecolor='blue', alpha=0.75, rwidth=0.2)  # , density=True)
        AUTE /= len(dr)
        AUTE_ind = np.where(AUTE == 0)
        AUTE = np.delete(AUTE, AUTE_ind)
        BUTE = (BUTE[:-1] + BUTE[1:]) / 2.
        BUTE = np.delete(BUTE, AUTE_ind)
        pdf_energy_dict[filename_1[i] + 'x'] = BUTE  # 存入dict
        pdf_energy_dict[filename_1[i] + 'y'] = AUTE  # - np.min(au1)
        print(filename_1[i])
        print(((dr / 1000 * 150) ** 2).mean(), (0.0002 * (deltatheta * 150) ** 2 / 16).mean())

    pdf_deltatheta_dict = {}
    pdf_deltar_dict = {}


    ys = [i + x + (i * x) ** 2 for i in range(len(filename_1))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    marker = ['o', 'v', 'D', '^', 's', 'h',  'p', '*', '+', 'x']

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
        x = pdf_trans_dict[filename_1[i] + 'x'][middle:]
        y = pdf_trans_dict[filename_1[i] + 'y'][middle:]
        popt, pcov = curve_fit(func, x, y, maxfev=10000)
        x1 = pdf_trans_dict[filename_1[i] + 'x'][middle+1::-1]
        y1 = pdf_trans_dict[filename_1[i] + 'y'][middle+1::-1]
        popt1, pcov1 = curve_fit(func, x1, y1, maxfev=10000)
        xnew = np.linspace(0, x.max()+0.3, 100) # 平滑画曲线
        y11 = [funce(p, 0.5*(popt[0]+popt1[0]), 0.5*(popt[1]+popt1[1]), 0.5*(popt[2]+popt1[2])) for p in xnew]
        ax1.plot(-xnew[::-1], np.array(y11[::-1]), color=colors[i],alpha=0.6)
        ax1.plot(xnew, np.array(y11), color=colors[i],alpha=0.6)

        expo1_y.append(0.5*(popt[2]+popt1[2]))
        # print(type(label[i][:-1]))
        expo1_x.append(float(label[i][:-2]))
    print(expo1_y,expo1_x)

    # plt.legend(label)
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)

    # plt.legend(label)
    leg1 = ax1.legend(loc='upper left')
    # leg = plt.legend(label,loc='bottom')
    leg1.get_frame().set_linewidth(0.0)
    ax1.set_xlabel('$\Delta x$ (mm)')  # ('$\Delta x(pixel)$')  #
    ax1.set_ylabel('$P(\Delta x)$')
    # ax1.set_xlim(-2.4, 2.4)
    ax1.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    ax1.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(which='minor', direction='in')

    ax3.xaxis.set_major_locator(MultipleLocator(15))
    ax3.xaxis.set_minor_locator(MultipleLocator(5))
    ax3.set_xlim(45,90)
    ax3.set_xlabel(r'$f$ (Hz)', fontsize=9)
    ax3.scatter(expo1_x, expo1_y, s=11)
    ax3.yaxis.set_major_locator(MultipleLocator(0.5))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax3.set_ylim(1,2)
    ax3.set_ylabel(r'$\beta$', fontsize=9)
    ax3.xaxis.set_label_coords(0.65, -0.2)
    ax3.yaxis.set_label_coords(-0.25, 0.5)
    ax3.tick_params(axis="x", direction="in", labelsize=9,pad=2)
    ax3.tick_params(which='minor', direction='in')
    ax3.tick_params(axis="y", direction="in", labelsize=9,pad=2)




    ax2 = fig.add_subplot(122)


    for i in range(len(filename_1)):#2,3):#
        ax2.scatter(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'],marker=marker[i],  c='', s=25, edgecolor=colors[i], cmap='hsv', label=label[i])
        ax2.errorbar(pdf_rot_dict[filename_1[i] + 'x'], pdf_rot_dict[filename_1[i] + 'y'],yerr=err_rot_dict[filename_1[i]], fmt='none', elinewidth=1, ms=1, ecolor=colors[i])
        x = pdf_rot_dict[filename_1[i] + 'x'] * 100.0
        y = pdf_rot_dict[filename_1[i] + 'y']

        if float(filename_1[i].split('_', 1)[0]) == 50:
            popt, pcov = curve_fit(funcer0, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min(), x.max(), 10000)  # 平滑画曲线
            y11 = [funcer0(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

            x_50_1 = np.arange(-0.25, 0.2, 0.01) * 100.0
            y_50_1 = [0.3047 ** 2 * np.exp(-(x_i + 8.6) ** 2 / 6 ** 2) for x_i in x_50_1]
            ax2.scatter(x_50_1 / 100.0, np.array(y_50_1), color='purple', alpha=0.3, s=3)
            x_50_2 = np.arange(-0.2, 0.35, 0.01) * 100.0
            y_50_2 = [0.3047 ** 2 * np.exp(-(x_i - 8.6) ** 2 / 6 ** 2) for x_i in x_50_2]
            ax2.scatter(x_50_2 / 100.0, np.array(y_50_2), color='purple', alpha=0.3, s=3)
            x_50_3 = np.arange(-0.15, 0.15, 0.0025) * 100.0
            y_50_3 = [0.325 ** 2 * np.exp(-x_i ** 2 / 3 ** 2) for x_i in x_50_3]
            ax2.scatter(x_50_3 / 100.0, np.array(y_50_3), color='purple', alpha=0.4, s=3)

        elif float(filename_1[i].split('_', 1)[0]) == 60:
            popt, pcov = curve_fit(funcer1, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-3, x.max()+3, 10000)  # 平滑画曲线
            y11 = [funcer1(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])
        elif float(filename_1[i].split('_', 1)[0]) == 70:
            popt, pcov = curve_fit(funcer2, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-3, x.max()+3, 10000)  # 平滑画曲线
            y11 = [funcer2(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])
        elif float(filename_1[i].split('_', 1)[0]) == 75:
            popt, pcov = curve_fit(funcer3, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-2, x.max()+2, 10000)  # 平滑画曲线
            y11 = [funcer3(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])
        elif float(filename_1[i].split('_', 1)[0]) == 85:
            popt, pcov = curve_fit(funcer3, x, y, maxfev=100000)  # ,bounds=([0,0,1], [np.inf, np.inf, 1.01]))
            print(popt)
            xnew = np.linspace(x.min()-2, x.max()+2, 10000)  # 平滑画曲线
            y11 = [funcer4(p, popt[0], popt[1], popt[2], popt[3], popt[4]) for p in xnew]
            print(filename[i])

        ax2.plot(xnew / 100.0, np.array(y11), color=colors[i], alpha=0.6)

    ax4 = fig.add_axes([left+0.434,bottom, width,height])
    ax4.scatter([50,60,70,75,85], np.array(area_per), s=11)
    ax4.xaxis.set_major_locator(MultipleLocator(15))
    ax4.xaxis.set_minor_locator(MultipleLocator(5))
    ax4.set_xlim(45, 90)
    ax4.set_xlabel(r'$f$ (Hz)', fontsize=9)
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.set_ylim(0,1)
    ax4.set_ylabel('fraction',fontsize=9)
    ax4.xaxis.set_label_coords(0.65, -0.2)
    ax4.yaxis.set_label_coords(-0.27,0.5)
    ax4.tick_params(axis="x",direction="in", labelsize=9,pad=2)
    ax4.tick_params(which='minor', direction='in')
    ax4.tick_params(axis="y",direction="in", labelsize=9,pad=2)


    ax5 = fig.add_axes([left+0.2155,bottom-0.001, width,height])
    ax5.scatter([50,60,70,75,85], [0.086,0.094,0.094,0.03,0.03], s=11)
    ax5.set_xlim(45, 90)
    ax5.xaxis.set_major_locator(MultipleLocator(15))
    ax5.xaxis.set_minor_locator(MultipleLocator(5))
    ax5.set_xlabel(r'$f$ (Hz)',fontsize=9)
    ax5.set_ylim(-0.02,0.12)
    ax5.yaxis.tick_right()
    ax5.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax5.yaxis.set_major_locator(MultipleLocator(0.05))
    ax5.tick_params(axis="x", direction="in",pad=2)
    ax5.tick_params(axis="y", direction="in",pad=2)
    ax5.tick_params(which='minor', direction='in')
    ax5.set_ylabel(r'$\Delta \Theta_0$ (rad)',fontsize=8)
    ax5.xaxis.set_label_coords(0.35, -0.2)
    ax5.yaxis.set_label_coords(1.555,0.5)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

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
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)


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
        if float(filename[i].split('_', 1)[0]) ==75:
            x = pdf_transenergy_dict[filename_1[i+1] + 'x'] * 10000
            y = pdf_transenergy_dict[filename_1[i+1] + 'y'] * 10
        else:
            x = pdf_transenergy_dict[filename_1[i] + 'x'] * 10000
            y = pdf_transenergy_dict[filename_1[i] + 'y'] * 10

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
        # if float(filename[i].split('_', 1)[0]) == 75:
        #     x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:-6] * 100000
        #     y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:-6] *100
        # elif float(filename[i].split('_', 1)[0]) == 85:
        #     x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:] * 100000
        #     y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:]*100
        # else:
        #     x = pdf_rotenergy_dict[filename_1[i] + 'x'] * 100000
        #     y = pdf_rotenergy_dict[filename_1[i] + 'y'] * 100
        x = pdf_rotenergy_dict[filename_1[i] + 'x'][1:] * 100000
        y = pdf_rotenergy_dict[filename_1[i] + 'y'][1:] * 100
        popt, pcov = curve_fit(funce, x, y, maxfev=100000)
        xnew = np.linspace(x.min(), np.array(pdf_rotenergy_dict[filename_1[i] + 'x'] * 100000).max(), 100)  # 平滑画曲线
        y11 = [funce(p, popt[0], popt[1], popt[2]) for p in xnew]
        ax2.plot(xnew * 0.01, np.array(y11) * 0.01, color=colors[i], alpha=0.6)
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
    expo4_y = [0.61193171,0.5736457, 0.68885616,1.56225701e-01, 1.55225701e-01]
    expo4_x = [50,60,70,75,85],
    ax4.scatter(expo4_x, expo4_y, s=11)
    # ax4.plot([40,50,60,80,100,110],[0.975,1.1,1.235,1.485,1.735,1.86],':',alpha=0.6)
    ax4.xaxis.set_major_locator(MultipleLocator(15))
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.xaxis.set_minor_locator(MultipleLocator(5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.set_xlim(45,90)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel(r'$f$ (Hz)', fontsize=9)
    ax4.set_ylabel(r'$\beta$', fontsize=9)
    ax4.xaxis.set_label_coords(0.5, -0.26)
    ax4.yaxis.set_label_coords(-0.25, 0.5)
    ax4.tick_params(axis="x", direction="in", labelsize=9)
    ax4.tick_params(which='minor', direction='in')
    ax4.tick_params(axis="y", direction="in", labelsize=9)

    left,bottom, width,height=[0.245+0.283,0.71,0.08,0.15]
    ax5 = fig.add_axes([left,bottom, width,height])
    expo5_y= [1.84827392,2.00267289,1.85182156,1.61463526,1.21470752]
    expo5_x=[50,60,70,75,85],
    ax5.scatter(expo5_x,expo5_y,s=11)
    ax5.xaxis.set_major_locator(MultipleLocator(15))
    ax5.yaxis.set_major_locator(MultipleLocator(0.5))
    ax5.xaxis.set_minor_locator(MultipleLocator(5))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax5.set_xlim(45,90)
    ax5.set_ylim(1,2.5)
    ax5.set_xlabel(r'$f$ (Hz)',fontsize=9)
    ax5.set_ylabel(r'$\beta$',fontsize=9)
    ax5.xaxis.set_label_coords(0.5, -0.26)
    ax5.yaxis.set_label_coords(-0.25,0.5)
    ax5.tick_params(axis="x", direction="in", labelsize=9)
    ax5.tick_params(which='minor',direction='in')
    ax5.tick_params(axis="y", direction="in", labelsize=9)

    plt.show()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # for i in range(len(filename_1)):#2,3):#
    #     ax1.scatter(pdf_transenergy_dict[filename_1[i] + 'x']*1000, np.clip(pdf_transenergy_dict[filename_1[i] + 'y'], 1e-18, None), alpha=0.75,color=colors[i], label=label[i])
    #
    # leg = ax1.legend(label)
    # leg.get_frame().set_linewidth(0.0)
    #
    # for i in range(len(filename_1)):  # 2,3):#
    #     ax2.scatter(pdf_rotenergy_dict[filename_1[i] + 'x']*1000, np.clip(pdf_rotenergy_dict[filename_1[i] + 'y'], 1e-18, None), alpha=0.75, color=colors[i], label=label[i])
    #
    # ax1.set_xlabel('${E_t(mJ)}$')
    # ax1.set_ylabel('${P(E_t)}$')
    # ax2.set_xlabel('${E_r(mJ)}$')
    # ax2.set_ylabel('${P(E_r)}$')
    # # ---------------------------------------------------------------------------
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    #
    #
    # plt.show()





