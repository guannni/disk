# 读取hdf文件(autocorre)
#  average

# 球心平动 'G:\\ball_new\\3_analysis\\msd\\trans\\'

# import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
from scipy.ndimage import filters
from matplotlib.ticker  import MultipleLocator
from matplotlib.ticker import FuncFormatter

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

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(np.logspace(-2, 2, 5))
    y_vals = intercept + slope * x_vals
    plt.plot(np.logspace(-2, 2, 5), np.logspace(-2, 2, 5), '--')

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



# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 1
FRE = 60  # todo eachF [50,85,5]没有80，手动排着输
# ACC = 5 # todo eachA [3,5,0.5]
# 
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\inactive\\'  # 60Hz
path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\total\\'  # 60Hz
# path2 = 'D:\\guan2019\\1_disk\\a_full\\5\\total\\'  # 5g 
# path2 = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\active_all0\\'  # 5g all0
# path2 = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\inactive_select0\\'  # 5g select

filename = [name for name in os.listdir(path2)]
fig = plt.figure()
ax = fig.add_subplot(111)
label = []
ys = [m + (m) ** 2 for m in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
marker = ['o', 'v', 'D','^','s', 'h', '2', 'p', '*',  '+', 'x']

for j in range(len(filename)):  #3,4):    #
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3) if name.endswith('.h5')] # 只取.h5文件
    file_n = [path3 + str(name) + '.h5' for name in filename1]
    print(filename1, file_n)

    autocorr_all = []
    tau = []
    lt = 0



    for i in range(len(file_n)):  #0,1):#
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
        # frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
        # print(frame_name)
        # if len(frame_name) > 3:
        #     frame_name = frame_name[1:]

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

        v_t = dr #dy#(dx+dy)/2 # /(step/fps)  # 平动delta r
        # v_t = dtheta #/(step/fps)  # 转动delta theta


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
        # plt.plot(time[0:len(ac)], ac/ac[0],marker=marker[i],markerfacecolor="None", markersize=5, color=colors[i], label=label[i])# color=colors[i])  # auto correlation
        # plt.scatter(time[0:len(ac)], ac/ac[0],alpha=0.65,marker=marker[i],c='', s=25, edgecolor=colors[i], label=label[i])  # auto correlation



        CORR = ac/ac[0]
        TAU = time[0:len(ac)]
        if i == 0 :
            tau = TAU
        print(len(TAU),len(tau))
        if len(TAU)<len(tau):
            tau = TAU
        print(len(tau))


        if i == 0 :
            CORR_all = CORR
            lt = len(CORR)
        elif i == 1 :
            CORR_1 = CORR_all
            print(CORR_1.shape[0])
            CORR_all = np.zeros((i + 1, max(CORR_1.shape[0],len(CORR))))
            CORR_all[0,:CORR_1.shape[0]] = CORR_1
            CORR_all[1,:len(CORR)] = CORR
            lt = min(lt,len(CORR))

        elif i > 1:
            CORR_1 = CORR_all
            CORR_all = np.zeros((i+1,max(CORR_1.shape[1],len(CORR))))
            CORR_all[0:i,:CORR_1.shape[1]] = CORR_1
            CORR_all[i,:len(CORR)] = CORR
            lt = min(lt, len(CORR))

    if len(file_n)>1:
        CORR_m = CORR_all.mean(axis=0)[:lt]
    else:
        CORR_m = CORR_all

    label.append(str(filename[j])+' g')

    a = np.logspace(-3,3,100)
    a_index = []
    TAU_new = []
    CORR_new = []
    for i in a:
        ind = find_nearest(tau, i)
        a_index.append(ind)
        TAU_new.append(tau[ind])#TAU[ind])
        CORR_new.append(CORR_m[ind])# MSD[ind])


    # plt.plot(TAU_new, CORR_new,'o', markerfacecolor='none', alpha=0.75, color=colors[j], label=label[j])
    plt.plot(TAU_new, CORR_new,marker=marker[j],markerfacecolor="None", markersize=5, color=colors[j], label=label[j])
    # ax.scatter(TAU_new, MSD_new, alpha=0.75,marker=marker[j], c='', s=25, edgecolor=colors[j])#, label=label[i])


ax.set_xlabel('$t$ (s)') #('lags')#

time = np.around(np.arange(0, (len(v_t) ) * 1 / 150, 1 / 150), decimals=3)
# plt.xticks(np.arange(len(v_t)), time)

# # # rotational
# ax.set_ylabel('$C_{\Delta \Theta}(t)$')
# # ax.set_xlim((0.005, 17))
# # ax.set_ylim((-0.5, 1.2))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.set_xscale('log')

# # translational
ax.set_ylabel('$C_{\Delta r}(t)$')
# ax.set_xlim((-0.01, 0.065)) # tlc
# ax.xaxis.set_major_locator(MultipleLocator(0.02))
# ax.xaxis.set_minor_locator(MultipleLocator(0.01))
# ax.set_xlim((-0.01, 0.5)) # 60Hz
# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.set_xlim((-0.01, 0.25)) # 5g
ax.set_ylim((-0.5, 1.2))
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
# ax.set_ylim((-0.6,1.15))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.legend(label)
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')

plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)

plt.show()

print(label)