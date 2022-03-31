# 读取hdf文件，计算v
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import matplotlib.mlab as mlab
import seaborn as sns

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step =  1
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
                print(h[1] )
                list.append(h[1])
        return list


# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\data\\'#_new\\'  # TODO: eachF
# path2 = 'D:\\guan2019\\1_disk\\a\\'+str(ACC)+'\\data_new\\'  # TODO: eachA

path2 = 'D:\\guan2019\\1_disk\\TLc\\all\\data\\'  #_1\\analysis_delta2\\pdf\\TL_originaldata\\'  # tianli  #
# path2 = 'D:\\guan2019\\1_disk\\f\\60Hz\\data\\'  # 60Hz
# path2 = 'D:\\guan2019\\1_disk\\a\\5\\data\\'  # 5g

filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

# path3 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\analysis\\'  # TODO: eachF
# path3 = 'D:\\guan2019\\1_disk\\a\\'+str(ACC)+'\\analysis_delta2\\'  # TODO: eachA
# frame_posi = [path3 + 'posi\\v\\' + name + '.jpg' for name in filename]  # 速度路径

energy_t = []
energy_r = []
for i in range(len(file_n)):
    store = pd.HDFStore(file_n[i], mode='r')
    # print(store.keys())
    center = store.get('center').values  # numpy array
    theta = store.get('theta').values
    store.close()

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
        if abs(deltatheta[k]) > 20 or abs(deltax[k]) > 3 or abs(deltay[k]) > 3 :  # 把明显由于识别错误产生的零星数据删掉
            index.append(k)

    deltax = np.delete(deltax, index)
    deltay = np.delete(deltay, index)
    deltar = np.delete(deltar, index)
    deltatheta = np.delete(deltatheta, index)

    dtheta = deltatheta / 180 * math.pi
    dx = deltax / 480 * 260
    dy = deltay / 480 * 260
    dr = deltar / 480 * 260

    time = np.around(np.arange(0, len(dtheta) * 1/150, 1/150), decimals=2)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # plt.scatter(time,dtheta,marker='.',s=0.5,alpha=0.3)
    # ax1.set_xlabel('time(s)')
    # ax1.set_ylabel('dtheta(rad)')
    # ax2 = fig.add_subplot(212)
    # plt.scatter(time,dx,marker='.',s=0.5,alpha=0.3)
    # ax2.set_xlabel('time(s)')
    # ax2.set_ylabel('dx(mm)')
    # plt.show()


    # print('max dx',max(dx),'max dtheta',max(dtheta))
    v2 = np.square(dr*0.001/(step/fps)).mean()
    w2 = np.square(dtheta/(step/fps)).mean()
    # print(w2)
    print(v2/2*0.0028, (w2*0.0028*0.005**2/4))

    energy_t.append(v2 / 2 * 0.0028*1e6)
    energy_r.append(w2/4*0.0028*0.005**2*1e6)

x = filename
ax = plt.subplot(111)



# plt.bar(x, np.array(energy_t)/(np.array(energy_t)+np.array(energy_r)), color='green', label='y1')
plt.bar(x, energy_t, label='translational')
plt.bar(x, energy_r, bottom=energy_t, label='rotational')
plt.title('Energy Distribution', fontsize=10)
for i in range(len(x)):
    plt.text(i-0.28, 0.05, "%.1f " % (100*energy_t[i]/(energy_t[i]+energy_r[i])) + '%', color='w')
plt.legend()
ax.set_xlabel('f(Hz)')
ax.set_ylabel('E(uJ)')

plt.show()