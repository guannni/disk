# 输出dr, dtheta in tianli's format
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 3
FRE = 60  # todo eachF [50,85,5]没有80，手动排着输
# ACC = 5  # todo eachA [3,5,0.5]

# -----------------------------------------------

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
                print(h[1] )
                list.append(h[1])
        return list


# path2 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\data\\'
path2 = 'D:\\guan2019\\1_disk\\TLc\\all\\data\\'
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

# path3 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis_delta2\\'
path3 = 'D:\\guan2019\\1_disk\\TLc\\all\\analysis_delta2\\'
frame_posi = [path3 + 'posi\\v\\' + name + '.jpg' for name in filename]  # 速度路径


print([i for i in range(len((1, 2, 3)))])

for i in range(len(file_n)):
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    center = store.get('center').values  # numpy array
    theta = store.get('theta').values
    store.close()

    N = len(theta)
    max_time = N / fps  # seconds
    frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    print(type(frame_name))

    traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'x': center[:, 0], 'y': center[:, 1]})

    # # trans
    x = center[:, 0]  # numpy array
    dx = x[step::1] - x[:-step:1]
    y = center[:, 1]
    dy = y[step::1] - y[:-step:1]
    dr = np.sqrt(dx**2+dy**2)

    THETA = theta.reshape(len(center))
    dtheta = THETA[step::1] - THETA[:-step:1]
    index = []
    for k in range(len(dtheta)):
        if dtheta[k] > 130:  # 处理周期行导致的大deltatheta
            dtheta[k] -= 180
        elif dtheta[k] < -130:
            dtheta[k] += 180
        if abs(dtheta[k]) > 30:  # 把明显由于识别错误产生的零星数据删掉
            index.append(k)
    dtheta = np.delete(dtheta, index)
    dr = np.delete(dr, index)
    time = np.linspace(0, (N-len(index)-step)/fps*step, N-len(index)-step)
    print(' [' + str(frame_name) + '$Hz$, ')
    print(0.001*(dr.mean() / 480 * 0.26/(step/fps))**2, 0.0005*(dtheta.mean() *math.pi/180/(step/fps)*0.005)**2)

    # flatten fluctuation-----------------------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6))
    ax0 = plt.subplot(211)
    plt.plot(time, dr / 480 * 260, color='yellowgreen')
    ax0.set_xlabel('time(s)')
    ax0.set_ylabel('dr(mm)')
    ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    plt.title('Fluctuation ' + ' [' + str(frame_name) + '$Hz$, ' + str(round((N-len(index)-step)/fps*step)) + '$s$]',fontsize=10)  # todo eachF
    # plt.title('Fluctuation ' + ' [' + str(frame_name) + '$Hz$, ' + str(ACC) + '$g$, ' + str(round(max_time)) + '$s$]',fontsize=10)  # todo eachA

    ax1 = plt.subplot(212)
    plt.plot(time, dtheta*math.pi/180)
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('dtheta(rad)')
    ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段

    plt.show()
    # fig.savefig(frame_posi_f[i])


