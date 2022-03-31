# 读取hdf文件，计算POSI, TRAJACTORY (PDF这个是错的，MSD是对的）
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import matplotlib.mlab as mlab
import seaborn as sns

# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 3
AMP = 0.6

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


# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\data\\'
path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\data\\'
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

# path3 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis_delta2\\'
path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\'
frame_traj = [path3 + 'traj\\' + name + '.jpg' for name in filename]  #
frame_traj_mag = [path3 + 'traj\\' + name + '_m.jpg' for name in filename]  #
frame_msd = [path3 + 'msd\\' + name + '.h5' for name in filename]
frame_msd_pic = [path3 + 'msd\\' + name + '.jpg' for name in filename]
frame_pdf = [path3 + 'pdf\\' + name + '.h5' for name in filename]
frame_pdf_pic = [path3 + 'pdf\\' + name + '.jpg' for name in filename]
frame_posi = [path3 + 'posi\\' + name + '.jpg' for name in filename]
frame_posi_f = [path3 + 'posi\\' + name + '_flat.jpg' for name in filename]  # '.._flat.jpg'为展开theta的position图
# 位置和角度从_copy文件夹直接读
# frame_posi_trans = [path3 + 'posi_trans\\' + name + '.h5' for name in filename]
# frame_posi_rot = [path3 + 'posi_rot\\' + name + '.h5' for name in filename]

print(frame_traj, frame_msd)
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

    # # traj pic
    traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'x': center[:, 0], 'y': center[:, 1]})
    print(traj.head())
    ax = traj.plot(x='x', y='y',alpha=0.6, linewidth=1, legend=False)
    plt.title('Trajectory '+' [0.6mm, ' + str(frame_name) +'$Hz$, ' + str(round(max_time)) +'$s$]', fontsize=10)

    plt.xticks(np.arange(0,520,40))
    plt.yticks(np.arange(0,520,40))
    plt.text(1.5, 1, '1 pixel=0.54 mm', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    ax.set_xlim(0, 480)  # (traj['x'].min()-10, traj['x'].max()+10)
    ax.set_ylim(0, 480)  # (traj['y'].min()-10, traj['y'].max()+10)
    ax.set_xlabel('x(pixel)')
    ax.set_ylabel('y(pixel)')
    ax.plot()
    # plt.show()
    fig = ax.get_figure()
    fig.savefig(frame_traj[i])

    ax = traj.plot(x='x', y='y', alpha=0.6, legend=False)
    plt.title('Trajectory '+ '[0.6mm, ' + str(frame_name) +'$Hz$, ' + str(round(max_time)) +'$s$]', fontsize=10)
    plt.text(1, -0.6, '1 pixel=0.54 mm', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    plt.axis('equal')
    ax.set_xlabel('x(pixel)')
    ax.set_ylabel('y(pixel)')
    ax.plot()
    # plt.show()
    fig = ax.get_figure()
    fig.savefig(frame_traj_mag[i])
    del ax,fig

    # # trans
    x = center[:-step:step, 0]  # numpy array
    dx = x[step::step] - x[:-step:step]
    y = center[:-step:step, 1]
    dy = y[step::step] - y[:-step:step]
    dr = np.sqrt(dx**2+dy**2)
    THETA = np.zeros(shape=(len(center), 3))
    THETA[:, 0] = theta.reshape(len(center)) - 180
    THETA[:, 1] = theta.reshape(len(center))
    THETA[:, 2] = theta.reshape(len(center)) + 180
    for k in range(len(theta)-1):
        DIS = np.array([[abs(THETA[k, 1] - THETA[k + 1, 0]), abs(THETA[k, 1] - THETA[k + 1, 1]),
                         abs(THETA[k, 1] - THETA[k + 1, 2])]])
        # THETA[k + 1, 1] = THETA[k + 1, np.argmin(DIS)]
        if np.argmin(DIS) == 0:
            THETA[k + 1:, :] -= 180
        elif np.argmin(DIS) == 2:
            THETA[k + 1:, :] += 180
    THETA = THETA[:-step:step, 1].flatten()
    index = []
    time = np.linspace(0, max_time, len(x))

    # flatten fluctuation-----------------------------------------------------------------------------------
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(12, 6))
    ax0 = plt.subplot(311)
    plt.plot(time, x, color='yellowgreen', linestyle='dotted')
    ax0.set_xlabel('time(s)')
    ax0.set_ylabel('X(pixel)')
    ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='dotted', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    plt.title('Trajectory ' + '[0.6mm, ' + str(frame_name) + '$Hz$, ' + str(round(max_time)) + '$s$]',
              fontsize=10)

    ax1 = plt.subplot(312)
    plt.plot(time, y, linestyle='dotted')
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('Y(pixel)')
    ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='dotted', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段

    ax2 = plt.subplot(313)
    print(len(THETA),len(x))
    plt.plot(time, THETA, linestyle='dotted')
    ax2.set_xlabel('time(s)')
    ax2.set_ylabel('THETA(degree)')
    ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='dotted', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    plt.subplots_adjust(top=0.9, bottom=0.1)
    # plt.show()
    fig.savefig(frame_posi_f[i])


    # [0,360]----------------------------------------------------------------------------------------------
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(12, 6))
    ax0 = plt.subplot(311)
    plt.plot(time, x, color='yellowgreen', linestyle='dotted')
    ax0.set_xlabel('time(s)')
    ax0.set_ylabel('X(pixel)')
    ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='dotted', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    plt.title('Trajectory ' + '[0.6mm, ' + str(frame_name) + '$Hz$, ' + str(round(max_time)) + '$s$]',
              fontsize=10)

    ax1 = plt.subplot(312)
    plt.plot(time, y, linestyle='dotted')
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('Y(pixel)')
    ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='dotted', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段

    ax2 = plt.subplot(313)
    THETA %= 360
    plt.plot(time, THETA, linestyle='dotted')
    ax2.set_xlabel('time(s)')
    ax2.set_ylabel('THETA(degree)')
    ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='dotted', color='0.75')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    # plt.subplots_adjust(top=0.9, bottom=0.1)
    # plt.show()
    fig.savefig(frame_posi[i])

