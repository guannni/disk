# 读取hdf文件，计算MSD
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
                print(h[1])
                list.append(h[1])
        return list


# read position from hdf5 file
# path2 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\data\\'
# path2 = 'D:\\guan2019\\1_disk\\TLc\\all\\data\\'
path2 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\pdf\\70_originaldata\\'
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

# path3 = 'D:\\guan2019\\1_disk\\TLc\\tianli_f_all\\0.6\\analysis_delta2\\msd_rot\\'
# path3 = 'D:\\guan2019\\1_disk\\TLc\\all\\analysis_delta2\\msd_rot\\'
# path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis_delta2\\msd_rot\\70\\'
path3 = 'D:\\guan2019\\1_disk\\TLc\\all_1\\analysis\\msd_rot\\msd_rot1\\'
frame_msd = [path3 + name + '.h5' for name in filename]
frame_msd_pic = [path3 + name + '.jpg' for name in filename]


print(frame_msd)

for i in range(3,4):#len(file_n)):
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    center = store.get('center').values[:,0]  # numpy array
    theta = store.get('theta').values
    store.close()
    print(len(center))

    # todo : 改变delta间隔------------------
    step = 1
    # --------------------------------------
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

    theta_1 = THETA[::step]/180*np.pi
    # -------------------------------------

    N = len(theta_1)
    max_time = N / fps * step  # seconds
    frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'x': theta_1})
    print(type(frame_name))

    # msd
    dt = max_time/N
    msd = compute_msd(traj, t_step=dt, coords=['x'])
    print(msd.head())
    ax = msd.plot(x="tau", y="msds", logx=True, logy=True, legend=False, title='MSD')
    ax.fill_between(msd['tau'], msd['msds'] - msd['msds_std'], msd['msds'] + msd['msds_std'], alpha=0.2)
    ax.plot()
    # plt.show()
    msd_i = pd.HDFStore(frame_msd[i], complib='blosc')
    msd_i.append(frame_name, msd, format='t', data_columns=True)
    fig = ax.get_figure()
    fig.savefig(frame_msd_pic[i])
    del msd, msd_i, ax, fig

