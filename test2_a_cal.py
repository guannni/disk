# 读取hdf文件，计算pdf
# 出D:\guan2019\1_disk\a\下 按a分类的数据的pdf图像
import tables as tb
import pandas as pd
import trackpy as tp
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import os.path
import matplotlib.mlab as mlab
import seaborn as sns
import scipy.signal as signal

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


# 把pdf的delta数据存出去
path2 = 'D:\\guan2019\\1_disk\\a_full\\'
filename = [name for name in os.listdir(path2)]  # fre 文件夹
file_n = [path2 + name + '\\data\\' for name in filename]  # 每个fre下原始data的路径
file_delta = [path2 + name + '\\analysis\\pdf\\data\\' for name in filename]  # 每个fre下pdf——delta的存储路径
print(filename, file_n, file_delta)


for p in range(len(file_n)):  # TODO:5g——range(4,len(file_n))
    path_1 = file_n[p]
    filename_1 = [os.path.splitext(name)[0] for name in os.listdir(path_1)]  # 每个fre下文件名
    file_n1 = [file_n[p] + name + '.h5' for name in filename_1]  # 每个fre下pdf_delta的存储文件
    file_s1 = [file_delta[p] + name + '.txt' for name in filename_1]  # 每个fre下pdf_delta的存储文件
    file_s2 = [file_delta[p] + name + '.jpg' for name in filename_1]  # 每个fre下pdf_delta的存储文件

    counts = np.zeros((len(file_n1)))

    pdf_trans_dict = {}
    pdf_rot_dict = {}
    delta_dict = {}
    for i in range(len(file_n1)):
        store = pd.HDFStore(file_n1[i], mode='r')
        print(store.keys())
        center_1 = store.get('center').values  # numpy array
        theta_1 = store.get('theta').values
        store.close()
        n1 = len(theta_1)


        # todo : 改变delta间隔------------------
        # 这里改了要在最后图片标题maxtime改一下！
        step = 3
        center = center_1[::step]
        theta = theta_1[::step]
        # -------------------------------------



        N = len(theta)
        max_time = N / fps  # seconds

        deltax = center[1:, 0] - center[:-1, 0]  # numpy array
        deltay = center[1:, 1] - center[:-1, 1]
        r = np.sqrt(center[:, 0] ** 2 + center[:, 1] ** 2)
        deltar = r[1:] - r[:-1]
        index = []
        for k in range(len(deltar)):
            if abs(deltar[k]) > 2.3:  # 把明显由于识别错误产生的零星数据删掉
                index.append(k)
        dr = np.delete(deltar, index)
        print(np.min(dr), np.max(dr))
        print(len(r), len(dr))

        x = center[:, 0]  # numpy array
        y = center[:, 1]
        THETA = theta.reshape(len(center))#np.zeros(shape=(len(center), 3))
        time = np.linspace(0, max_time, N)
        deltatheta = THETA[1:] - THETA[:-1]
        index = []
        for k in range(len(deltatheta)):
            if deltatheta[k] > 130:  # 处理周期行导致的大deltatheta
                deltatheta[k] -= 180
            elif deltatheta[k] < -130:
                deltatheta[k] += 180
            # if abs(deltatheta[k]) > 32:  # 把明显由于识别错误产生的零星数据删掉
            #     index.append(k)
        dtheta = np.delete(deltatheta, index)
        dr = np.delete(deltar, index)
        # print(np.min(dtheta), np.max(dtheta))
        # print(len(THETA), len(dtheta))

        deltatheta = dtheta
        deltar = dr

        weights_r = np.ones_like(deltar) / float(len(deltar))
        au, bu, cu = plt.hist(deltar, int(80./3*max(3, max(deltar) - min(deltar))), histtype='bar', facecolor='yellowgreen', weights=weights_r, alpha=0.75, rwidth=0.5)# au是counts，bu是deltar
        pdf_trans_dict[filename_1[i]+'y'] = au
        # pdf_trans_dict_bar[filename_1[i] + 'x'] = bu
        bu1 = (bu[:-1]+bu[1:])/2.
        pdf_trans_dict[filename_1[i]+'x'] = bu1  # 存入dict

        weights_theta = np.ones_like(deltatheta) / float(len(deltatheta))
        AU, BU, CU = plt.hist(deltatheta, int(80./30*max(30, max(deltatheta) - min(deltatheta))), histtype='bar', facecolor='blue', weights=weights_theta, alpha=0.75, rwidth=0.5)
        pdf_rot_dict[filename_1[i] + 'x'] = AU
        # pdf_rot_dict_bar[filename_1[i] + 'x'] = BU
        BU1 = (BU[:-1]+BU[1:])/2.
        pdf_rot_dict[filename_1[i] + 'y'] = BU1  # 存入dict

        delta_dict[filename_1[i]+'r'] = deltar
        delta_dict[filename_1[i]+'theta'] = deltatheta

    pdf_rot_dict_new = {}
    pdf_trans_dict_new = {}


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    # label = []
    for i in range(len(filename_1)):
        sns.distplot(delta_dict[filename_1[i] + 'r'], bins=100, kde=True, hist=False,
                     label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.scatter(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], alpha=0.5, s=3)
        #
        # plt.plot(pdf_trans_dict[filename_1[i] + 'x'], pdf_trans_dict[filename_1[i] + 'y'], linewidth=2, alpha=0.5)
        # label += list([filename_1[i].split('_', 1)[0] + 'Hz'])
    # plt.legend(label)
    ax1.set_title('translational PDF'+ ' [' + filename[p]+'g, ' + str(round(max_time*step)) + '$s$]',fontsize=10)
    ax1.set_xlabel('$\Delta R(pixel)$')
    ax1.set_ylabel('P')
    # ax1.set_xlim(-2.4, 2.4)
    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)
    print(delta_dict[filename_1[i] + 'r'][signal.argrelextrema(x, np.greater)])
    print(signal.argrelextrema(x, np.less))



    ax2 = fig.add_subplot(122)
    label = []
    for i in range(len(filename_1)):
        sns.distplot(delta_dict[filename_1[i] + 'theta'], bins=100, kde=True, hist=False,
                     label=filename_1[i].split('_', 1)[0] + 'Hz')
        # plt.scatter(pdf_rot_dict[filename_1[i] + 'y'], pdf_rot_dict[filename_1[i] + 'x'], alpha=0.5, s=3)
        # plt.plot(pdf_rot_dict[filename_1[i] + 'y'], pdf_rot_dict[filename_1[i] + 'x'], linewidth=2, alpha=0.5)
        # label += list([filename_1[i].split('_', 1)[0] + 'Hz'])
    # plt.legend(label)
    ax2.set_title('rotational PDF' + ' [' + filename[p] + 'g, ' + str(round(max_time*step)) + '$s$]', fontsize=10)
    ax2.set_xlabel('$\Delta theta (degree)$')
    ax2.set_ylabel('P')
    # ax2.set_xlim(-17, 17)
    plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)
    plt.show()

