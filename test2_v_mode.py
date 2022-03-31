# 读取hdf文件，changepoints的确定、作图及输出
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import matplotlib.mlab as mlab
import seaborn as sns
import pywt
import ruptures as rpt  # our package
from scipy import signal
from itertools import groupby
from operator import itemgetter





# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0
step = 1
FRE = 60  # todo eachF [50,85,5]没有80，手动排着输
ACC = 5 # todo eachA [3,5,0.5]

# -----------------------------------------------
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

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
# path2 = 'D:\\guan2019\\1_disk\\f\\'+str(FRE)+'Hz\\data_copy\\'  # TODO: eachF
path2 = 'D:\\guan2019\\1_disk\\a_full\\'+str(ACC)+'\\data\\'  # TODO: eachA
# path2 = 'D:\\guan2019\\1_disk\\a_full\\'+str(ACC)+'\\mode\\inactive_all0\\70\\'  # TODO: eachA
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

# path3 = 'D:\\guan2019\\1_disk\\f\\60Hz\\mode\\'  # save active/inactive mode raw data separately
path3 = 'D:\\guan2019\\1_disk\\a_full\\5\\mode\\'  # save active/inactive mode raw data separately

for i in range(len(file_n)):
    store = pd.HDFStore(file_n[i], mode='r')
    print(file_n[i],store.keys())
    center = store.get('center').values  # numpy array
    theta = store.get('theta').values
    store.close()

    N = len(theta)
    max_time = N / fps  # seconds
    frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    print(type(frame_name))

    x = center[:, 0]  # numpy array
    dx = x[step::1] - x[:-step:1]
    y = center[:, 1]
    dy = y[step::1] - y[:-step:1]
    dr = np.sqrt(dx**2+dy**2)

    THETA = theta.reshape(len(center))
    THETA_new = THETA.copy()
    dtheta = THETA[step::1] - THETA[:-step:1]
    index = []
    for k in range(len(dtheta)):
        if dtheta[k] > 150:
            THETA_new[k+1:]-=180
        elif dtheta[k] < -150:
            THETA_new[k + 1:] += 180
        if dtheta[k] > 130:  # 处理周期行导致的大deltatheta
            dtheta[k] -= 180
        elif dtheta[k] < -130:
            dtheta[k] += 180
        if abs(dtheta[k]) > 30:  # 把明显由于识别错误产生的零星数据删掉
            index.append(k)
    dtheta = np.delete(dtheta, index)
    dr = np.delete(dr, index)
    dx = np.delete(dx, index)
    dy = np.delete(dy, index)
    center = np.delete(center, index,axis=0)

    # timestep change!
    THETA_new1 = THETA.copy()[::1500]  # 1s取一个点
    dtheta1 = THETA_new1[step::1] - THETA_new1[:-step:1]

    for k in range(len(dtheta1)):
        if dtheta1[k] > 90:
            THETA_new1[k + 1:] -= 180
        elif dtheta1[k] < -90:
            THETA_new1[k + 1:] += 180

    dtheta1 = THETA_new1[step::1] - THETA_new1[:-step:1]
    # dtheta1 = np.delete(dtheta1, index)


    # 1/150s timestep
    THETA_new2 = THETA.copy()  # 1s取一个点
    dtheta2 = THETA_new2[step::1] - THETA_new2[:-step:1]
    for k in range(len(dtheta2)):
        if dtheta2[k] > 90:
            THETA_new2[k + 1:] -= 180
        elif dtheta2[k] < -90:
            THETA_new2[k + 1:] += 180

    dtheta2 = THETA_new2[step::1] - THETA_new2[:-step:1]
    dtheta2 = np.delete(dtheta2, index)

    time = np.linspace(0, (N-len(index)-step)/150, N-len(index)-step)


# # fluctuation1  x,y,theta
#     fig, (ax0,ax1, ax2) = plt.subplots(nrows=3, figsize=(12, 6))
#     ax0 = plt.subplot(311)
#     # plt.plot(time, dr/(step/fps), color='yellowgreen', lw=0.1)
#     plt.plot(time, dx/480*260, color='yellowgreen', lw=0.5,marker='o',markersize=0.7)  #mm
#     # plt.scatter(time, dx/480*260, color='yellowgreen', lw=0.5,marker='o')  #mm
#     ax0.set_xlabel('time(s)')
#     # ax0.set_ylabel('$V(pixels/s)$')
#     ax0.set_ylabel('$dx(mm)$')
#     # ax0.set_ylim((-1,1))
#     ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
#     plt.title('Displacement '+ '[60Hz, ' + str(frame_name) +'$g$]', fontsize=10)
#     ax1 = plt.subplot(312)
#     plt.plot(time, dy/480*260, lw=0.5,color='orange',marker='o',markersize=0.7)  #mm
#     ax1.set_xlabel('time(s)')
#     ax1.set_ylabel('$dy(mm)$')
#     # ax1.set_ylim((-1,1))
#     ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
#     ax2 = plt.subplot(313)
#     # plt.plot(time, dtheta2/(step/fps), lw=0.1)  # degree/s
#     plt.plot(time, dtheta2*math.pi/180, lw=0.5,marker='o',markersize=0.7)  # rad
#     # ax2.set_ylabel('$\omega (degrees/s)$')  # degree/s
#     # plt.plot(time, dtheta2*math.pi/(step*180/fps), lw=0.1)  # rad/s
#     # ax2.set_ylabel('$omiga(rad/s)$')  # rad/s
#     ax2.set_ylabel('$d \Theta(rad)$')  # rad/s
#     ax2.set_xlabel('time (s)')
#     ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
#     # ax2.set_ylim((-0.4, 0.4))
#     plt.subplots_adjust(top=0.9, bottom=0.1)
    
#     # plt.text(0, 24, 'resolution 480*480 (1 pixel=0.54 mm)', fontsize=10, horizontalalignment='left', verticalalignment='top')
#     # plt.text(0, 30, 'angular resolution < 0.1°', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    
    
#     plt.show()

#-----------------
# # fluctuation2 r, theta

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    plt.title('Displacement '+ '[60Hz, ' + str(frame_name) +'$g$]', fontsize=10)
    ax1 = plt.subplot(211)
    plt.plot(time, np.sqrt(dy**2+dx**2)/480*260, lw=0.5,color='orange',marker='o',markersize=0.7)  #mm
    
    # # # #--- (no real) magnitude of angular acceleration rad/s^2
    # omega_mag = np.abs(np.diff(np.diff(np.abs(dtheta2)*math.pi/180)))
    # omega_mag_0 = [np.max(omega_mag[i*10:(10*i+10)]) for i in range(int(len(omega_mag)/10))]
    # omega_mag_1 = [np.max(omega_mag_0[i*5:(5*i+5)]) for i in range(int(len(omega_mag_0)/5))]
    # omega_mag_2 = [np.max(omega_mag_1[i*3:(3*i+3)]) for i in range(int(len(omega_mag_1)/3))]
    # # plt.plot(time[::150][:len(omega_mag_2)],omega_mag_2, lw=0.5,marker='o',markersize=0.7)  # magnitude of angular acceleration rad/s^2
    # b,a = signal.butter(8,0.01,'lowpass') #lowpassfilter wn=2*5/1000=0.01
    # activemode = signal.filtfilt(b,a,omega_mag)
    # ave_active = [activemode[:50].mean()]
    # for k in range(len(activemode)//50):
    #     if 50*k+50<len(activemode):
    #         ave_active.append(activemode[50*k:50*(k+1)].mean())
    #     else:
    #         ave_active.append(activemode[50*k:].mean())
    # ave_active=np.where(np.array(ave_active)<0.023,0,0.04)
    # plt.plot(time[:len(omega_mag)],activemode, lw=0.5,marker='o',markersize=0.7)  # magnitude of angular acceleration rad/s^2
    # plt.scatter(time[:len(omega_mag):50],ave_active,c='r')
    # plt.axhline(y=0.023,c='y')


    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('$dr(mm)$')
    # ax1.set_ylim((0,1.5))
    ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    plt.title('Displacement '+ '[60Hz, ' + str(frame_name) +'$g$]', fontsize=10)
    ax2 = plt.subplot(212)
    # plt.plot(time, dtheta2/(step/fps), lw=0.1)  # degree/s
    plt.plot(time, dtheta2*math.pi/180, lw=0.5,marker='o',markersize=0.7)  # rad
    # ax2.scatter(ip[:,0]/150.0,ip[:,1],c='r') 

    # ax2.set_ylabel('$\omega (degrees/s)$')  # degree/s
    # plt.plot(time, dtheta2*math.pi/(step*180/fps), lw=0.1)  # rad/s
    # ax2.set_ylabel('$omiga(rad/s)$')  # rad/s
    ax2.set_ylabel('$d \Theta(rad)$')  # rad/s
    ax2.set_xlabel('time (s)')
    ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    ax2.set_ylim((-0.35, 0.35))
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # plt.text(0, 24, 'resolution 480*480 (1 pixel=0.54 mm)', fontsize=10, horizontalalignment='left', verticalalignment='top')
    # plt.text(0, 30, 'angular resolution < 0.1°', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    
    
    plt.show()

# ###########--changepoints------------------------
#     # # #--- (no real) magnitude of angular acceleration rad/s^2
#     omega_mag = np.abs(np.diff(np.diff(np.abs(dtheta2)*math.pi/180)))
#     b,a = signal.butter(8,0.01,'lowpass') #lowpassfilter wn=2*5/1000=0.01
#     activemode = signal.filtfilt(b,a,omega_mag)
#     ave_active = [activemode[:50].mean()]  # 1s 3points
#     for k in range(len(activemode)//50):
#         if 50*k+50<len(activemode):
#             ave_active.append(activemode[50*k:50*(k+1)].mean())
#         else:
#             ave_active.append(activemode[50*k:].mean())
#     ave_active=np.where(np.array(ave_active)<0.023,0,0.04)
#     timesmarker=np.where(ave_active>0)[0]
#     print(timesmarker)
#     # draw changepoints
#     fig, (ax1, ax2) = plt.subplots(nrows=2)
#     plt.title('Displacement '+ '[60Hz, ' + str(frame_name) +'$g$]', fontsize=10)
#     ax1 = plt.subplot(211)
#     plt.plot(time[:len(omega_mag)],activemode, lw=0.5,marker='o',markersize=0.7)  # magnitude of angular acceleration rad/s^2
#     plt.scatter(time[:len(omega_mag):50],ave_active,c='r')
#     plt.axhline(y=0.023,c='y')

#     ax1.set_xlabel('time(s)')
#     ax1.set_ylabel('$a$')
#     # ax1.set_ylim((0,1.5))
#     ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
#     plt.title('Displacement '+ '[60Hz, ' + str(frame_name) +'$g$]', fontsize=10)
#     ax2 = plt.subplot(212)
#     plt.plot(time, dtheta2*math.pi/180, lw=0.5,marker='o',markersize=0.7)  # rad
#     ax2.set_ylabel('$d \Theta(rad)$')  # rad/s
#     ax2.set_xlabel('time (s)')
#     ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
#     ax2.set_ylim((-0.35, 0.35))
#     plt.subplots_adjust(top=0.9, bottom=0.1)
#     plt.show()

    
#     # find real changepoints
#     changepoints = ranges(timesmarker)
#     print(changepoints)
#     cp_new=[]

#     max_dis = 4 # fix5g(55hznotgood) 5 # fix60hz(allgood)

#     for k in range(len(changepoints)):
#         if  changepoints[k][1]-changepoints[k][0]<max_dis:
#             continue
#         if not cp_new or changepoints[k][1]-changepoints[k][0]>max_dis:
#             if k>=1 and cp_new and changepoints[k][0]-cp_new[-1][1]<=max_dis:
#                 cp_new[-1]=(cp_new[-1][0],changepoints[k][1])
#             else:
#                 cp_new.append(changepoints[k])
#     print(cp_new)  #changepoints
# # ------------------------------------------------------

    # # # output file
    # nm = ['active\\','inactive\\']
    # if len(cp_new)==0:
    #     names = ['center', 'theta']
    #     ellipse = tuple([center,theta])
    #     store_path =path3 + nm[1] + filename[i] + '.h5'
    #     store_temp =pd.HDFStore(store_path, complib='blosc')
    #     for p in range(len(ellipse[0])):
    #         ellipse_dict = dict(zip(names, tuple((ellipse[0][p],ellipse[1][p]))))
    #         store_temp.append(key='center', value=pd.DataFrame((ellipse_dict['center'],)))  # 参数输出
    #         store_temp.append(key='theta', value=pd.DataFrame((ellipse_dict['theta'],)))
    #     store_temp.close()
    #     del store_temp
    # else:
    #     count = 0
    #     for k in range(len(cp_new)): #active
    #         names = ['center', 'theta']
    #         ellipse = tuple((center[50*cp_new[k][0]:50*cp_new[k][1]].tolist(),theta[50*cp_new[k][0]:50*cp_new[k][1]].tolist()))
    #         store_path =path3 + nm[0] + filename[i] +'_'+str(count)+ '.h5'
    #         store_temp =pd.HDFStore(store_path, complib='blosc')
    #         for p in range(len(ellipse[0])):
    #             ellipse_dict = dict(zip(names, tuple((ellipse[0][p],ellipse[1][p]))))
    #             store_temp.append(key='center', value=pd.DataFrame((ellipse_dict['center'],)))  # 参数输出
    #             store_temp.append(key='theta', value=pd.DataFrame((ellipse_dict['theta'],)))
    #         store_temp.close()
    #         count+=1
    #         del store_temp
    #     count = 0
    #     for k in range(len(cp_new)+1): #inactive
    #         names = ['center', 'theta']
    #         if k ==0:
    #             if cp_new[k][0]!=0:
    #                 ellipse = tuple([center[:50*cp_new[k][0]],theta[:50*cp_new[k][0]]])
    #             else:
    #                 continue
    #         if k == len(cp_new):
    #             if 3*cp_new[k-1][1]!=len(theta):
    #                 ellipse = tuple([center[50*cp_new[k-1][1]:],theta[50*cp_new[k-1][1]:]])
    #             else:
    #                 continue
    #         if k > 0 and k < len(cp_new):
    #             ellipse = tuple([center[50*cp_new[k-1][1]:50*cp_new[k][0]],theta[50*cp_new[k-1][1]:50*cp_new[k][0]]])
    #         store_path =path3 + nm[1] + filename[i] +'_'+str(count)+ '.h5'
    #         store_temp =pd.HDFStore(store_path, complib='blosc')
    #         for p in range(len(ellipse[0])):
    #             ellipse_dict = dict(zip(names, tuple((ellipse[0][p],ellipse[1][p]))))
    #             store_temp.append(key='center', value=pd.DataFrame((ellipse_dict['center'],)))  # 参数输出
    #             store_temp.append(key='theta', value=pd.DataFrame((ellipse_dict['theta'],)))
    #         store_temp.close()
    #         count+=1
    #         del store_temp




# ##########-每个文件traj pic--------------------
    # fig, ax = plt.subplots()
    # ax = plt.subplot(111)
    # # traj = pd.DataFrame({'x': center[1::1500,0], 'y': center[1::1500,1]}) # timestep=10s
    # # traj1 = pd.DataFrame({'x': center[1::1, 0], 'y': center[1::1, 1]}) # timestep=1/150s
    # # print(traj.head())
    # st=int(30)
    # plt.plot(center[::st, 0],  center[::st, 1],color='b', lw=0.5,marker='o',markersize=0.7,label= 'timestep ='+str(round(st/150.0,4))+'s')  # rad 1/150s timestep
    # ## # active mode--============================
    # # # 4g_4
    # plt.plot(center[290*150:294*150:st, 0],  center[290*150:294*150:st, 1],color='r', lw=0.5,marker='o',markersize=0.7) 
    # plt.plot(center[397*150:400*150:st, 0],  center[397*150:400*150:st, 1],color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[444*150:450*150:st, 0],  center[444*150:450*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[548*150:551*150:st, 0],  center[548*150:551*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[578*150:588*150:st, 0],  center[578*150:588*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[644*150:646*150:st, 0],  center[644*150:646*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[670*150:672*150:st, 0],  center[670*150:672*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[719*150:754*150:st, 0],  center[719*150:754*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[828*150:838*150:st, 0],  center[828*150:838*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[884*150:912*150:st, 0],  center[884*150:912*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[969*150:982*150:st, 0],  center[969*150:982*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[991*150:1014*150:st, 0],  center[991*150:1014*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[1105*150:1115*150:st, 0],  center[1105*150:1115*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[1167*150:1173*150:st, 0],  center[1167*150:1173*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[1272*150:1282*150:st, 0],  center[1272*150:1282*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # plt.plot(center[1345*150:1356*150:st, 0],  center[1345*150:1356*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # # # 4g_1
    # # plt.plot(center[64*150:138*150:st, 0],  center[64*150:138*150:st, 1],color='r', lw=0.5,marker='o',markersize=0.7) 
    # # plt.plot(center[377*150:448*150:st, 0],  center[377*150:448*150:st, 1],color='r', lw=0.5,marker='o',markersize=0.7)  
    # # plt.plot(center[452*150:532*150:st, 0],  center[452*150:532*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    # # # 4g_0
    # # plt.plot(center[828*150:872*150:st, 0],  center[828*150:872*150:st, 1],color='r', lw=0.5,marker='o',markersize=0.7) 
    # # plt.plot(center[927*150:335*150:st, 0],  center[950*150:335*150:st, 1],color='r', lw=0.5,marker='o',markersize=0.7)  
    # # plt.plot(center[1006*150:1073*150:st, 0],  center[1006*150:1073*150:st, 1], color='r', lw=0.5,marker='o',markersize=0.7)  
    
    # # plt.plot(center[1::1500,0], center[1::1500,1], lw=2, marker='o', markersize=1.4, label = 'timestep = 10s')  # rad #10s 的步长
    # # ax=traj.plot(x='x', y='y', alpha=0.6, legend='timestep = 1/150s', title='trajectory')
    # ax.set_xlim(0, 480)  # (traj['x'].min()-10, traj['x'].max()+10)
    # ax.set_ylim(0, 480)  # (traj['y'].min()-10, traj['y'].max()+10)
    # ax.set_xlabel('x(pixel)')
    # ax.set_ylabel('y(pixel)')
    # ax.legend()
    # leg = ax.legend()
    # leg.get_frame().set_linewidth(0.0)
    # ax.plot()
    # plt.text(1.5, 1, '1 pixel=0.54 mm', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    # plt.show()
    # fig = ax.get_figure()
    # del ax, fig


# ##########-每个文件x-deltax method 1--------------------
# #     traj = pd.DataFrame({'x': dtheta1, 'y': THETA_new1[1:]%360.0})
#     fig, ax = plt.subplots()
#     # ax = traj.plot(x='x', y='y', alpha=0.6, linewidth=0.1, legend=False, title='$\Theta$-$\Delta \Theta$')
#     plt.scatter(dx[:15000],center[1:15001,0], alpha=0.6, marker='.',s=1)  # THETA_new1[1:15001], # dtheta1[:15000] # dx[:10000] # dr # center[1:5001,0]  # dy[0:5000]
#     plt.title('$x-\Delta x$') # \Theta  # \Delta \Theta  # \Delta y
#     ax.set_ylabel('$x(pixel)$')  # \Delta x(pixel)  # \Theta(degree)  # x(pixel)
#     ax.set_xlabel('$\Delta x(pixel)$')  # \Delta x(pixel)  # \Delta \Theta(degree)
#     ax.plot()
#     # plt.text(1.5, 1, '1 pixel=0.54 mm', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
#     plt.show()
#
#     del ax, fig

# ##########-chaos tiseanio--------------------



#### angle chang timestpe fluc---------
    # fig, ax = plt.subplots()
    # ax = plt.subplot(111)
    #
    # time = np.linspace(0, (len(THETA_new2)) / 150, len(THETA_new2))
    #
    # # # plt.plot(time, dtheta/(step/fps), lw=0.1)  # degree/s
    # # plt.plot(time, THETA_new1*math.pi/180, lw=0.5,marker='o',markersize=0.7)  # rad
    # plt.plot(time, THETA_new2*math.pi/180, lw=0.5,marker='o',markersize=0.7,label= 'timestep = 1/150s')  # rad 1/150s timestep
    # plt.plot(time[::1500], THETA_new1 * math.pi / 180, lw=2, marker='o', markersize=1.4, label = 'timestep = 10s')  # rad #10s 的步长
    # # plt.scatter(time, THETA_new1, marker='.', s=2)
    # ax.set_ylabel('$ \Theta(rad)$')  # rad/s
    # ax.set_xlabel('time(s)')
    # ax.legend()
    # ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    # leg = ax.legend()
    # leg.get_frame().set_linewidth(0.0)
    # plt.show()

