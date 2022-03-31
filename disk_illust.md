[TOC]



## 1st STEP

###### test1.py 

​	——处理图像

​	【图像位置D:\guan2019\1_disk\1.2_pics】

​	【输出数据位置：D:\guan2019\1_disk\2_data】



## 2nd STEP

### 2.1 [FOR ..._a &..._f data]

##### 2.1.1 基础代码

###### test1_cal.py 

​	——计算msd，pdf，position，trajectory的基础代码

​	——输出traj，position

​	【源数据位置：D:\guan2019\1_disk\f\...or\\a...】

​	【输出数据位置：D:\guan2019\1_disk\f...or\\a...】





##### 2.1.2 计算MSD

###### test2_msd.py

​		——计算平动MSD，细节

###### test2_msd_temp.py

​		——与上一个相同，用于并行计算

###### test2_msd_tempslope.py

​		——计算斜率变化

###### test2_msd_diffusioncoefficient.py

​		——计算扩散系数

###### test2_msd_drawtogether.py

​		——把MSD画在一起

​		——平动/转动都用这个

------------------------

###### test2_msdrot.py

​		——计算转动MSD

###### test2_msdrot_diffusioncoefficient.py

​		——计算扩散系数



##### 2.1.3 计算v

###### test2_v.py

​		——计算速度



##### 2.1.4 计算PDF

###### test2_fre_cal.py 

​		——计算 D:\guan2019\1_disk\f\下 按fre分类的数据的pdf图像

​		——细节，可截断

​		——改步长

​		**！！！！重点数据：60Hz下各个加速度**

###### test2_pdf_fitting_60Hz.py 

​		——用lmfit库中模型https://lmfit.github.io/lmfit-py/builtin_models.html#skewedgaussianmodel 拟合

###### test2_pdf_fitting_5g.py 

​		——5g pdf ,可拟合 还没拟合出结果



###### test2_a_cal.py 

​		——计算 D:\guan2019\1_disk\a\下 按fre分类的数据的pdf图像

​		——同上

​		**！！！！重点数据：5g下各个频率**

【分析总结位置：D:\guan2019\1_disk\1.2_pics中..._else文件夹中的readme.md】

------

### 2.2 [FOR data in ...//TLc/...]

##### 2.2.1 基础代码

######  test1_cal_TLc.py 

​	——计算msd，pdf，position，trajectory的基础代码

​	——输出traj，position

​	【源数据位置：D:\guan2019\1_disk\TLc\\\...】



##### 2.2.2 计算MSD

###### test2_msd_TLc.py

​		——计算MSD，细节

###### test2_msdrot_TLc.py

​		——计算MSD，细节



##### 2.2.3 计算v

###### test2_v_TLc.py

​		——计算速度



##### 2.2.4 计算PDF

###### test2_fre_cal_TLc.py 

​		——计算pdf图像

​		——细节，可截断

​		——改步长

###### test2_pdf_fitting_TLc.py 

​		——用lmfit库中模型https://lmfit.github.io/lmfit-py/builtin_models.html#skewedgaussianmodel 拟合



##### 2.2.5 计算能量分配

###### test2_energy.py

​		——计算能量分配



##### 2.2.5 计算相关性

###### test2_autocorrelation.py

​		——计算自相关

​		——转动平动自相关都用这个

##### 2.2.5 计算功率谱

###### test2_powerspectrum.py



## 3rd STEP——输出

###### op_drdtheta.py

​		——输出dr,dtheta图

​		——计算能量均值