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




# # translational
# fig = plt.figure(figsize=(4, 4))
# ax1 = fig.add_subplot(111)
# # 60Hz [3.5,4,4.5,5] [1.2180025810076303, 1.044909894908142, 1.555060069462591, 1.416293592715279]
# # 5g [50.0, 60.0, 70.0, 75.0, 85.0] [1.5353867441020084, 1.3856000172953038, 1.534442019708877, 1.3816277749330572, 1.1570582641201572]
# # 0.6mm [50, 60, 80, 100] [1.094605023811152, 1.2404976473407272, 1.9093275796870413, 2.1897792752892813]
# # # a-----------------------------
# a1 = [3.5,4,4.5,5]
# a2 = [5,5,5,5,5]
# a3 = [(2*math.pi*50)**2*0.0006/9.8,(2*math.pi*60)**2*0.0006/9.8,(2*math.pi*80)**2*0.0006/9.8,(2*math.pi*100)**2*0.0006/9.8]
# ay1 = [1.2180025810076303, 1.044909894908142, 1.555060069462591, 1.416293592715279]
# ay2 = [1.5353867441020084, 1.3856000172953038, 1.534442019708877, 1.3816277749330572, 1.1570582641201572]
# ay3 = [1.094605023811152, 1.2404976473407272, 1.9093275796870413, 2.1897792752892813]
# ax1.set_xlabel(r'$a$ ($g$)', fontsize=9)
#
# # # # f-----------------------------
# # a1 = [60,60,60,60]
# # a2 = [50.0, 60.0, 70.0, 75.0, 85.0]
# # a3 = [50, 60, 80, 100]
# # ay1 = [1.2180025810076303, 1.044909894908142, 1.555060069462591, 1.416293592715279]
# # ay2 = [1.5353867441020084, 1.3856000172953038, 1.534442019708877, 1.3816277749330572, 1.1570582641201572]
# # ay3 = [1.094605023811152, 1.2404976473407272, 1.9093275796870413, 2.1897792752892813]
# # ax1.set_xlabel(r'$f$ (Hz)', fontsize=9)
#
# # # # A-----------------------------
# # a1 = [3.5*9.8/(2*math.pi*60)**2,4*9.8/(2*math.pi*60)**2,4.5*9.8/(2*math.pi*60)**2,5*9.8/(2*math.pi*60)**2]
# # a2 = [5*9.8/(2*math.pi*50)**2, 5*9.8/(2*math.pi*60)**2, 5*9.8/(2*math.pi*70)**2, 5*9.8/(2*math.pi*75)**2, 5*9.8/(2*math.pi*85)**2]
# # a3 = [0.0006, 0.0006, 0.0006, 0.0006]
# # ay1 = [1.2180025810076303, 1.044909894908142, 1.555060069462591, 1.416293592715279]
# # ay2 = [1.5353867441020084, 1.3856000172953038, 1.534442019708877, 1.3816277749330572, 1.1570582641201572]
# # ay3 = [1.094605023811152, 1.2404976473407272, 1.9093275796870413, 2.1897792752892813]
# # ax1.set_xlabel(r'$A$ (m)', fontsize=9)
#
# label = [r'$f=60$ Hz',r'$a=5g$',r'$A=0.6$ mm']
# ax1.scatter(a1, ay1,  c='',edgecolor='b',marker='o',s=25,label=label[0])
# ax1.scatter(a2, ay2, c='',edgecolor='g',marker='^',s=25,label=label[1])
# ax1.scatter(a3, ay3,  c='',edgecolor='r',marker='s',s=25,label=label[2])
#
# leg1 = ax1.legend(loc='upper left')
# leg1.get_frame().set_linewidth(0.0)
# ax1.tick_params(axis="x", direction="in", labelsize=9)
# ax1.tick_params(which='minor', direction='in')
# ax1.tick_params(axis="y", direction="in", labelsize=9)
# plt.show()


# rotational--- delta theta_0
fig = plt.figure(figsize=(4, 4))
ax1 = fig.add_subplot(111)
# # a-----------------------------
a1 = [3.5,4,4.5,5]
a2 = [5,5,5,5,5]
a3 = [(2*math.pi*50)**2*0.0006/9.8,(2*math.pi*60)**2*0.0006/9.8,(2*math.pi*80)**2*0.0006/9.8,(2*math.pi*100)**2*0.0006/9.8]
ay1 = [0.093,0.084,0.084,0.089]
ay2 = [0.086,0.094,0.094,0.03,0.03]
ay3 = [2.21E-05,0.079,0.13,0.22]
ax1.set_xlabel(r'$a$ ($g$)', fontsize=9)

# # # f-----------------------------
# a1 = [60,60,60,60]
# a2 = [50.0, 60.0, 70.0, 75.0, 85.0]
# a3 = [50, 60, 80, 100]
# ay1 = [0.093,0.084,0.084,0.089]
# ay2 = [0.086,0.094,0.094,0.03,0.03]
# ay3 = [2.21E-05,0.079,0.13,0.22]
# ax1.set_xlabel(r'$f$ (Hz)', fontsize=9)

# # # A-----------------------------
# a1 = [3.5*9.8/(2*math.pi*60)**2,4*9.8/(2*math.pi*60)**2,4.5*9.8/(2*math.pi*60)**2,5*9.8/(2*math.pi*60)**2]
# a2 = [5*9.8/(2*math.pi*50)**2, 5*9.8/(2*math.pi*60)**2, 5*9.8/(2*math.pi*70)**2, 5*9.8/(2*math.pi*75)**2, 5*9.8/(2*math.pi*85)**2]
# a3 = [0.0006, 0.0006, 0.0006, 0.0006]
# ay1 = [0.093,0.084,0.084,0.089]
# ay2 = [0.086,0.094,0.094,0.03,0.03]
# ay3 = [2.21E-05,0.079,0.13,0.22]
# ax1.set_xlabel(r'$A$ (m)', fontsize=9)

ax1.set_ylabel(r'$\Delta \Theta_0$ (rad)', fontsize=9)
ax1.yaxis.set_label_coords(-0.1,0.5)
label = [r'$f=60$ Hz',r'$a=5g$',r'$A=0.6$ mm']
ax1.scatter(a1, ay1,  c='',edgecolor='b',marker='o',s=25,label=label[0])
ax1.scatter(a2, ay2, c='',edgecolor='g',marker='^',s=25,label=label[1])
ax1.scatter(a3, ay3,  c='',edgecolor='r',marker='s',s=25,label=label[2])

leg1 = ax1.legend(loc='upper left')
leg1.get_frame().set_linewidth(0.0)

ax1.tick_params(axis="x", direction="in", labelsize=9)
ax1.tick_params(which='minor', direction='in')
ax1.tick_params(axis="y", direction="in", labelsize=9)
plt.show()