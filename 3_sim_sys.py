import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import copy
plt.rcParams["font.family"] = "Avenir"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
import seaborn as sb

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import copy
import pickle
plt.rcParams["font.family"] = "Avenir"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
# from Simulated_HighDimensional_Systems import *
import os
import shutil
# import keras
# from keras.models import Sequential
# from keras.layers import Dense


# Constant Parameters
# Activator Repressor
ar_gamma_A = 1.
ar_gamma_B = 0.5
ar_delta_A = 1.
ar_delta_B = 1.
ar_alpha_A0= 0.04
ar_alpha_B0= 0.004
ar_alpha_A = 250.
ar_alpha_B = 30.
ar_K_A = 1.
ar_K_B = 1.5
ar_kappa_A = 1.
ar_kappa_B = 1.
ar_n = 2.
ar_m = 3.

ar2_gamma_A = .1
ar2_gamma_B = 0.17
ar2_delta_A = 1.5
ar2_delta_B = .3
ar2_alpha_A0= .4
ar2_alpha_B0= 4.004
ar2_alpha_A = 250.
ar2_alpha_B = 30.
ar2_K_A = 4.5
ar2_K_B = 0.5
ar2_kappa_A = 1.
ar2_kappa_B = 1.
ar2_n = 2.
ar2_m = 3.

def Ar(x,t, couple_ar_rrr = 1.,couple_ar_ts = 1.):
    xdot = np.zeros(len(x))
    # Activator Repressor
    xdot[0] = - ar_gamma_A * x[0] + ar_kappa_A / ar_delta_A * (ar_alpha_A * (x[0] / ar_K_A) ** ar_n + ar_alpha_A0) / (1 + (x[0] / ar_K_A) ** ar_n + (x[1] / ar_K_B) ** ar_m)
    xdot[1] = - ar_gamma_B * x[1] + ar_kappa_B / ar_delta_B * (ar_alpha_B * (x[0] / ar_K_A) ** ar_n + ar_alpha_B0) / (1 + (x[0] / ar_K_A) ** ar_n)
    xdot[2] = x[1] - ar2_gamma_A * x[2] + ar2_kappa_A / ar2_delta_A * (ar2_alpha_A * (x[2] / ar2_K_A) ** ar2_n + ar2_alpha_A0) / (
                1 + (x[2] / ar2_K_A) ** ar2_n + (x[3] / ar2_K_B) ** ar2_m)
    xdot[3] = - ar2_gamma_B * x[3] + ar2_kappa_B / ar2_delta_B * (ar2_alpha_B * (x[2] / ar2_K_A) ** ar2_n + ar2_alpha_B0) / (
                1 + (x[2] / ar2_K_A) ** ar2_n)
    return xdot

simulation_time = 200
sampling_time = 0.5
t = np.arange(0, simulation_time, sampling_time)
x0_init_rrr = np.array([0,0,400,200])
X = odeint(Ar,x0_init_rrr,t)
plt.plot(t,X)
plt.show()