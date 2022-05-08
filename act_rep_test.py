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
u_ar_step = 0.

def Act_Rep(x,t):
    xdot = np.zeros(len(x))
    # Activator Repressor
    xdot[0] = - ar_gamma_A * x[0] + ar_kappa_A / ar_delta_A * (ar_alpha_A * (x[0] / ar_K_A) ** ar_n + ar_alpha_A0) / (1 + (x[0] / ar_K_A) ** ar_n + (x[1] / ar_K_B) ** ar_m)
    xdot[1] = - ar_gamma_B * x[1] + ar_kappa_B / ar_delta_B * (ar_alpha_B * (x[0] / ar_K_A) ** ar_n + ar_alpha_B0) / (1 + (x[0] / ar_K_A) ** ar_n)
    return xdot


numpy_random_initial_condition_seed = 10
# Simulation Parameters
simulation_time = 30
sampling_time = 0.5
t = np.arange(0, simulation_time, sampling_time)
# x0_init_ar = np.array([0.1,0.8])
x0_init_ar = np.array([150,150])



X = odeint(Act_Rep,x0_init_ar,t)

plt.plot(X)
plt.show()