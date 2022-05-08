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

def organize_output_data_for_learning(ls_data, n_delay_embeddings =1, ls_output_columns = [0,1,2]):
    # Organize the data
    ls_dataset_indices = list(range(len(ls_data)))
    datapoints_per_dataset = len(ls_data[0]['Y'])
    X = np.empty((0, n_delay_embeddings * len(ls_output_columns)))
    Y = np.empty((0, len(ls_output_columns)))
    for d_i, j in product(ls_dataset_indices, list(range(n_delay_embeddings, datapoints_per_dataset))): # j tracks the output
        X = np.concatenate([X, ls_data[d_i]['Y'][j-n_delay_embeddings:j,ls_output_columns].T.reshape(1,-1)], axis=0)
        Y = np.concatenate([Y, ls_data[d_i]['Y'][j:j+1,ls_output_columns]], axis=0)
    return X,Y
def organize_output_data_for_deepDMD(ls_data, n_delay_embeddings =1, ls_output_columns = [0,1,2], with_intersection = True):
    # Organize the data
    ls_dataset_indices = list(range(len(ls_data)))
    datapoints_per_dataset = len(ls_data[0]['YT'])
    XpT = np.empty((0, n_delay_embeddings * len(ls_output_columns)))
    XfT = np.empty((0, n_delay_embeddings * len(ls_output_columns)))
    if with_intersection:
        for d_i, j in product(ls_dataset_indices, list(range(n_delay_embeddings, datapoints_per_dataset))): # j tracks the output
            XpT = np.concatenate([XpT, ls_data[d_i]['YT'][j-n_delay_embeddings:j,ls_output_columns].T.reshape(1,-1)], axis=0)
            XfT = np.concatenate([XfT, ls_data[d_i]['YT'][j-n_delay_embeddings+1:j+1,ls_output_columns].T.reshape(1,-1)], axis=0)
    else:
        n_delay_embedded_points_per_dataset = np.int(np.floor(datapoints_per_dataset / n_delay_embeddings))
        for d_i, j in product(ls_dataset_indices, list(range(n_delay_embedded_points_per_dataset - 1))):
            XpT = np.concatenate([XpT, ls_data[d_i]['YT'][j * n_delay_embeddings:(j + 1) * n_delay_embeddings,
                                       ls_output_columns].T.reshape(1, -1)], axis=0)
            XfT = np.concatenate([XfT, ls_data[d_i]['YT'][(j + 1) * n_delay_embeddings:(j + 2) * n_delay_embeddings,
                                       ls_output_columns].T.reshape(1, -1)], axis=0)
    return XpT, XfT
def organize_state_data_for_deepDMD(ls_data, ls_state_columns = [0,1,2,3,4,5]):
    # Organize the data
    ls_dataset_indices = list(range(len(ls_data)))
    n_states = len(ls_state_columns)
    XpT = np.empty((0, n_states))
    XfT = np.empty((0, n_states))
    for d_i in ls_dataset_indices:
        XpT = np.concatenate([XpT, ls_data[d_i]['XT'][0:-1,ls_state_columns]], axis=0)
        XfT = np.concatenate([XfT, ls_data[d_i]['XT'][1:,ls_state_columns]], axis=0)
    return XpT, XfT
def organize_state_and_output_data_for_deepDMD(ls_data, ls_state_columns = [0,1,2,3,4,5,6], ls_output_columns = [0]):
    # Organize the data
    ls_dataset_indices = list(range(len(ls_data)))
    n_states = len(ls_state_columns)
    n_outputs = len(ls_output_columns)
    XpT = np.empty((0, n_states))
    XfT = np.empty((0, n_states))
    YpT = np.empty((0, n_outputs))
    YfT = np.empty((0, n_outputs))
    for d_i in ls_dataset_indices:
        XpT = np.concatenate([XpT, ls_data[d_i]['XT'][0:-1,ls_state_columns]], axis=0)
        XfT = np.concatenate([XfT, ls_data[d_i]['XT'][1:,ls_state_columns]], axis=0)
        YpT = np.concatenate([YpT, ls_data[d_i]['YT'][0:-1,ls_output_columns]], axis=0)
        YfT = np.concatenate([YfT, ls_data[d_i]['YT'][1:,ls_output_columns]], axis=0)
    return XpT, XfT, YpT, YfT

def create_folder(SYSTEM_NO):
    storage_folder = '/Users/shara/Desktop/TS_AR_IFFL/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
        get_input = input('Do you wanna delete the existing system[y/n]? ')
        if get_input == 'y':
            shutil.rmtree(storage_folder)
            os.mkdir(storage_folder)
        else:
            return
    else:
        os.mkdir(storage_folder)
    return storage_folder


def save_system(ls_data, dict_measurement_info, SYSTEM_NO):
    # STEP 2: ORGANIZE THE DATA FOR THE DELAY EMBEDDING
    # Simulation Parameters
    state_measured = dict_measurement_info['state_measured']
    output_measured = dict_measurement_info['output_measured']
    n_delay_embeddings = dict_measurement_info['n_delay_embedding']
    ls_measured_output_indices = dict_measurement_info['ls_measured_output_indices']
    ls_measured_state_indices = dict_measurement_info['ls_measured_state_indices']
    formulate_Koopman_output_data_with_intersection = dict_measurement_info[
        'formulate_Koopman_output_data_with_intersection']

    # Incorporate all of these into the functions
    if state_measured:
        n_delay_embeddings = 1
        formulate_Koopman_output_data_with_intersection = False
    if not output_measured:
        ls_measured_output_indices = []

    # Fill these functions
    if output_measured and not state_measured:
        XpT, XfT = organize_output_data_for_deepDMD(ls_data, n_delay_embeddings=n_delay_embeddings,
                                                    ls_output_columns=ls_measured_output_indices,
                                                    with_intersection=formulate_Koopman_output_data_with_intersection)
    elif state_measured and not output_measured:
        XpT, XfT = organize_state_data_for_deepDMD(ls_data, ls_state_columns=ls_measured_state_indices)
    elif state_measured and output_measured:
        XpT, XfT, YpT, YfT = organize_state_and_output_data_for_deepDMD(ls_data, ls_state_columns=ls_measured_state_indices, ls_output_columns=ls_measured_output_indices)

    # Partition to training and testing data
    n_one_third_data = np.int(np.floor(XpT.shape[0] / 3))

    dict_DATA_RAW = {}
    dict_DATA_RAW['train'] = {'XpT': XpT[0:n_one_third_data, :], 'XfT': XfT[0:n_one_third_data, :]}
    dict_DATA_RAW['valid'] = {'XpT': XpT[n_one_third_data:2 * n_one_third_data, :],
                              'XfT': XfT[n_one_third_data:2 * n_one_third_data, :]}
    dict_DATA_RAW['test'] = {'XpT': XpT[2 * n_one_third_data:, :], 'XfT': XfT[2 * n_one_third_data:, :]}
    dict_DATA_RAW['embedding'] = n_delay_embeddings

    dict_Scaler = {}
    dict_Scaler['XT'] = StandardScaler(with_mean=True, with_std=True).fit(dict_DATA_RAW['train']['XpT'])

    if state_measured and output_measured:
        dict_DATA_RAW['train']['YpT'] = YpT[0:n_one_third_data, :]
        dict_DATA_RAW['train']['YfT'] = YfT[0:n_one_third_data, :]
        dict_DATA_RAW['valid']['YpT'] = YpT[n_one_third_data:2 * n_one_third_data, :]
        dict_DATA_RAW['valid']['YfT'] = YfT[n_one_third_data:2 * n_one_third_data, :]
        dict_DATA_RAW['test']['YpT'] = YpT[2 * n_one_third_data:, :]
        dict_DATA_RAW['test']['YfT'] = YfT[2 * n_one_third_data:, :]
        dict_Scaler['YT'] = StandardScaler(with_mean=True, with_std=True).fit(dict_DATA_RAW['train']['YpT'])

    dict_DATA_SCALED = {}
    for i in ['train', 'test', 'valid']:
        dict_DATA_SCALED[i] = {}
        for items in dict_DATA_RAW[i]:
            if items in ['XpT', 'XfT']:
                dict_DATA_SCALED[i][items] = dict_Scaler['XT'].transform(dict_DATA_RAW[i][items])
            elif items in ['YpT', 'YfT']:
                dict_DATA_SCALED[i][items] = dict_Scaler['YT'].transform(dict_DATA_RAW[i][items])
    dict_DATA_SCALED['embedding'] = n_delay_embeddings

    # STEP 3: SAVE THE DATA TO THE REQUIED FOLDER
    storage_folder = create_folder(SYSTEM_NO)
    # Save the scaler
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
        pickle.dump(dict_Scaler, handle)
    # Save the data for OC_deepDMD
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DeepDMDdata_Scaled.pickle', 'wb') as handle:
        pickle.dump(dict_DATA_SCALED, handle)
    # Save the unscaled deepDMD data
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DeepDMDdata_Unscaled.pickle', 'wb') as handle:
        pickle.dump(dict_DATA_SCALED, handle)
    # Save the original data
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'wb') as handle:
        pickle.dump(ls_data, handle)
    # Store the data in Koopman
    # with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
    #     pickle.dump(dict_DATA_SCALED, handle)
    # Save the simulated data information as well
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_SimulatedDataInfo.pickle', 'wb') as handle:
        pickle.dump(dict_measurement_info, handle)
    return

def output_type_1(A,B,Vm=1,KA=1,KB=1,nA=1,nB=1):
    # A--->C | B---|C
    # Model: C = Vm * A /(1 + B/K)
    C = Vm*(A/KA)**nA/(1 + (A/KA)**nA + (B/KB)**nB)
    return C
def output_type_2(A,Vm=1,KA=1,KB=1,nA=1,nB=1):
    # A--->C | B---|C
    # Model: C = Vm * A /(1 + B/K)
    C = Vm*(A/KA)**nA/(1 + (A/KA)**nA)
    return C

# Activator Repressor
ar_gamma_A = 1.
ar_gamma_B = 0.5
ar_delta_A = 1.
ar_delta_B = 1.
ar_alpha_A0= 0.04
ar_alpha_B0= 0.004
ar_alpha_A = 250
ar_alpha_B = 30
ar_K_A = 1.
ar_K_B = 1.
ar_kappa_A = 1.
ar_kappa_B = 1.
ar_n = 2.
ar_m = 4.
u_ar_step = 0.
# Repressilator
rrr_gamma1 = 0.3
rrr_gamma2 = 0.3
rrr_gamma3 = 0.3
rrr_K1 = 1
rrr_K2 = 1
rrr_K3 = 1
rrr_n1 = 2
rrr_n2 = 4
rrr_n3 = 3
rrr_alpha1 = 10
rrr_alpha2 = 10
rrr_alpha3 = 10
# Toggle Switch
ts_beta = 1.
ts_K = 10.
ts_gamma = 0.09
ts_n = 1.
u_ts_step = 0
ts_k0 = 0.01
def rrr(x,t):
    xdot = np.zeros(len(x))
    xdot[0] = - ar_gamma_A * x[0] + ar_kappa_A / ar_delta_A * (ar_alpha_A * (x[0] / ar_K_A) ** ar_n + ar_alpha_A0) / (
                1 + (x[0] / ar_K_A) ** ar_n + (x[1] / ar_K_B) ** ar_m)
    xdot[1] = - ar_gamma_B * x[1] + ar_kappa_B / ar_delta_B * (ar_alpha_B * (x[0] / ar_K_A) ** ar_n + ar_alpha_B0) / (
                1 + (x[0] / ar_K_A) ** ar_n)
    # Repressilator
    xdot[2] = 0.1*x[0] - rrr_gamma1 * x[2] + rrr_alpha1 / (1 + (x[4] / rrr_K1) ** rrr_n1)
    xdot[3] = - rrr_gamma2 * x[3] + rrr_alpha2 / (1 + (x[2] / rrr_K2) ** rrr_n2)
    xdot[4] = - rrr_gamma3 * x[4] + rrr_alpha3 / (1 + (x[3] / rrr_K3) ** rrr_n3)
    # Toggle Switch
    xdot[5] = ts_beta / (1 + (x[6] / ts_K) ** ts_n) - ts_gamma*3.8 * x[5] + 1e-3 * x[1]
    xdot[6] = ts_beta / (1 + (x[5] / ts_K) ** ts_n) - ts_gamma * x[6]
    return xdot


# simulation_time = 100
# sampling_time = 0.5
# t = np.arange(0, simulation_time, sampling_time)
# x0_init_rrr = np.array([0,0,50.,10., 10., 40., 80.4])
# X = odeint(rrr,x0_init_rrr,t)
# plt.plot(t,X)
# plt.legend(list(range(1,len(X[0])+1)))
# plt.xlim([0,50])
# plt.show()


# STEP 1: SIMULATE THE SYSTEM
toggle_switch_active = True
rrr_active = True
numpy_random_initial_condition_seed = 10
# Simulation Parameters
N_SIMULATIONS = 30
simulation_time = 100
sampling_time = 0.5
t = np.arange(0, simulation_time, sampling_time)
x0_init_ar = np.array([100.1,20.1])
x0_init_rrr = np.array([10.,10., 10.])
x0_init_ts = np.array([100.1,100.1])


# ToggleSwitch_ActRep_rrr
dict_output_params = {}
dict_output_params['Ar'] = {'act_state_no':0, 'rep_state_no':1, 'Vm': 2, 'KA':1, 'KB':0.4, 'nA':1, 'nB':1}
dict_output_params['rrr'] = {'act_state_no':3, 'rep_state_no':4, 'Vm':1, 'KA':0.02, 'KB':1, 'nA':1, 'nB':2}
dict_output_params['Ts'] = {'act_state_no':5, 'rep_state_no':6, 'Vm':1, 'KA':120, 'KB':1, 'nA':1, 'nB':3}
x0_init = np.array([])
x0_init = np.concatenate([x0_init, x0_init_ar])
x0_init = np.concatenate([x0_init, x0_init_rrr])
x0_init = np.concatenate([x0_init, x0_init_ts])
simulation_system = rrr

# Output parameters



np.random.seed(numpy_random_initial_condition_seed)
ls_seed_for_initial_condition = np.random.randint(0,10000,(N_SIMULATIONS))
ls_data = []
i=0
for i in range(N_SIMULATIONS):
    np.random.seed(ls_seed_for_initial_condition[i])
    x0_init_i = x0_init + np.random.uniform(0, 4, size=x0_init.shape)
    X = odeint(simulation_system,x0_init_i,t)
    # Make the Output
    Y = output_type_1(X[:, [dict_output_params['Ar']['act_state_no']]],X[:, [dict_output_params['Ar']['act_state_no']]],
                                         Vm=dict_output_params['Ar']['Vm'], KA=dict_output_params['Ar']['KA'],
                                         KB=dict_output_params['Ar']['KB'], nA=dict_output_params['Ar']['nA'],
                                         nB=dict_output_params['Ar']['nB'])
    # Y = np.concatenate([Y, output_type_1(X[:, 2:3], X[:, 3:4], Vm=1,KA=0.02,KB=1,nA=1,nB=2)], axis=1)
    Y = np.concatenate([Y, output_type_1(X[:, [dict_output_params['rrr']['act_state_no']]],
                                         X[:, [dict_output_params['rrr']['act_state_no']]],
                                         Vm=dict_output_params['rrr']['Vm'], KA=dict_output_params['rrr']['KA'],
                                         KB=dict_output_params['rrr']['KB'], nA=dict_output_params['rrr']['nA'],
                                         nB=dict_output_params['rrr']['nB'])], axis=1)
    Y = np.concatenate([Y, output_type_2(X[:, [dict_output_params['Ts']['act_state_no']]],
                                         Vm=dict_output_params['Ts']['Vm'], KA=dict_output_params['Ts']['KA'],
                                         KB=dict_output_params['Ts']['KB'], nA=dict_output_params['Ts']['nA'],
                                         nB=dict_output_params['Ts']['nB'])], axis=1)
    if (np.sum(X>500) > 0) or (np.sum(X==0)>5):
        print('SYSTEM BLEW UP')
        continue
    else:
        dict_data_i = {'XT': X, 'YT': Y}
        ls_data.append(dict_data_i)

Xs = (X -np.mean(X,axis=0))/np.std(X,axis=0)
f,ax = plt.subplots(2,2,figsize=(20,20))
ax = ax.reshape(-1)
ax[0].plot(t,X)
ax[0].set_title('Unscaled')
ax[1].plot(t,Xs)
ax[1].legend(['x1','x2','x3','x4','x5','x6', 'x7'], ncol=2, loc='upper right')
ax[1].set_title('Scaled')
ax[2].plot(t, Y)
ax[2].set_title('output')

try:
    XT_all = np.empty((0, ls_data[0]['XT'].shape[1]))
    for data_i in ls_data:
        XT_all = np.concatenate([XT_all, data_i['XT']], axis=0)
except:
    XT_all = np.empty((0, X.shape[1]))
    print('No data')

sb.heatmap(np.corrcoef(Xs.T), cmap='Blues', annot= True, ax=ax[3])
bottom, top = ax[3].get_ylim()
ax[3].set_ylim(bottom + 0.5, top - 0.5)
f.show()

# f,ax = plt.subplots(1,3,figsize=(15,5))
# for i in range(N_SIMULATIONS):
#     ax[0].plot(ls_data[i]['XT'][:, 0], ls_data[i]['XT'][:, 1])
#     ax[1].plot(ls_data[i]['XT'][:, 2], ls_data[i]['XT'][:, 3])
#     ax[2].plot(ls_data[i]['XT'][:, 4], ls_data[i]['XT'][:, 5])
# ax[0].set_title('Toggle_Switch')
# ax[1].set_title('Activator_Repressor')
# ax[2].set_title('rrr')
# # ax[2].set_ylim([0,1])
# # ax[2].set_xlim([0,8])
# f.show()

np.corrcoef(XT_all.T)

dict_simulation_parameters = {'numpy_random_initial_condition_seed': numpy_random_initial_condition_seed, 'simulation_time':simulation_time, 'sampling_time': sampling_time, 't':t, 'x0_init_ar':x0_init_ar, 'x0_init_rrr':x0_init_rrr, 'x0_init_ts':x0_init_ts}
## SYSTEM 6 - Toggle_Switch___Activator_Repressor - State x1:6 Output y1,y2,y3
dict_measurement_info_6 = {'state_measured': True, 'output_measured': True, 'n_delay_embedding': 1, 'ls_measured_output_indices': [0,1,2], 'ls_measured_state_indices': [0,1,2,3,4,5,6], 'formulate_Koopman_output_data_with_intersection': False}
dict_measurement_info_6.update(dict_simulation_parameters)
save_system(ls_data, dict_measurement_info = dict_measurement_info_6, SYSTEM_NO =15)


## Bash Script Generation

N_NODES_PER_OBSERVABLE = 4

dict_hp={}
dict_hp['ls_dict_size'] = [5] # [15,20,25]
dict_hp['ls_nn_layers'] = [4]
dict_hp['System_no'] = []
dict_hp['System_no'] = dict_hp['System_no'] + [15]
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(1,7))   #mt
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(11,13))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(21,29))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(31,40))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(41,50))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(51,60))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(61,70))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(71,80))

system_running = 'goldentensor'
# system_running = 'optictensor'
# system_running = 'microtensor'
# system_running = 'quantensor'

file = open('/Users/shara/Desktop/TS_AR_IFFL/' + system_running + '_run.sh','w')
if system_running in ['microtensor', 'quantensor', 'goldentensor', 'optictensor']:
    ls_device = [' \'/cpu:0\' ']
# elif system_running in ['goldentensor', 'optictensor']:
#     ls_device = [' \'/cpu:0\' ', ' \'/gpu:0\' ', ' \'/gpu:1\' ', ' \'/gpu:2\' ', ' \'/gpu:3\' ']

# For each system of interest
dict_system_next_run = {}
for system_no in dict_hp['System_no']:
    # Create a MyRunInfo folder
    runinfo_folder = 'System_' + str(system_no) + '/MyRunInfo'
    if not os.path.exists(runinfo_folder):
        os.mkdir(runinfo_folder)
    if not os.path.exists(runinfo_folder + '/dummy_proxy.txt'):
        with open(runinfo_folder + '/dummy_proxy.txt', 'w') as f:
            f.write('This is created so that git does not experience an issue with')
    # Get the latest run number for each system # TODO - Check this part of the code
    try:
        ls_all_run_files = os.listdir('System_' + str(system_no) + '/MyMac')
        ls_run_numbers = [np.int(i[4:]) for i in ls_all_run_files if 'RUN_' in i]
        next_run = np.int(np.max(ls_run_numbers)) + 1
    except:
        next_run = 0
    dict_system_next_run[system_no] = next_run

file.write('#!/bin/bash \n')
# file.write('rm nohup.out \n')
file.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] \n')
ls_all_runs = []
n_devices = len(ls_device)
for system_no,n_x,n_l in product(dict_hp['System_no'],dict_hp['ls_dict_size'],dict_hp['ls_nn_layers']):
    run_number = dict_system_next_run[system_no]
    device_name = ls_device[np.mod(run_number,n_devices)]
    # Check if run file exists
    run_info_file = ' > System_' + str(system_no) + '/MyRunInfo/Run_' + str(run_number) + '.txt & \n'
    n_n = np.int(np.ceil(n_x*N_NODES_PER_OBSERVABLE))
    file.write('python3 deepDMD.py' + device_name + str(system_no) + ' ' + str(run_number) + ' ' + str(n_x) + ' ' + str(n_l) + ' ' + str(n_n) + run_info_file)
    if device_name == ls_device[-1]:
        file.write('wait\n')
    # Incrementing to the next run
    dict_system_next_run[system_no] = dict_system_next_run[system_no] + 1
file.write('echo "All sessions are complete" \n')
file.write('echo "=======================================================" \n')
file.close()

