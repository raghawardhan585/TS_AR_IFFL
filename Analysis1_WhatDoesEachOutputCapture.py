import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import product
import random
import seaborn as sb
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import copy
import pickle
plt.rcParams["font.family"] = "Avenir"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
# from Simulated_HighDimensional_Systems import *
import os
import shutil
import tensorflow as tf

def get_dict_param(run_folder_name_curr,SYS_NO,sess):
    dict_p = {}
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name_curr + '/System_' + str(SYS_NO) + '_DeepDMDdata_Scaled.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name_curr))
    try:
        psixpT = tf.get_collection('psixpT')[0]
        psixfT = tf.get_collection('psixfT')[0]
        xpT_feed = tf.get_collection('xpT_feed')[0]
        xfT_feed = tf.get_collection('xfT_feed')[0]
        KxT = tf.get_collection('KxT')[0]
        KxT_num = sess.run(KxT)
        dict_p['psixpT'] = psixpT
        dict_p['psixfT'] = psixfT
        dict_p['xpT_feed'] = xpT_feed
        dict_p['xfT_feed'] = xfT_feed
        dict_p['KxT_num'] = KxT_num
    except:
        print('State info not found')
    try:
        ypT_feed = tf.get_collection('ypT_feed')[0]
        yfT_feed = tf.get_collection('yfT_feed')[0]
        dict_p['ypT_feed'] = ypT_feed
        dict_p['yfT_feed'] = yfT_feed
        WhT = tf.get_collection('WhT')[0]
        WhT_num = sess.run(WhT)
        dict_p['WhT_num'] = WhT_num
    except:
        print('No output info found')
    return dict_p

# Plot the observability matrix
def plot_observability_matrix(O):
    f,ax = plt.subplots(2,1, figsize = (5,10))
    sb.heatmap(O,cmap='Blues', ax=ax[0])
    ax[0].set_title('Original observability matrix')
    U, S, VT = np.linalg.svd(O)
    ax[1].plot(np.arange(1, 1 + len(S)), 100 - np.cumsum(S ** 2) / np.sum(S ** 2) * 100)
    ax[1].set_title('Scree plot of singular values \n of Observability matrix')
    f.show()
    return

# Plot the sytem matrices before and after decomposition
def plot_LOD_Koopman_matrices(K,Ka,Wh,Wha):
    f,ax = plt.subplots(2,2, figsize=(10,10))
    sb.heatmap(Ka,cmap='Blues', ax=ax[0,1])
    sb.heatmap(K,cmap='Blues', ax=ax[0,0])
    ax[0,0].set_title('Original K')
    ax[0,1].set_title('Linear observable \n decomposition of K')
    sb.heatmap(Wh,cmap='Blues', ax=ax[1,0])
    sb.heatmap(Wha,cmap='Blues', ax=ax[1,1])
    ax[1,0].set_title('Original Wh')
    ax[1,1].set_title('Linear observable \n decomposition of Wh')
    f.show()
    return


# Import the best run for the system of interest
# Output should always be available
SYSTEM_NO =14 # Between [2,4,6]

# try:
#     # Import the computed result  statistics
#     with open('model_statistics_dataframe.pickle', 'rb') as handle:
#         df_results = pickle.load(handle)
#     df_results = df_results.loc[df_results['system'] ==SYSTEM_NO,]
#     df_series_important_stat = (df_results['r2_train_nstep'] + df_results['r2_valid_nstep'])/2
#     # computed_best_avg_accuracy = np.max((df_results['r2_train_nstep'] + df_results['r2_valid_nstep'])/2)
#     dict_results = df_results.loc[df_series_important_stat == np.max(df_series_important_stat),:].iloc[0:1,:].T.to_dict()
#     dict_results = dict_results[list(dict_results.keys())[0]]
#     print('======================')
#     print('BEST MODEL STATISTICS')
#     print('======================')
#     for result_name in dict_results.keys():
#         print(result_name + ' : ' + str(dict_results[result_name]))
#     RUN_NO = dict_results['run_no']
# except:
#     print('INFO: The Result for the current system under consideration does NOT EXIST')

RUN_NO = 2

# Load the dataset
simulation_data_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle'
with open(simulation_data_file, 'rb') as handle:
    ls_data = pickle.load(handle)
system_runs_folder = 'System_' + str(SYSTEM_NO) + '/MyMac'
# Load the simulation info [required to convert the data to the required format]
simulation_datainfo_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_SimulatedDataInfo.pickle'
with open(simulation_datainfo_file, 'rb') as handle:
    dict_data_info = pickle.load(handle)
n_states = len(dict_data_info['ls_measured_state_indices'])
n_outputs = len(dict_data_info['ls_measured_output_indices'])
# Load the scalers
scaler_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle'
with open(scaler_file, 'rb') as handle:
    dict_Scaler = pickle.load(handle)



# Load the tensorflow environment
sess_i = tf.InteractiveSession()
run_folder_name_i = 'System_' + str(SYSTEM_NO) + '/MyMac/RUN_' + str(RUN_NO)
dict_model = get_dict_param(run_folder_name_i, SYSTEM_NO, sess_i)
# Load the hyperparameters for each run
with open(run_folder_name_i + '/dict_hyperparameters.pickle', 'rb') as handle:
    dict_run_i_info = pickle.load(handle)

# Iterate through each dataset
ls_data_pred = []
for data_i in ls_data:
    # Data
    XT_true = data_i['XT'][:, dict_data_info['ls_measured_state_indices']]
    YT_true = data_i['YT'][:, dict_data_info['ls_measured_output_indices']]
    # Scale the data
    XTs_true = dict_Scaler['XT'].transform(XT_true)
    # Generate the observables
    psiXT_true = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: XTs_true})
    # Generate 1-step predictions
    psiXT_est_1step = np.concatenate([psiXT_true[0:1, :], np.matmul(psiXT_true[0:-1, :], dict_model['KxT_num'])],axis=0)
    YTs_est_1step = psiXT_est_1step @ dict_model['WhT_num']  # 1 - step
    # Generate n-step predictions
    psiXT_est_nstep = psiXT_true[0:1, :]
    for step_i in range(XT_true.shape[0] - 1):
        psiXT_est_nstep = np.concatenate([psiXT_est_nstep, psiXT_est_nstep[-1:, :] @ dict_model['KxT_num']],axis=0)
    YTs_est_nstep = psiXT_est_nstep @ dict_model['WhT_num']  # n - step
    # Get the states back from the observables
    XTs_est_1step = psiXT_est_1step[:, 0:n_states]
    XTs_est_nstep = psiXT_est_nstep[:, 0:n_states]
    # Reverse scale the data
    XT_est_1step = dict_Scaler['XT'].inverse_transform(XTs_est_1step)
    XT_est_nstep = dict_Scaler['XT'].inverse_transform(XTs_est_nstep)
    YT_est_1step = dict_Scaler['YT'].inverse_transform(YTs_est_1step)
    YT_est_nstep = dict_Scaler['YT'].inverse_transform(YTs_est_nstep)
    ls_data_pred.append({
        'XT_true': XT_true,
        'XT_est_1step': XT_est_1step,
        'XT_est_nstep': XT_est_nstep,
        'psiXT_true': psiXT_true,
        'psiXT_est_1step': psiXT_est_1step,
        'psiXT_est_nstep': psiXT_est_nstep,
        'YT_true': YT_true,
        'YT_est_1step': YT_est_1step,
        'YT_est_nstep': YT_est_nstep
    })


# Find the accuracy of the model

XT_1step_est_all = np.empty((0,n_states))
XT_nstep_est_all = np.empty((0,n_states))
XT_nstep_all = np.empty((0,n_states))
YT_1step_est_all = np.empty((0,n_outputs))
YT_nstep_est_all = np.empty((0,n_outputs))
YT_nstep_all = np.empty((0,n_outputs))
for data_i in ls_data_pred:
    XT_nstep_all = np.concatenate([XT_nstep_all,data_i['XT_true']], axis=0)
    XT_nstep_est_all = np.concatenate([XT_nstep_est_all, data_i['XT_est_nstep']], axis=0)
    XT_1step_est_all = np.concatenate([XT_1step_est_all, data_i['XT_est_1step']], axis=0)
    YT_nstep_all = np.concatenate([YT_nstep_all, data_i['YT_true']], axis=0)
    YT_nstep_est_all = np.concatenate([YT_nstep_est_all, data_i['YT_est_nstep']], axis=0)
    YT_1step_est_all = np.concatenate([YT_1step_est_all, data_i['YT_est_1step']], axis=0)
print("[1 - step] State Prediction Accuracy : ", r2_score(XT_nstep_all, XT_1step_est_all))
print("[1 - step] Output Prediction Accuracy : ", r2_score(YT_nstep_all, YT_1step_est_all))
print("[n - step] State Prediction Accuracy : ", r2_score(XT_nstep_all, XT_nstep_est_all))
print("[n - step] Output Prediction Accuracy : ", r2_score(YT_nstep_all, YT_nstep_est_all))


#

# TODO - the observability stuff same as in the first example - one sensitivity plot for each output
ls_output_index = [0]
psi_o_tolerence = 0.9

nL = psiXT_true.shape[1]
WhT = dict_model['WhT_num'][:,ls_output_index]
KT = dict_model['KxT_num']
# Construct the observability matrix
OT = np.empty((WhT.shape[0],0))
for i in range(nL):
    OT = np.concatenate([OT, np.linalg.matrix_power(KT,i) @ WhT], axis=1)
O = OT.T
# Decomposition of the observability matrix
U,S,VT = np.linalg.svd(O)
V = VT.T
# Transformation
T = VT
Tinv = V
K = KT.T
Wh = WhT.T
# Converted matrix
Ka = T @ K @ np.linalg.inv(T)
Wha = Wh @ np.linalg.inv(T)


# plot_observability_matrix(O)
# plot_LOD_Koopman_matrices(K,Ka,Wh,Wha)

#
# Estimate the optimal number of observable states
ls_nPC = np.arange(1,len(S)+1,1)
ls_output_accuracy = []
for i in range(len(ls_nPC)):
    nPC = ls_nPC[i]
    Ko = Ka[0:nPC,0:nPC]
    Who = Wha[:,0:nPC]
    ls_transformed_data = []
    YT_actual = YT_pred_scaled = XT_all = []
    for data_i in ls_data_pred:
        psiXTi = data_i['psiXT_est_nstep']
        psiXTi_ou = psiXTi @ T.T
        psiXTi_o = psiXTi_ou[:,0:nPC]
        YTi_o_nstep_scaled = psiXTi_o @ Who.T
        psiXTi_hat = psiXTi_o @ Tinv[:, 0:nPC].T
        try:
            YT_pred_scaled = np.concatenate([YT_pred_scaled, YTi_o_nstep_scaled], axis=0)
            YT_actual = np.concatenate([YT_actual, data_i['YT_est_nstep'][:,ls_output_index]], axis=0)
            XT_all = np.concatenate([XT_all, data_i['XT_est_nstep']], axis=0)
        except:
            YT_pred_scaled = YTi_o_nstep_scaled
            YT_actual = data_i['YT_est_nstep'][:,ls_output_index]
            XT_all = data_i['XT_est_nstep']
    XTs_all = dict_Scaler['XT'].transform(XT_all)
    # Special inverse transform
    YT_pred_scaled_intermediate = np.zeros((XTs_all.shape[0], n_outputs))
    YT_pred_scaled_intermediate[:,ls_output_index] = YT_pred_scaled
    YT_pred = dict_Scaler['YT'].inverse_transform(YT_pred_scaled_intermediate)[:,ls_output_index]
    ls_output_accuracy.append(r2_score(YT_actual, YT_pred, multioutput='uniform_average'))
    print('# states : ', ls_nPC[i], ' r2 :', ls_output_accuracy[i])
nPC_opt = ls_nPC[np.where(np.array(ls_output_accuracy)>psi_o_tolerence)[0][0]]
print('The optimal number of observed states is : ', nPC_opt)

#
# nPC_opt = 1
Ko = Ka[0:nPC_opt, 0:nPC_opt]
Who = Wha[:, 0:nPC_opt]

T_tensor = tf.constant(T[0:nPC_opt,:], dtype=tf.float32)
psi_oT_tensor = tf.matmul(dict_model['psixpT'], tf.transpose(T_tensor))
psi_o_sensitivity_matrix = np.empty((0,ls_data_pred[0]['XT_est_nstep'].shape[1]))
for i in range(nPC_opt):
    func = psi_oT_tensor[:,i:i+1]
    func_grad = tf.gradients(func, dict_model['xpT_feed'])
    sensitivity_all_points_for_func = func_grad[0].eval(feed_dict={dict_model['xpT_feed']: XTs_all})
    psi_o_sensitivity_matrix= np.concatenate([psi_o_sensitivity_matrix ,np.max(sensitivity_all_points_for_func,axis=0).reshape(1,-1)],axis=0)

#
ls_xticks = []
for i in range(ls_data_pred[0]['XT_true'].shape[1]):
    ls_xticks.append('$x_{' +str(i+1)+'}$')
for i in range(ls_data_pred[0]['psiXT_true'].shape[1] - ls_data_pred[0]['XT_true'].shape[1]-1):
    ls_xticks.append('$\\varphi_{' + str(i+1) + '}$')
ls_xticks.append('$1$')
ls_psi_o_labels = []
for i in range(nPC_opt):
    ls_psi_o_labels.append('$\psi_{o'+str(i+1)+ '}$')


f,((ax0,dummy_ax),(ax1,cbar_ax1)) = plt.subplots(2,2,figsize=(8,10), sharex='col', gridspec_kw={'height_ratios': [2, 4], 'width_ratios': [20, 1]})
sb.heatmap(psi_o_sensitivity_matrix, cmap='Blues',ax=ax1, cbar_ax=cbar_ax1)

x_tick_pos = np.arange(0.5,ls_data_pred[0]['XT_true'].shape[1])

ax1.set_xticks(x_tick_pos)
ax1.set_xticklabels(ls_xticks[0:ls_data_pred[0]['XT_true'].shape[1]])
ax1.set_yticks(np.arange(0.5,nPC_opt))
ax1.set_yticklabels(ls_psi_o_labels,rotation=0)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)

# Energy plot of psi_i(x) contributing to

ax0.bar(x_tick_pos, np.linalg.norm(psi_o_sensitivity_matrix, axis=0, ord=2))
# ax0.set_title('Energy of each observable')
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
dummy_ax.axis('off')
plt.tight_layout()
f.show()

# Close the tensorflow environment
tf.reset_default_graph()
sess_i.close()

##
plt.plot(psiXTi_o[:,0:nPC_opt])
plt.show()
##

XT_all = dict_Scaler['XT'].inverse_transform(XTs_all)
plt.figure(figsize=(10,10))
ax = sb.heatmap(np.corrcoef(XT_all.T),cmap='Blues', annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()