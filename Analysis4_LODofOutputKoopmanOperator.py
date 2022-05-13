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

SYSTEM_NO =23
RUN_NO = 7


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
n_delay_embedding = np.int(np.mod(SYSTEM_NO,10))
# Load the scalers
scaler_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle'
with open(scaler_file, 'rb') as handle:
    dict_Scaler = pickle.load(handle)
ZT_Scaler = dict_Scaler['XT']
# Load the tensorflow environment
sess = tf.InteractiveSession()
run_folder_name_i = 'System_' + str(SYSTEM_NO) + '/MyMac/RUN_' + str(RUN_NO)
dict_model = get_dict_param(run_folder_name_i, SYSTEM_NO, sess)
# TODO Add the WhT_num matrix to the model
dict_model['WhT_num'] = np.zeros((dict_model['KxT_num'].shape[0],n_outputs))
for i in range(n_outputs):
    dict_model['WhT_num'][n_delay_embedding*i,i] = 1


# Load the hyperparameters for each run
with open(run_folder_name_i + '/dict_hyperparameters.pickle', 'rb') as handle:
    dict_run_i_info = pickle.load(handle)

# Iterate through each dataset

ls_data_pred = []
for data_i in ls_data:
    # Find the initial condition
    zT0 = data_i['YT'][0:n_delay_embedding, dict_data_info['ls_measured_output_indices']].T.reshape(1, -1)
    # Organize the data
    ZT_true = np.empty((0, zT0.shape[1]))
    if dict_data_info['formulate_Koopman_output_data_with_intersection']:
        for j in range(n_delay_embedding, data_i['YT'].shape[0]):
            ZT_true = np.concatenate([ZT_true, data_i['YT'][j - n_delay_embedding:j,dict_data_info['ls_measured_output_indices']].T.reshape(1,-1)],axis=0)
    else:
        n_delay_embedded_points = np.int(np.floor(data_i['YT'].shape[0] / n_delay_embedding))
        for j in range(n_delay_embedded_points):
            ZT_true = np.concatenate([ZT_true, data_i['YT'][j * n_delay_embedding:(j + 1) * n_delay_embedding,dict_data_info['ls_measured_output_indices']].T.reshape(1,-1)], axis=0)
    # Scale the data
    zT0s = ZT_Scaler.transform(zT0)
    ZTs_true = ZT_Scaler.transform(ZT_true)
    # Generate the observables
    psiZT_true = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: ZTs_true})
    # Generate 1-step predictions
    psiZT_est_1step = np.concatenate([psiZT_true[0:1,:],np.matmul(psiZT_true[0:-1, :], dict_model['KxT_num'])],axis=0)
    # Generate n-step predictions
    psiZT_est_nstep = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: zT0s})
    for step_i in range(ZT_true.shape[0] - 1):
        psiZT_est_nstep = np.concatenate([psiZT_est_nstep, np.matmul(psiZT_est_nstep[-1:, :], dict_model['KxT_num'])], axis=0)
    # Get the states back from the observables
    ZTs_est_1step = psiZT_est_1step[:, 0:zT0.shape[1]]
    ZTs_est_nstep = psiZT_est_nstep[:, 0:zT0.shape[1]]
    # Reverse scale the data
    ZT_est_1step = ZT_Scaler.inverse_transform(ZTs_est_1step)
    ZT_est_nstep = ZT_Scaler.inverse_transform(ZTs_est_nstep)
    # Reverse the delay embedding
    YT_true = np.empty((0, n_outputs))
    YT_est_1step = np.empty((0, n_outputs))
    YT_est_nstep = np.empty((0, n_outputs))
    # YT_true = XT_true[0:,:].reshape(n_outputs,-1).T
    # YT_est_1step = XT_est_1step[0:,:].reshape(n_outputs,-1).T
    # YT_est_nstep = XT_est_nstep[0:,:].reshape(n_outputs,-1).T
    if dict_data_info['formulate_Koopman_output_data_with_intersection']:
        for j in range(ZT_true.shape[0]):
            YT_true = np.concatenate([YT_true, ZT_true[j,:].reshape(n_outputs,-1).T[-1:,:]], axis=0)
            YT_est_1step = np.concatenate([YT_est_1step, ZT_est_1step[j,:].reshape(n_outputs, -1).T[-1:,:]], axis=0)
            YT_est_nstep = np.concatenate([YT_est_nstep, ZT_est_nstep[j,:].reshape(n_outputs, -1).T[-1:,:]], axis=0)
    else:
        for j in range(ZT_true.shape[0]):
            YT_true = np.concatenate([YT_true, ZT_true[j,:].reshape(n_outputs,-1).T], axis=0)
            YT_est_1step = np.concatenate([YT_est_1step, ZT_est_1step[j,:].reshape(n_outputs, -1).T], axis=0)
            YT_est_nstep = np.concatenate([YT_est_nstep, ZT_est_nstep[j,:].reshape(n_outputs, -1).T], axis=0)
    dict_data_pred_i = {
        'YT_true': YT_true,
        'YT_est_1step': YT_est_1step,
        'YT_est_nstep': YT_est_nstep,
        'ZT_true': ZT_true,
        'ZT_est_1step': ZT_est_1step,
        'ZT_est_nstep': ZT_est_nstep,
        'psiZT_true': psiZT_true,
        'psiZT_est_1step': psiZT_est_1step,
        'psiZT_est_nstep': psiZT_est_nstep
    }
    ls_data_pred.append(dict_data_pred_i)

## The Observable Decomposition

ls_output_index = [0]
psi_o_tolerence = 0.9

WhT = dict_model['WhT_num'][:,ls_output_index]
KT = dict_model['KxT_num']
nL = KT.shape[0]
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
        psiZTi = data_i['psiZT_est_nstep']
        psiZTi_ou = psiZTi @ T.T
        psiZTi_o = psiZTi_ou[:,0:nPC]
        YTi_o_nstep_scaled = psiZTi_o @ Who.T
        psiZTi_hat = psiZTi_o @ Tinv[:, 0:nPC].T
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

f,ax = plt.subplots(2,2,figsize=(10,10))
sb.heatmap(Ka,cmap = 'Blues',ax = ax[0,0])
sb.heatmap(Wha,cmap = 'Blues',ax = ax[0,1])
sb.heatmap(Ko,cmap = 'Blues',ax = ax[1,0])
sb.heatmap(Who,cmap = 'Blues',ax = ax[1,1])
plt.show()







