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
import os
import shutil
import tensorflow as tf
import math

def weight_variable(shape):
    std_dev = math.sqrt(3.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32))
def bias_variable(shape):
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32))
def initialize_Wblist(n_u, hv_list):
    # INITIALIZATION - going from input to first layer
    W_list = [weight_variable([n_u, hv_list[0]])]
    b_list = [bias_variable([hv_list[0]])]
    # PROPAGATION - consecutive layers
    for k in range(1,len(hv_list)):
        W_list.append(weight_variable([hv_list[k - 1], hv_list[k]]))
        b_list.append(bias_variable([hv_list[k]]))
    return W_list, b_list


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


SYSTEM_NO = 20


# Get all the delay embeddings
ls_delay_embeddings = [sys_name[8:] for sys_name in os.listdir() if sys_name[0:8] == 'System_'+ str(np.int(np.floor(SYSTEM_NO/10)))]
ls_delay_embeddings = np.sort([np.int(i) for i in ls_delay_embeddings if not i == ''])
# Load the dataset
SYSTEM_NO = np.int(np.floor(SYSTEM_NO/10)*10)
simulation_data_file = 'System_' + str(SYSTEM_NO + 1) + '/System_' + str(SYSTEM_NO + 1) + '_SimulatedData.pickle'
with open(simulation_data_file, 'rb') as handle:
    ls_data = pickle.load(handle)

##

model_no = 1
dict_model_stats = {}
# Iterate through each delay embedding
for n_delay_embedding in ls_delay_embeddings:
    delay_embedded_system_no = SYSTEM_NO + n_delay_embedding
    system_runs_folder = 'System_' + str(delay_embedded_system_no) + '/MyMac'
    # Load the simulation info [required to convert the data to the required format]
    simulation_datainfo_file = 'System_' + str(delay_embedded_system_no) + '/System_' + str(delay_embedded_system_no) + '_SimulatedDataInfo.pickle'
    with open(simulation_datainfo_file, 'rb') as handle:
        dict_data_info = pickle.load(handle)
    n_outputs = len(dict_data_info['ls_measured_output_indices'])
    # Load the scalers
    scaler_file = 'System_' + str(delay_embedded_system_no) + '/System_' + str(delay_embedded_system_no) + '_DataScaler.pickle'
    with open(scaler_file, 'rb') as handle:
        dict_Scaler = pickle.load(handle)
    # Get the exhaustive list of runs
    ls_runs = np.sort([np.int(items[4:]) for items in os.listdir(system_runs_folder) if items[0:4] == 'RUN_'])
    # Iterate through each of the runs and each dataset
    for run_i in ls_runs:
        # Load the tensorflow environment
        sess_i = tf.InteractiveSession()
        run_folder_name_i = 'System_' + str(delay_embedded_system_no) + '/MyMac/RUN_' + str(run_i)
        dict_model_i = get_dict_param(run_folder_name_i, delay_embedded_system_no, sess_i)
        # Load the hyperparameters for each run
        with open(run_folder_name_i + '/dict_hyperparameters.pickle', 'rb') as handle:
            dict_run_i_info = pickle.load(handle)
        # Iterate through each dataset
        ls_data_pred = []
        for data_i in ls_data:
            # Find the initial condition
            x0 = data_i['YT'][0:n_delay_embedding, dict_data_info['ls_measured_output_indices']].T.reshape(1, -1)
            # Organize the data
            XT_true = np.empty((0, x0.shape[1]))
            if dict_data_info['formulate_Koopman_output_data_with_intersection']:
                for j in range(n_delay_embedding, data_i['YT'].shape[0]):
                    XT_true = np.concatenate([XT_true, data_i['YT'][j - n_delay_embedding:j,
                                                       dict_data_info['ls_measured_output_indices']].T.reshape(1,-1)],axis=0)
            else:
                n_delay_embedded_points = np.int(np.floor(data_i['YT'].shape[0] / n_delay_embedding))
                for j in range(n_delay_embedded_points):
                    XT_true = np.concatenate([XT_true, data_i['YT'][j * n_delay_embedding:(j + 1) * n_delay_embedding,
                                                   dict_data_info['ls_measured_output_indices']].T.reshape(1,-1)], axis=0)
            # Scale the data
            x0s = dict_Scaler['XT'].transform(x0)
            XTs_true = dict_Scaler['XT'].transform(XT_true)
            # Generate the observables
            psiXT_true = dict_model_i['psixpT'].eval(feed_dict={dict_model_i['xpT_feed']: XTs_true})
            # Generate 1-step predictions
            psiXT_est_1step = np.concatenate([psiXT_true[0:1,:],np.matmul(psiXT_true[0:-1, :], dict_model_i['KxT_num'])],axis=0)
            # Generate n-step predictions
            psiXT_est_nstep = dict_model_i['psixpT'].eval(feed_dict={dict_model_i['xpT_feed']: x0s})
            for step_i in range(XT_true.shape[0] - 1):
                psiXT_est_nstep = np.concatenate([psiXT_est_nstep, np.matmul(psiXT_est_nstep[-1:, :], dict_model_i['KxT_num'])], axis=0)
            # Get the states back from the observables
            XTs_est_1step = psiXT_est_1step[:, 0:x0.shape[1]]
            XTs_est_nstep = psiXT_est_nstep[:, 0:x0.shape[1]]
            # Reverse scale the data
            XT_est_1step = dict_Scaler['XT'].inverse_transform(XTs_est_1step)
            XT_est_nstep = dict_Scaler['XT'].inverse_transform(XTs_est_nstep)
            # Reverse the delay embedding
            YT_true = np.empty((0, n_outputs))
            YT_est_1step = np.empty((0, n_outputs))
            YT_est_nstep = np.empty((0, n_outputs))
            # YT_true = XT_true[0:,:].reshape(n_outputs,-1).T
            # YT_est_1step = XT_est_1step[0:,:].reshape(n_outputs,-1).T
            # YT_est_nstep = XT_est_nstep[0:,:].reshape(n_outputs,-1).T
            if dict_data_info['formulate_Koopman_output_data_with_intersection']:
                for j in range(XT_true.shape[0]):
                    YT_true = np.concatenate([YT_true, XT_true[j,:].reshape(n_outputs,-1).T[-1:,:]], axis=0)
                    YT_est_1step = np.concatenate([YT_est_1step, XT_est_1step[j,:].reshape(n_outputs, -1).T[-1:,:]], axis=0)
                    YT_est_nstep = np.concatenate([YT_est_nstep, XT_est_nstep[j,:].reshape(n_outputs, -1).T[-1:,:]], axis=0)
            else:
                for j in range(XT_true.shape[0]):
                    YT_true = np.concatenate([YT_true, XT_true[j,:].reshape(n_outputs,-1).T], axis=0)
                    YT_est_1step = np.concatenate([YT_est_1step, XT_est_1step[j,:].reshape(n_outputs, -1).T], axis=0)
                    YT_est_nstep = np.concatenate([YT_est_nstep, XT_est_nstep[j,:].reshape(n_outputs, -1).T], axis=0)
            dict_data_pred_i = {
                'YT_true': YT_true,
                'YT_est_1step': YT_est_1step,
                'YT_est_nstep': YT_est_nstep,
                'psiXT_true': XTs_true,
                'psiXT_est_1step': psiXT_est_1step,
                'psiXT_est_nstep': psiXT_est_nstep
            }
            ls_data_pred.append(dict_data_pred_i)
        # Record the run statistics
        # *Assumed training, validation and test are split equally
        dict_model_stats[model_no] = {'system': SYSTEM_NO, 'n_delay_embedding': n_delay_embedding, 'system_no': delay_embedded_system_no, 'run_no': run_i, 'total_observables': n_outputs*n_delay_embedding + dict_run_i_info['x_obs']}
        dict_model_stats[model_no].update(dict_run_i_info)
        # Split the data as training, validaiton and test
        dict_data_run_i = {}
        for item in ['train','valid','test']:
            dict_data_run_i[item] = {'true': np.empty((0,n_outputs)),'1-step': np.empty((0,n_outputs)),'n-step': np.empty((0,n_outputs))}
        n_datasets = len(ls_data_pred)
        for i in range(n_datasets):
            if i <= (n_datasets/3):
                item = 'train'
            elif i <= (2*n_datasets/3):
                item = 'valid'
            else:
                item = 'test'
            dict_data_run_i[item]['true'] = np.concatenate([dict_data_run_i[item]['true'], ls_data_pred[i]['YT_true']],axis=0)
            dict_data_run_i[item]['1-step'] = np.concatenate([dict_data_run_i[item]['1-step'], ls_data_pred[i]['YT_est_1step']], axis=0)
            dict_data_run_i[item]['n-step'] = np.concatenate([dict_data_run_i[item]['n-step'], ls_data_pred[i]['YT_est_nstep']], axis=0)
        # Get the full model training, validation and test scores for 1-step and n-step
        for item in ['train','valid','test']:
            dict_model_stats[model_no]['r2_' + str(item) + '_1step'] = r2_score(dict_data_run_i[item]['true'],dict_data_run_i[item]['1-step'])
            dict_model_stats[model_no]['r2_' + str(item) + '_nstep'] = r2_score(dict_data_run_i[item]['true'],dict_data_run_i[item]['n-step'])
        # # Get the 1-step and n-step prediction errors for individual outputs as well
        # dict_data_run_i_all = {}
        # for item in ['true','1-step','n-step']:
        #     dict_data_run_i_all[item] = np.concatenate([np.concatenate([dict_data_run_i['train'][item],dict_data_run_i['valid'][item]],axis=0),dict_data_run_i['test'][item]],axis=0)
        # for i in range(ls_data[0]['YT'].shape[1]):
        #     if i in dict_data_info['ls_measured_output_indices']:
        #         dict_model_stats[model_no]['r2_y' + str(i)+'_1step'] = r2_score(dict_data_run_i_all['true'],dict_data_run_i_all['1-step'])
        #         dict_model_stats[model_no]['r2_y' + str(i) + '_nstep'] = r2_score(dict_data_run_i_all['true'],dict_data_run_i_all['n-step'])
        #     else:
        #         dict_model_stats[model_no]['r2_y' + str(i) + '_1step'] = np.nan
        #         dict_model_stats[model_no]['r2_y' + str(i) + '_nstep'] = np.nan
        model_no = model_no + 1
        # Undo the tensorflow session
        tf.reset_default_graph()
        sess_i.close()

df_results = pd.DataFrame(dict_model_stats).T
##
HARDCODE_DS_FACTOR = 3
HARDCODED_SAMPLING_TIME = 0.5
SINGLE_PLOT = True
for data_index in [21]:# range(30):
    data_now = ls_data[data_index]
    t = HARDCODED_SAMPLING_TIME*np.arange(0,len(data_now['XT']))
    if SINGLE_PLOT:
        f,ax = plt.subplots(1,1,figsize=(6.5,5))
    else:
        f,ax = plt.subplots(1,3,figsize=(15,5))
    # Best run for each delay
    for n_delay_embedding in ls_delay_embeddings[3:4]:
        df_results_i = df_results.loc[df_results['n_delay_embedding'] == n_delay_embedding,:]
        best_accuracy = np.max(df_results_i['r2_train_nstep'] + df_results_i['r2_valid_nstep'])
        dict_result_i = df_results_i.loc[df_results_i['r2_train_nstep'] + df_results_i['r2_valid_nstep'] == best_accuracy,:].T.to_dict()
        dict_result_i = dict_result_i[list(dict_result_i.keys())[0]]
        # Get info of best run
        system_i = dict_result_i['system_no']
        run_i = dict_result_i['run_no']
        # Load the simulation info [required to convert the data to the required format]
        simulation_datainfo_file = 'System_' + str(system_i) + '/System_' + str(system_i) + '_SimulatedDataInfo.pickle'
        with open(simulation_datainfo_file, 'rb') as handle:
            dict_data_info = pickle.load(handle)
        n_outputs = len(dict_data_info['ls_measured_output_indices'])
        # Load the scalers
        scaler_file = 'System_' + str(system_i) + '/System_' + str(system_i) + '_DataScaler.pickle'
        with open(scaler_file, 'rb') as handle:
            dict_Scaler = pickle.load(handle)
        Z_Scaler = dict_Scaler['XT']
        # Load the tensorflow environment
        sess_i = tf.InteractiveSession()
        run_folder_name_i = 'System_' + str(system_i) + '/MyMac/RUN_' + str(run_i)
        dict_model_i = get_dict_param(run_folder_name_i, system_i, sess_i)
        # Find the initial condition
        zT0 = data_now['YT'][0:n_delay_embedding, dict_data_info['ls_measured_output_indices']].T.reshape(1, -1)
        # Organize the data
        ZT_true = np.empty((0, zT0.shape[1]))
        if dict_data_info['formulate_Koopman_output_data_with_intersection']:
            for j in range(n_delay_embedding, data_now['YT'].shape[0]):
                ZT_true = np.concatenate([ZT_true, data_now['YT'][j - n_delay_embedding:j,
                                                   dict_data_info['ls_measured_output_indices']].T.reshape(1, -1)],
                                         axis=0)
        else:
            n_delay_embedded_points = np.int(np.floor(data_now['YT'].shape[0] / n_delay_embedding))
            for j in range(n_delay_embedded_points):
                ZT_true = np.concatenate([ZT_true, data_now['YT'][j * n_delay_embedding:(j + 1) * n_delay_embedding,
                                                   dict_data_info['ls_measured_output_indices']].T.reshape(1, -1)],axis=0)
        # Scale the data
        zT0s = Z_Scaler.transform(zT0)
        ZTs_true = Z_Scaler.transform(ZT_true)
        # Generate the observables
        psiZT_true = dict_model_i['psixpT'].eval(feed_dict={dict_model_i['xpT_feed']: ZTs_true})
        # Generate 1-step predictions
        psiZT_est_1step = np.concatenate([psiZT_true[0:1, :], np.matmul(psiZT_true[0:-1, :], dict_model_i['KxT_num'])], axis=0)
        # Generate n-step predictions
        psiZT_est_nstep = dict_model_i['psixpT'].eval(feed_dict={dict_model_i['xpT_feed']: zT0s})
        for step_i in range(ZT_true.shape[0] - 1):
            psiZT_est_nstep = np.concatenate([psiZT_est_nstep, np.matmul(psiZT_est_nstep[-1:, :], dict_model_i['KxT_num'])], axis=0)
        # Get the states back from the observables
        ZTs_est_1step = psiZT_est_1step[:, 0:zT0.shape[1]]
        ZTs_est_nstep = psiZT_est_nstep[:, 0:zT0.shape[1]]
        # Reverse scale the data
        ZT_est_1step = Z_Scaler.inverse_transform(ZTs_est_1step)
        ZT_est_nstep = Z_Scaler.inverse_transform(ZTs_est_nstep)
        # Reverse the delay embedding
        YT_true = np.empty((0, n_outputs))
        YT_est_1step = np.empty((0, n_outputs))
        YT_est_nstep = np.empty((0, n_outputs))
        if dict_data_info['formulate_Koopman_output_data_with_intersection']:
            for j in range(ZT_true.shape[0]):
                YT_true = np.concatenate([YT_true, ZT_true[j, :].reshape(n_outputs, -1).T[-1:, :]], axis=0)
                YT_est_1step = np.concatenate([YT_est_1step, ZT_est_1step[j, :].reshape(n_outputs, -1).T[-1:, :]], axis=0)
                YT_est_nstep = np.concatenate([YT_est_nstep, ZT_est_nstep[j, :].reshape(n_outputs, -1).T[-1:, :]], axis=0)
        else:
            for j in range(ZT_true.shape[0]):
                YT_true = np.concatenate([YT_true, ZT_true[j, :].reshape(n_outputs, -1).T], axis=0)
                YT_est_1step = np.concatenate([YT_est_1step, ZT_est_1step[j, :].reshape(n_outputs, -1).T], axis=0)
                YT_est_nstep = np.concatenate([YT_est_nstep, ZT_est_nstep[j, :].reshape(n_outputs, -1).T], axis=0)

        if SINGLE_PLOT:
            ax.plot([], [], linestyle = 'None', marker = 's', markersize = 10, color='#F88A00', label='y1')
            ax.plot([], [], '.', markersize = 13, color = 'grey', label = 'simulation\ndata')
            ax.plot([], [], linestyle = 'None', marker = 's', markersize = 10, color='#2657AF', label='y2')
            ax.plot([], [], linestyle = '--', color = 'grey', label='1-step\n prediction')
            ax.plot([], [],linestyle = 'None',  marker = 's', markersize = 10, color='#2D9290', label='y3')
            ax.plot([], [], color = 'grey', label='n-step\nprediction')

            ax.plot(t[0:-1:HARDCODE_DS_FACTOR], YT_true[:, 0][0:-1:HARDCODE_DS_FACTOR], '.', color='#F88A00', markersize = 13)
            ax.plot(t[0:-1:3*HARDCODE_DS_FACTOR], YT_true[:, 1][0:-1:3*HARDCODE_DS_FACTOR], '.', color='#2657AF', markersize = 13)
            ax.plot(t[0:-1:3*HARDCODE_DS_FACTOR], YT_true[:, 2][0:-1:3*HARDCODE_DS_FACTOR], '.', color='#2D9290', markersize = 13)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        else:
            if n_delay_embedding == 1:
                ax[0].plot(t, YT_true[:, 0], '.', color='#F88A00')
                ax[1].plot(t, YT_true[:, 1], '.', color='#2657AF')
                ax[2].plot(t, YT_true[:, 2], '.', color='#2D9290')
        if SINGLE_PLOT:
            ax.plot(t, YT_est_1step[:, 0], color='#F88A00', linestyle = '--')
            ax.plot(t, YT_est_1step[:, 1], color='#2657AF', linestyle = '--')
            ax.plot(t, YT_est_1step[:, 2], color='#2D9290', linestyle = '--')
            ax.plot(t, YT_est_nstep[:, 0], color='#F88A00')
            ax.plot(t, YT_est_nstep[:, 1], color='#2657AF')
            ax.plot(t, YT_est_nstep[:, 2], color='#2D9290')
        else:
            ax[0].plot(t, YT_est_nstep[:, 0], color='#F88A00')
            ax[1].plot(t, YT_est_nstep[:, 1], color='#2657AF')
            ax[2].plot(t, YT_est_nstep[:, 2], color='#2D9290')

        tf.reset_default_graph()
        sess_i.close()

    # plt.title('Data = ' + str(data_index))
    # ax[0].set_ylim([0,5])
    ax.legend(ncol=3,fontsize=15, loc = 'upper right',columnspacing = 0.5, handletextpad=0.1)
    ax.set_ylim([-0.1,1.5])
    ax.set_yticks([0,0.5,1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Output prediction')
    plt.show()

## Checking diffeomorphism

delay_embedded_system_no = 24
RUN_NO = 3


# Load the dataset
simulation_data_file = 'System_' + str(delay_embedded_system_no) + '/System_' + str(delay_embedded_system_no) + '_SimulatedData.pickle'
with open(simulation_data_file, 'rb') as handle:
    ls_data = pickle.load(handle)
# Load the simulation info [required to convert the data to the required format]
simulation_datainfo_file = 'System_' + str(delay_embedded_system_no) + '/System_' + str(delay_embedded_system_no) + '_SimulatedDataInfo.pickle'
with open(simulation_datainfo_file, 'rb') as handle:
    dict_data_info = pickle.load(handle)
n_outputs = len(dict_data_info['ls_measured_output_indices'])
# Load the scalers
scaler_file = 'System_' + str(delay_embedded_system_no) + '/System_' + str(delay_embedded_system_no) + '_DataScaler.pickle'
with open(scaler_file, 'rb') as handle:
    dict_Scaler = pickle.load(handle)
Z_Scaler = dict_Scaler['XT']

n_delay_embedding = np.int(np.mod(delay_embedded_system_no,10))
Xdim = ls_data[0]['XT'].shape[1]
Ydim = ls_data[0]['YT'].shape[1]
n_timepts = ls_data[0]['XT'].shape[0]
Zdim = Ydim*n_delay_embedding


# Organizing the data
input_X_data = np.empty((0,Xdim))
input_Y_data = np.empty((0,Ydim))
input_Z_data = np.empty((0,n_delay_embedding*Ydim))
for data_i in ls_data:
    for i in range(n_timepts - n_delay_embedding):
        input_X_data = np.concatenate([input_X_data, data_i['XT'][i:i + 1, :]],axis=0)
        input_Y_data = np.concatenate([input_Y_data, data_i['YT'][i:i + 1, :]], axis=0)
        input_Z_data = np.concatenate([input_Z_data, data_i['YT'][i:i + n_delay_embedding, :].T.reshape(1, -1)], axis=0)

# Separate the data into training, validation and test
n_train = np.int(np.ceil(input_Y_data.shape[0]/3))
X_train_data = input_X_data [0:n_train,:]
Y_train_data = input_Y_data [0:n_train,:]
Z_train_data = input_Z_data [0:n_train,:]
X_valid_data = input_X_data [n_train:2*n_train,:]
Y_valid_data = input_Y_data [n_train:2*n_train,:]
Z_valid_data = input_Z_data [n_train:2*n_train,:]
X_test_data = input_X_data [2*n_train:,:]
Y_test_data = input_Y_data [2*n_train:,:]
Z_test_data = input_Z_data [2*n_train:,:]

# Define a Scaler for the state
X_Scaler = StandardScaler()
X_Scaler.fit(X_train_data)
Y_Scaler = StandardScaler()
Y_Scaler.fit(Y_train_data)

# Construct the scaled data
Xs_train_data = X_Scaler.transform(X_train_data)
Xs_valid_data = X_Scaler.transform(X_valid_data)
Xs_test_data = X_Scaler.transform(X_test_data)
Ys_train_data = Y_Scaler.transform(Y_train_data)
Ys_valid_data = Y_Scaler.transform(Y_valid_data)
Ys_test_data = Y_Scaler.transform(Y_test_data)
Zs_train_data = Z_Scaler.transform(Z_train_data)
Zs_valid_data = Z_Scaler.transform(Z_valid_data)
Zs_test_data = Z_Scaler.transform(Z_test_data)


# Import the tensorflow session
sess = tf.InteractiveSession()
run_folder_name = 'System_' + str(delay_embedded_system_no) + '/MyMac/RUN_' + str(RUN_NO)
dict_model = get_dict_param(run_folder_name, delay_embedded_system_no, sess)

# # Load the hyperparameters for each run
# with open(run_folder_name_i + '/dict_hyperparameters.pickle', 'rb') as handle:
#     dict_run_i_info = pickle.load(handle)

# Lift the data to construct psiZ
psiZ_train = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: Zs_train_data})
psiZ_valid = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: Zs_valid_data})
psiZ_test = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: Zs_test_data})
psiZdim = psiZ_train.shape[1]

# Define the autoencoder training parameters
n_hidden_layers = 2
n_nodes = psiZdim


DEVICE_NAME = '/cpu:0'
n_embedded_state_dimension = Xdim
keep_prob = 1
activation_flag = 1
res_net = 0
ls_dict_training_params = []
ls_dict_training_params.append({'step_size': 0.5, 'max_epochs':50000})
ls_dict_training_params.append({'step_size': 0.3, 'max_epochs':50000})
ls_dict_training_params.append({'step_size': 0.1, 'max_epochs':50000})
ls_dict_training_params.append({'step_size': 0.05, 'max_epochs':50000})
ls_dict_training_params.append({'step_size': 0.03, 'max_epochs':50000})
ls_dict_training_params.append({'step_size': 0.01, 'max_epochs':50000})
enc_hidden_vars_list = [n_nodes] * n_hidden_layers
enc_hidden_vars_list.append(n_embedded_state_dimension)
dec_hidden_vars_list = [n_nodes] * n_hidden_layers
dec_hidden_vars_list.append(psiZdim)

# Define and train the autoencoder
# Neural networks initialized
enc_Wx_list, enc_bx_list = initialize_Wblist(psiZdim, enc_hidden_vars_list)
dec_Wx_list, dec_bx_list = initialize_Wblist(n_embedded_state_dimension, dec_hidden_vars_list)
psiZ_feed = tf.placeholder(tf.float32, shape=[None, psiZdim])
psiZ_encoded_feed = tf.placeholder(tf.float32, shape=[None, n_embedded_state_dimension])
X_feed = tf.placeholder(tf.float32, shape=[None, Xdim])
step_size = tf.placeholder(tf.float32, shape=[])
# ENCODER
enc_z_list = []
if activation_flag == 1:  # RELU
    enc_z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(psiZ_feed, enc_Wx_list[0]) + enc_bx_list[0]), keep_prob))
if activation_flag == 2:  # ELU
    enc_z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(psiZ_feed, enc_Wx_list[0]) + enc_bx_list[0]), keep_prob))
if activation_flag == 3:  # tanh
    enc_z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(psiZ_feed, enc_Wx_list[0]) + enc_bx_list[0]), keep_prob))
for k in range(1, len(enc_hidden_vars_list) - 1):
    prev_layer_output = tf.matmul(enc_z_list[k - 1], enc_Wx_list[k]) + enc_bx_list[k]
    if activation_flag == 1:  # RELU
        enc_z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), keep_prob))
    if activation_flag == 2:  # ELU
        enc_z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), keep_prob))
    if activation_flag == 3:  # tanh
        enc_z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), keep_prob))
psiZ_encoded = tf.matmul(enc_z_list[-1], enc_Wx_list[-1]) + enc_bx_list[-1]
# DECODER
dec_z_list = []
if activation_flag == 1:  # RELU
    dec_z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(psiZ_encoded, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob))
    psiZ_decoded_from_encoded = tf.nn.dropout(tf.nn.relu(tf.matmul(psiZ_encoded_feed, dec_Wx_list[0]) + dec_bx_list[0]),keep_prob)
if activation_flag == 2:  # ELU
    dec_z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(psiZ_encoded, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob))
    psiZ_decoded_from_encoded = tf.nn.dropout(tf.nn.elu(tf.matmul(psiZ_encoded_feed, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob)
if activation_flag == 3:  # tanh
    dec_z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(psiZ_encoded, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob))
    psiZ_decoded_from_encoded = tf.nn.dropout(tf.nn.tanh(tf.matmul(psiZ_encoded_feed, dec_Wx_list[0]) + dec_bx_list[0]),keep_prob)
for k in range(1, len(dec_hidden_vars_list) - 1):
    prev_layer_output = tf.matmul(dec_z_list[k - 1], dec_Wx_list[k]) + dec_bx_list[k]
    psiZ_decoded_from_encoded = tf.matmul(psiZ_decoded_from_encoded, dec_Wx_list[k]) + dec_bx_list[k]
    if activation_flag == 1:  # RELU
        dec_z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), keep_prob))
        psiZ_decoded_from_encoded = tf.nn.dropout(tf.nn.relu(psiZ_decoded_from_encoded), keep_prob)
    if activation_flag == 2:  # ELU
        dec_z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), keep_prob))
        psiZ_decoded_from_encoded = tf.nn.dropout(tf.nn.elu(psiZ_decoded_from_encoded), keep_prob)
    if activation_flag == 3:  # tanh
        dec_z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), keep_prob))
        psiZ_decoded_from_encoded = tf.nn.dropout(tf.nn.tanh(psiZ_decoded_from_encoded), keep_prob)
psiZ_decoded_from_encoded = tf.matmul(psiZ_decoded_from_encoded, dec_Wx_list[-1]) + dec_bx_list[-1]
psiZ_decoded = tf.matmul(dec_z_list[-1], dec_Wx_list[-1]) + dec_bx_list[-1]
# Outputs - Yencoded_from_decoded, Y_decoded, Y_encoded
# OBJECTIVE FUNCTION CONSTRUCTION
truth = tf.concat([psiZ_feed, X_feed], axis=1)
prediction = tf.concat([psiZ_decoded, psiZ_encoded], axis=1)
# truth = Y_feed
# prediction = Y_decoded
SSE = tf.math.reduce_mean(tf.math.square(truth - prediction))
# SST = tf.math.reduce_sum(tf.math.square(truth - tf.math.reduce_mean(truth,axis=0)))
# r2 = (1 - tf.divide(SSE, SST)) * 100

loss_fn = SSE
optimizer = tf.train.AdagradOptimizer(step_size).minimize(loss_fn)
sess.run(tf.global_variables_initializer())
# Feed the right data
dict_fed_training_data = {psiZ_feed: psiZ_train, X_feed: Xs_train_data}
dict_fed_validation_data = {psiZ_feed: psiZ_valid, X_feed: Xs_valid_data}
# TRAIN THE NEURAL NET
print('Training Error : ', loss_fn.eval(dict_fed_training_data))
print('Validation Error : ', loss_fn.eval(dict_fed_validation_data))
for dict_training_params in ls_dict_training_params:
    dict_fed_training_data[step_size] = dict_training_params['step_size']
    print('==================================')
    print('Running Step Size :', dict_training_params['step_size'])
    for i in range(dict_training_params['max_epochs']):
        optimizer.run(feed_dict=dict_fed_training_data)
    print('Training Error : ', loss_fn.eval(dict_fed_training_data))
    print('Validation Error : ', loss_fn.eval(dict_fed_validation_data))
    print('X equal accuracy : ', np.max([0, 100 * r2_score(X_train_data, X_Scaler.inverse_transform(psiZ_encoded.eval({psiZ_feed: psiZ_train})))]))
    psiZ_est = psiZ_decoded.eval({psiZ_feed: psiZ_train})
    # print('psiZ reconstruction accuracy : ', np.max([0, 100 * r2_score(psiZ_train, psiZ_est)]))
    print('Z reconstruction accuracy : ', np.max([0, 100 * r2_score(Z_train_data, Z_Scaler.inverse_transform(psiZ_est[:,0:Zdim])  )]))



# TODO Compute the reconstruction stats - maybe make the reconstructed X and Y into a final list of datasets with the truth value as well


# ## State reconstruction
#
# # The inputs for analysis
# n_delay_embedding = np.int(np.mod(delay_embedded_system_no,10))
# XT_true = data_i['XT'][0:-1:n_delay_embedding]
#
##
#
XT_true = X_test_data[0:-1:n_delay_embedding]
XT_est = X_Scaler.inverse_transform(psiZ_encoded.eval({psiZ_feed: psiZ_test}))[0:-1:n_delay_embedding]



f,ax = plt.subplots(3,3,sharex=True,figsize = (10,10))
ax = ax.reshape(-1)
for i in range(7):
    # ax[i].plot(t[0:-1:n_delay_embedding],XT_true[:,i],'.')
    # ax[i].plot(t[0:-1:n_delay_embedding],XT_est[:,i])
    ax[i].plot(XT_true[0:49, i], '.')
    ax[i].plot(XT_est[0:49, i])
f.show()

r2_score(XT_true, XT_est)

##

n_delay_embedding = np.int(np.mod(delay_embedded_system_no,10))
XT_true = data_now['XT'][0:-1:n_delay_embedding]
t_less = t[0:-1:n_delay_embedding]
# XT_est = X_Scaler.inverse_transform(psiZ_encoded.eval({psiZ_feed: psiZT_true}))
XT_est = X_Scaler.inverse_transform(psiZ_encoded.eval({psiZ_feed: psiZT_est_1step}))
XT_est_n = X_Scaler.inverse_transform(psiZ_encoded.eval({psiZ_feed: psiZT_est_nstep}))
f,ax = plt.subplots(3,3,sharex=True,figsize = (10,10))
ax = ax.reshape(-1)
for i in range(7):
    ax[i].plot(t_less ,XT_true[:,i],'.')
    ax[i].plot(t_less ,XT_est[:,i])
    ax[i].plot(t_less, XT_est_n[:, i])
f.show()
r2_score(XT_true, XT_est)

##
XT_est = XT_est_n
simdata_marker_size = 10
legend_face_color = '#D3D3D3'
DS_FACTOR = 2
f = plt.figure(figsize=(15,5))

plt.plot([], '.', color='#AAAAAA', markersize = simdata_marker_size, label='simulation data')
plt.plot([], '.', color='#AAAAAA', linewidth = 2, label='reconstructed state')


# State x1
plt.subplot(2,3,1)
plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,0][0:-1:DS_FACTOR], '.', color='#F88A00', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,0], color='#F88A00', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_1$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x2
plt.subplot(2,3,4)
plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,1][0:-1:DS_FACTOR], '.', color='#F88A00', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,1], color='#F88A00', linewidth = 2)
# plt.legend(title = '$x_2$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(3,3,2)
plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,2][0:-1:DS_FACTOR], '.', color='#2657AF', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,2], color='#2657AF', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_3$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(3,3,5)
plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,3][0:-1:DS_FACTOR], '.', color='#2657AF', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,3], color='#2657AF', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_4$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(3,3,8)
plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,4][0:-1:DS_FACTOR], '.', color='#2657AF', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,4], color='#2657AF', linewidth = 2)
# plt.legend(title = '$x_5$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(2,3,3)
plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,5][0:-1:DS_FACTOR ], '.', color='#2D9290', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,5], color='#2D9290', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_6$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(2,3,6)
plt.plot(t_less[0:-1:DS_FACTOR ], XT_true[:,6][0:-1:DS_FACTOR ], '.', color='#2D9290', markersize = simdata_marker_size)
plt.plot(t_less, XT_est[:,6], color='#2D9290', linewidth = 2)
# l = plt.legend(title = '$x_7$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0, title_color = '#2D9290')

f.legend(loc = 'upper center')
plt.show()

# f,ax =plt.subplots(1,3, sharey=True, figsize = (15,5))
# # Plot truth









# f.show()