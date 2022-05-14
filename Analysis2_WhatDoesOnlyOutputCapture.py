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

def get_autoencoder_decoder(system_no, sess):
    dict_p = {}
    run_folder = 'AutoEncoder_Decoder/System_' + str(np.int(np.floor(system_no/10)*10))
    # Pickle data info
    with open(run_folder+ '/key_details.pickle', 'rb') as handle:
        dict_model = pickle.load(handle)
    # Tensorflow session info
    saver = tf.compat.v1.train.import_meta_graph(run_folder + '/autoenc_dec.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder))
    dict_model['XT_feed'] = tf.get_collection('XT_feed')[0]
    dict_model['psiZT_feed'] = tf.get_collection('psiZT_feed')[0]
    dict_model['psiZT_encoded_feed'] = tf.get_collection('psiZT_encoded_feed')[0]
    dict_model['psiZT_decoded_from_encoded'] = tf.get_collection('psiZT_decoded_from_encoded')[0]
    dict_model['psiZT_decoded'] = tf.get_collection('psiZT_decoded')[0]
    dict_model['psiZT_encoded'] = tf.get_collection('psiZT_encoded')[0]
    return dict_model


##
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
    ls_output_indices = dict_data_info['ls_measured_output_indices']
    n_outputs = len(ls_output_indices)
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
            x0 = data_i['YT'][0:n_delay_embedding, ls_output_indices].T.reshape(1, -1)
            # Organize the data
            XT_true = np.empty((0, x0.shape[1]))
            if dict_data_info['formulate_Koopman_output_data_with_intersection']:
                for j in range(n_delay_embedding, data_i['YT'].shape[0]):
                    XT_true = np.concatenate([XT_true, data_i['YT'][j - n_delay_embedding:j,ls_output_indices].T.reshape(1,-1)],axis=0)
            else:
                n_delay_embedded_points = np.int(np.floor(data_i['YT'].shape[0] / n_delay_embedding))
                for j in range(n_delay_embedded_points):
                    XT_true = np.concatenate([XT_true, data_i['YT'][j * n_delay_embedding:(j + 1) * n_delay_embedding,ls_output_indices].T.reshape(1,-1)], axis=0)
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
for data_index in range(21,30):
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
    ax.legend(ncol=3,fontsize=15, loc = 'upper right',columnspacing = 0.5, handletextpad=0.1, frameon = False)
    ax.set_ylim([-0.1,1.5])
    ax.set_yticks([0,0.5,1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Output prediction')
    plt.show()

## Check the differomorphism
dict_diffeomorphic_data = {}

DATA_INDEX = 2

for SYSTEM in [20,30,40,50]:
    sess = tf.InteractiveSession()
    dict_encdec = get_autoencoder_decoder(SYSTEM, sess)
    run_folder_name_i = 'System_' + str(dict_encdec['system_no']) + '/MyMac/RUN_' + str(dict_encdec['run_no'])
    # Load the hyperparameters for the run
    with open(run_folder_name_i + '/dict_hyperparameters.pickle', 'rb') as handle:
        dict_run_i_info = pickle.load(handle)
    # Load the model
    dict_model_i = get_dict_param(run_folder_name_i, dict_encdec['system_no'], sess)

    # Iterate through each dataset

    # Consider 1 dataset
    simulation_data_file = 'System_' + str(dict_encdec['system_no']) + '/System_' + str(dict_encdec['system_no']) + '_SimulatedData.pickle'
    with open(simulation_data_file, 'rb') as handle:
        ls_data = pickle.load(handle)
    # Load the simulation info [required to convert the data to the required format]
    simulation_datainfo_file = 'System_' + str(dict_encdec['system_no']) + '/System_' + str(dict_encdec['system_no']) + '_SimulatedDataInfo.pickle'
    with open(simulation_datainfo_file, 'rb') as handle:
        dict_data_info = pickle.load(handle)
    ls_output_indices = dict_data_info['ls_measured_output_indices']
    data_i = ls_data[DATA_INDEX]

    ZT = np.empty((0,len(ls_output_indices)*dict_encdec['n_delay_embedding']))
    for i in range(data_i['XT'].shape[0] - dict_encdec['n_delay_embedding']):
        ZT = np.concatenate([ZT , data_i['YT'][i: i+dict_encdec['n_delay_embedding'],ls_output_indices].T.reshape(1,-1)],axis=0)
    ZTs = dict_encdec['ZT_Scaler'].transform(ZT)
    psiZT = dict_model_i['psixpT'].eval({dict_model_i['xpT_feed']: ZTs})
    XTshat = dict_encdec['psiZT_encoded'].eval({dict_encdec['psiZT_feed']:psiZT})
    XThat = dict_encdec['XT_Scaler'].inverse_transform(XTshat)
    XT = data_i['XT'][0:len(XThat),:]
    dict_diffeomorphic_data[SYSTEM] = {'XT':XT, 'XThat':XThat, 't':dict_data_info['t'][0:len(XThat)]}
    tf.reset_default_graph()
    sess.close()


## Let's try a bar plot of the accuracy
dict_reconstruction_stats = {}
for SYSTEM in dict_diffeomorphic_data.keys():
    if SYSTEM == 20:
        key = '$y_1-y_2-y_3$'
    elif SYSTEM == 30:
        key = '$y_1$'
    elif SYSTEM == 40:
        key = '$y_2$'
    elif SYSTEM == 50:
        key = '$y_3$'
    else:
        key = 'Unknown'
    dict_reconstruction_stats[key] = {}
    for i in range(7):
        dict_reconstruction_stats[key]['$x_'+str(i+1) + '$'] = r2_score(dict_diffeomorphic_data[SYSTEM]['XT'][:,i], dict_diffeomorphic_data[SYSTEM]['XThat'][:,i])


df_reconstruction_stats = np.maximum(0,pd.DataFrame(dict_reconstruction_stats)).T

##
# f,ax = plt.subplots(1,1)
# df_reconstruction_stats.plot(kind='bar', rot = 0, ax = ax,color=['#333333','#FFB371', '#ABBEE0', '#AAD5D4'], width=0.7)
f,ax = plt.subplots(1,1, figsize = (14,5))
df_reconstruction_stats.plot(kind='bar', rot = 0, ax = ax,color=['#F88A00','#FFB371', '#2657AF', '#6D8EC9', '#ABBEE0', '#2D9290', '#AAD5D4'], width=0.8)
ax.set_ylim([0,1.25])
ax.set_yticks([0,0.5,1.])
ax.legend(ncol=7, loc= 'upper right',frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlabel('State')
# ax.set_ylabel('Reconstruction\naccuracy', labelpad=-0.1)
# ax.yaxis.set_label_coordinates(0,0)
f.show()
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
simdata_marker_size = 10
legend_face_color = '#D3D3D3'
DS_FACTOR = 2
f = plt.figure(figsize=(15,5))

plt.plot([], '.', color='#AAAAAA', markersize = simdata_marker_size, label='simulation data')
plt.plot([], '.', color='#AAAAAA', linewidth = 2, label='reconstructed state')


# State x1
plt.subplot(2,3,1)
# plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,0][0:-1:DS_FACTOR], '.', color='#F88A00', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,0], color='#F88A00', linewidth = 2)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,0], color='#F88A00')
plt.xticks([])
# plt.legend(title = '$x_1$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x2
plt.subplot(2,3,4)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,1], color='#F88A00')
# plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,1][0:-1:DS_FACTOR], '.', color='#F88A00', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,1], color='#F88A00', linewidth = 2)
# plt.legend(title = '$x_2$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(3,3,2)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,2], color='#2657AF')
# plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,2][0:-1:DS_FACTOR], '.', color='#2657AF', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,2], color='#2657AF', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_3$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(3,3,5)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,3], color='#2657AF')
# plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,3][0:-1:DS_FACTOR], '.', color='#2657AF', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,3], color='#2657AF', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_4$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(3,3,8)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,4], color='#2657AF')
# plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,4][0:-1:DS_FACTOR], '.', color='#2657AF', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,4], color='#2657AF', linewidth = 2)
# plt.legend(title = '$x_5$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(2,3,3)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,5], color='#2D9290')
# plt.plot(t_less[0:-1:DS_FACTOR], XT_true[:,5][0:-1:DS_FACTOR ], '.', color='#2D9290', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,5], color='#2D9290', linewidth = 2)
plt.xticks([])
# plt.legend(title = '$x_6$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0)


# State x1
plt.subplot(2,3,6)
for SYSTEM in dict_diffeomorphic_data.keys():
    plt.plot(dict_diffeomorphic_data[SYSTEM]['t'], dict_diffeomorphic_data[SYSTEM]['XThat'][:,6], color='#2D9290')
# plt.plot(t_less[0:-1:DS_FACTOR ], XT_true[:,6][0:-1:DS_FACTOR ], '.', color='#2D9290', markersize = simdata_marker_size)
# plt.plot(t_less, XT_est[:,6], color='#2D9290', linewidth = 2)
# l = plt.legend(title = '$x_7$', loc = 'upper right', facecolor = legend_face_color, edgecolor = legend_face_color, framealpha = 1, labelspacing = 0, title_color = '#2D9290')

f.legend(loc = 'upper center')
plt.show()

# f,ax =plt.subplots(1,3, sharey=True, figsize = (15,5))
# # Plot truth









# f.show()