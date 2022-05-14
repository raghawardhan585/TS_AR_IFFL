# Checking diffeomorphism

# delay_embedded_system_no = 24
# RUN_NO = 3

# delay_embedded_system_no = 34
# RUN_NO = 1

delay_embedded_system_no = 44
RUN_NO = 0

# delay_embedded_system_no = 53
# RUN_NO = 1






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

# Load the dataset
simulation_data_file = 'System_' + str(delay_embedded_system_no) + '/System_' + str(delay_embedded_system_no) + '_SimulatedData.pickle'
with open(simulation_data_file, 'rb') as handle:
    ls_data = pickle.load(handle)
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
Z_Scaler = dict_Scaler['XT']

n_delay_embedding = np.int(np.mod(delay_embedded_system_no,10))
Xdim = ls_data[0]['XT'].shape[1]
Ydim = len(ls_output_indices)
n_timepts = ls_data[0]['XT'].shape[0]
Zdim = Ydim*n_delay_embedding


# Organizing the data
input_X_data = np.empty((0,Xdim))
input_Y_data = np.empty((0,Ydim))
input_Z_data = np.empty((0,n_delay_embedding*Ydim))
for data_i in ls_data:
    for i in range(n_timepts - n_delay_embedding):
        input_X_data = np.concatenate([input_X_data, data_i['XT'][i:i + 1, :]],axis=0)
        input_Y_data = np.concatenate([input_Y_data, data_i['YT'][i:i + 1, ls_output_indices]], axis=0)
        input_Z_data = np.concatenate([input_Z_data, data_i['YT'][i:i + n_delay_embedding, ls_output_indices].T.reshape(1, -1)], axis=0)

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

## Save the run
dict_dump = {}
dict_dump['system_no'] = delay_embedded_system_no
dict_dump['n_delay_embedding'] = np.int(np.mod(delay_embedded_system_no,10))
dict_dump['run_no'] = RUN_NO
# All scalers
dict_dump['XT_Scaler'] = X_Scaler
dict_dump['YT_Scaler'] = Y_Scaler
dict_dump['ZT_Scaler'] = Z_Scaler

saver_i = tf.compat.v1.train.Saver()
# Feed Variables
tf.compat.v1.add_to_collection('XT_feed', X_feed)
tf.compat.v1.add_to_collection('psiZT_feed', psiZ_feed)
tf.compat.v1.add_to_collection('psiZT_encoded_feed', psiZ_encoded_feed)
# Evaluation Variables
tf.compat.v1.add_to_collection('psiZT_decoded_from_encoded', psiZ_decoded_from_encoded)
tf.compat.v1.add_to_collection('psiZT_decoded', psiZ_decoded)
tf.compat.v1.add_to_collection('psiZT_encoded', psiZ_encoded)

storage_folder = 'AutoEncoder_Decoder/System_' + str(np.int(np.floor(delay_embedded_system_no/10)*10))
if not os.path.exists(storage_folder):
    if not os.path.exists('AutoEncoder_Decoder'):
        os.mkdir('AutoEncoder_Decoder')
    os.mkdir(storage_folder)
saver_path_curr = saver_i.save(sess, storage_folder + '/autoenc_dec' + str() + '.ckpt')
with open(storage_folder + '/key_details.pickle', 'wb') as handle:
    pickle.dump(dict_dump, handle)
