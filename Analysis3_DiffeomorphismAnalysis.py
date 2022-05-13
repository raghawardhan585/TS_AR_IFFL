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
import keras
from keras import layers


import tensorflow as tf
import math

sess = tf.InteractiveSession()

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

SYSTEM_NO = 20
ls_n_delay_embeddings = [1,2,3]


# Load the dataset
SYSTEM_NO = np.int(np.floor(SYSTEM_NO/10)*10)
simulation_data_file = 'System_' + str(SYSTEM_NO + 1) + '/System_' + str(SYSTEM_NO + 1) + '_SimulatedData.pickle'
with open(simulation_data_file, 'rb') as handle:
    ls_data = pickle.load(handle)
Xdim = ls_data[0]['XT'].shape[1]
Ydim = ls_data[0]['YT'].shape[1]
n_timepts = ls_data[0]['XT'].shape[0]
n_delay_embedding = 4 #TODO to iterate across ls_n_delay_embeddings later


input_Y_data = np.empty((0,n_delay_embedding*Ydim))
input_X_data = np.empty((0,Xdim))
for data_i in ls_data:
    for i in range(n_timepts - n_delay_embedding):
        input_Y_data = np.concatenate([input_Y_data, data_i['YT'][i:i+n_delay_embedding,:].T.reshape(1,-1)],axis=0)
        input_X_data = np.concatenate([input_X_data, data_i['XT'][i:i + 1, :]],axis=0)

n_train = np.int(np.ceil(input_Y_data.shape[0]/3))

X_train_data = input_X_data [0:n_train,:]
Y_train_data = input_Y_data [0:n_train,:]
X_valid_data = input_X_data [n_train:2*n_train,:]
Y_valid_data = input_Y_data [n_train:2*n_train,:]
X_test_data = input_X_data [2*n_train:,:]
Y_test_data = input_Y_data [2*n_train:,:]

X_Scaler = StandardScaler()
X_Scaler.fit(X_train_data)
Xs_train_data = X_Scaler.transform(X_train_data)
Xs_valid_data = X_Scaler.transform(X_valid_data)
Xs_test_data = X_Scaler.transform(X_test_data)
Y_Scaler = StandardScaler()
Y_Scaler.fit(Y_train_data)
Ys_train_data = Y_Scaler.transform(Y_train_data)
Ys_valid_data = Y_Scaler.transform(Y_valid_data)
Ys_test_data = Y_Scaler.transform(Y_test_data)



ls_dict_training_params = []
ls_dict_training_params.append({'step_size': 0.5, 'max_epochs':5000})
ls_dict_training_params.append({'step_size': 0.3, 'max_epochs':5000})
ls_dict_training_params.append({'step_size': 0.1, 'max_epochs':5000})
# ls_dict_training_params.append({'step_size': 0.05, 'max_epochs':5000})


n_embedded_state_dimension = Xdim
n_hidden_layers = 2
n_nodes = n_delay_embedding*Ydim
DEVICE_NAME = '/cpu:0'
keep_prob = 1
activation_flag = 1
res_net = 0



enc_hidden_vars_list = [n_nodes] * n_hidden_layers
enc_hidden_vars_list.append(n_embedded_state_dimension)
dec_hidden_vars_list = [n_nodes] * n_hidden_layers
dec_hidden_vars_list.append(n_delay_embedding*Ydim)
with tf.device(DEVICE_NAME):
    # Neural networks initialized
    enc_Wx_list, enc_bx_list = initialize_Wblist(n_delay_embedding * Ydim, enc_hidden_vars_list)
    dec_Wx_list, dec_bx_list = initialize_Wblist(n_embedded_state_dimension, dec_hidden_vars_list)
    Y_feed = tf.placeholder(tf.float32, shape=[None, n_delay_embedding * Ydim])
    Yencoded_feed = tf.placeholder(tf.float32, shape=[None, n_embedded_state_dimension])
    if n_embedded_state_dimension == Xdim:
        X_feed = tf.placeholder(tf.float32, shape=[None, Xdim])
    step_size = tf.placeholder(tf.float32, shape=[])
    # ENCODER
    enc_z_list = []
    if activation_flag == 1:  # RELU
        enc_z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(Y_feed, enc_Wx_list[0]) + enc_bx_list[0]),keep_prob))
    if activation_flag == 2:  # ELU
        enc_z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(Y_feed, enc_Wx_list[0]) + enc_bx_list[0]),keep_prob))
    if activation_flag == 3:  # tanh
        enc_z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(Y_feed, enc_Wx_list[0]) + enc_bx_list[0]),keep_prob))
    for k in range(1, len(enc_hidden_vars_list)-1):
        prev_layer_output = tf.matmul(enc_z_list[k - 1], enc_Wx_list[k]) + enc_bx_list[k]
        if activation_flag == 1:  # RELU
            enc_z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), keep_prob))
        if activation_flag == 2:  # ELU
            enc_z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), keep_prob))
        if activation_flag == 3:  # tanh
            enc_z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), keep_prob))
    Y_encoded = tf.matmul(enc_z_list[-1], enc_Wx_list[-1]) + enc_bx_list[-1]
    # DECODER
    dec_z_list = []
    if activation_flag == 1:  # RELU
        dec_z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(Y_encoded, dec_Wx_list[0]) + dec_bx_list[0]),keep_prob))
        Ydecoded_from_encoded = tf.nn.dropout(tf.nn.relu(tf.matmul(Yencoded_feed, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob)
    if activation_flag == 2:  # ELU
        dec_z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(Y_encoded, dec_Wx_list[0]) + dec_bx_list[0]),keep_prob))
        Ydecoded_from_encoded = tf.nn.dropout(tf.nn.elu(tf.matmul(Yencoded_feed, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob)
    if activation_flag == 3:  # tanh
        dec_z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(Y_encoded, dec_Wx_list[0]) + dec_bx_list[0]),keep_prob))
        Ydecoded_from_encoded = tf.nn.dropout(tf.nn.tanh(tf.matmul(Yencoded_feed, dec_Wx_list[0]) + dec_bx_list[0]), keep_prob)
    for k in range(1, len(dec_hidden_vars_list)-1):
        prev_layer_output = tf.matmul(dec_z_list[k - 1], dec_Wx_list[k]) + dec_bx_list[k]
        Ydecoded_from_encoded = tf.matmul(Ydecoded_from_encoded, dec_Wx_list[k]) + dec_bx_list[k]
        if activation_flag == 1:  # RELU
            dec_z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), keep_prob))
            Ydecoded_from_encoded = tf.nn.dropout(tf.nn.relu(Ydecoded_from_encoded), keep_prob)
        if activation_flag == 2:  # ELU
            dec_z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), keep_prob))
            Ydecoded_from_encoded = tf.nn.dropout(tf.nn.elu(Ydecoded_from_encoded), keep_prob)
        if activation_flag == 3:  # tanh
            dec_z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), keep_prob))
            Ydecoded_from_encoded = tf.nn.dropout(tf.nn.tanh(Ydecoded_from_encoded), keep_prob)
    Ydecoded_from_encoded = tf.matmul(Ydecoded_from_encoded, dec_Wx_list[-1]) + dec_bx_list[-1]
    Y_decoded = tf.matmul(dec_z_list[-1], dec_Wx_list[-1]) + dec_bx_list[-1]
    # Outputs - Yencoded_from_decoded, Y_decoded, Y_encoded
    # OBJECTIVE FUNCTION CONSTRUCTION
    truth = tf.concat([Y_feed, X_feed], axis=1)
    prediction = tf.concat([Y_decoded, Y_encoded], axis=1)
    # truth = Y_feed
    # prediction = Y_decoded
    SSE = tf.math.reduce_mean(tf.math.square(truth - prediction))
    # SST = tf.math.reduce_sum(tf.math.square(truth - tf.math.reduce_mean(truth,axis=0)))
    # r2 = (1 - tf.divide(SSE, SST)) * 100

    loss_fn = SSE
    optimizer = tf.train.AdagradOptimizer(step_size).minimize(loss_fn)
    sess.run(tf.global_variables_initializer())
    # Feed the right data
    dict_fed_training_data = {Y_feed: Ys_train_data, X_feed: Xs_train_data}
    dict_fed_validation_data = {Y_feed: Ys_valid_data, X_feed: Xs_valid_data}
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
        if n_embedded_state_dimension == Xdim:
            print('X equal accuracy : ', np.max([0, 100*r2_score( X_train_data , X_Scaler.inverse_transform(Y_encoded.eval({Y_feed: Ys_train_data})))]))
        print('Y reconstruction accuracy : ', np.max([0, 100*r2_score( Y_train_data , Y_Scaler.inverse_transform(Y_decoded.eval({Y_feed: Ys_train_data})))]))


# ##
# plt.plot(Y_train_data[0:100])
# plt.show()
# plt.plot(X_train_data[0:100])
# plt.show()

##
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept = True, multioutput='uniform_average')
linear_model.fit(Y_train_data, X_train_data)
print('[Lienar Model] Training Accuracy : ', linear_model.score(Y_train_data, X_train_data)*100)
print('[Lienar Model] Validation Accuracy : ', linear_model.score(Y_valid_data, X_valid_data)*100)