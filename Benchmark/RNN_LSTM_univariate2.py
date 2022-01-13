#!/usr/bin/env python

# coding: utf-8

# %% Loading Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MSE, MAE, MAPE
from tensorflow.keras.callbacks import LearningRateScheduler

import sys

from tensorflow.python.ops.gen_batch_ops import batch

import sys
import os
import sys
import platform

from tensorflow.python.ops.gen_math_ops import Tanh
if platform.node() in ['msbq']:
    os.chdir('/home/ms/github/fxpred')
    # os.chdir('../.')
    sys.path.append(os.path.join(os.getcwd(), 'Transformer'))
# from utils import data_read_dict, data_read_concat, data_merge
from utils import get_fx_and_metric_data_wo_weekend, mde
from utils_NN_opt_learning_rate import opt_learn_rate_plot


dtype = np.float32  # np.float64
tf.keras.backend.set_floatx('float32')
# %% read in data and adapt
df = get_fx_and_metric_data_wo_weekend(dtype=dtype)
target_column = list(df.columns).index('EURUSD BGNE Curncy Bid Close')
df = df.iloc[:, target_column : target_column + 1]


# %% 
def actual_pred_plot(preds, y_test, error=False):
    '''
    Plot the actual vs. prediction
    '''
    if not error:
        actual_pred = pd.DataFrame(columns = ['Adj. Close', 'prediction'])
        actual_pred['prediction'] = preds
        actual_pred['Adj. Close'] = y_test  #.loc['2019':,'Adj Close'][0:len(preds)]
    else:
        actual_pred = pd.DataFrame(columns = ['Error'])
        actual_pred['Error'] = preds - y_test

    from tensorflow.keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(y_test),np.array(preds))
    return (m.result().numpy(), actual_pred.plot() )

# %% plot one closing currency pair
df['EURUSD BGNE Curncy Bid Close'].plot()
# plt.show()

# %% Data split into train and test data
def ts_train_test_normalize(df,time_steps,for_periods, target_column=3):
    '''
    input: 
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2020/11/2-2020/12/31
      X_test:  data from 2021 -
      sc:      insantiated MinMaxScaler object fit to the training data
    '''    # create training and test set
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    ts_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    index_train = df[df.index < last_20pct].index
    ts_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    index_val = df[(df.index >= last_20pct) & (df.index < last_10pct)].index
    ts_test = df[(df.index >= last_10pct)]
    index_test = df[(df.index >= last_10pct)].index

    ts_train_len = len(ts_train)
    ts_val_len = len(ts_test)
    ts_test_len = len(ts_test)

    '''Normalize price columns'''
    #   df = (df - df.mean()) / (df.max() - df.min())
    #   df.columns[np.isnan(df).any(axis=0)]
    
    # sc = MinMaxScaler((-1, 1)).fit(ts_train)
    sc = MinMaxScaler().fit(ts_train)
    sc_target = MinMaxScaler().fit(ts_train.iloc[:, target_column:target_column+1])
    ts_train_scaled = sc.transform(ts_train)
    ts_val_scaled = sc.transform(ts_val)

    


    # ts_train_scaled = pd.DataFrame(
    #     (df.values - df.values.mean(axis=0)[np.newaxis]) / df.values.std(axis=0)[np.newaxis]
    # )
    # ts_train_scaled.index = df.index
    # ts_train_scaled.columns = df.columns
    
    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i])
        y_train.append(ts_train_scaled[i:i+for_periods,target_column])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_val = []
    y_val = []
    y_val_stacked = []
    for i in range(time_steps,ts_val_len-1): 
        X_val.append(ts_val_scaled[i-time_steps:i])
        y_val.append(ts_val_scaled[i:i+for_periods,target_column])
    X_val, y_val = np.array(X_val), np.array(y_val)
    # Reshaping X_train for efficient modelling
    #     X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    ts_test_scaled = sc.transform(ts_test)
    # Preparing X_test
    X_test = []
    y_test = []
    for i in range(time_steps,ts_test_len-for_periods):
        X_test.append(ts_test_scaled[i-time_steps:i])
        y_test.append(ts_test_scaled[i:i+for_periods,target_column])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    return X_train, y_train, X_val, y_val , X_test, y_test, sc, sc_target, index_train, index_val, index_test

X_train, y_train, X_val, y_val , X_test, y_test, sc, sc_target, index_train, index_val, index_test = \
    ts_train_test_normalize(df, 128, 1, 0)

# %% RNN model
def simple_rnn_model(X_train, y_train, X_test, sc):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN
    
    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(2)) # The time step of the output

    my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
        # fit the RNN model
    my_rnn_model.fit(X_train, y_train, epochs=100, batch_size=150, verbose=0)

    # Finalizing predictions
    rnn_predictions = my_rnn_model.predict(X_test)
    from sklearn.preprocessing import MinMaxScaler
    rnn_predictions = sc.inverse_transform(rnn_predictions)

    return my_rnn_model, rnn_predictions

def LSTM_model():
    # The LSTM architecture
    loss_fct = 'tanh'

    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=128, activation=loss_fct, return_sequences=True, dropout=0.5))
    my_LSTM_model.add(BatchNormalization())
    my_LSTM_model.add(LSTM(units=128, activation=loss_fct, return_sequences=True, dropout=0.5))
    my_LSTM_model.add(BatchNormalization())
    my_LSTM_model.add(LSTM(units=64, activation=loss_fct, return_sequences=True, dropout=0.5))
    # my_LSTM_model.add(tf.keras.layers.BatchNormalization())
    my_LSTM_model.add(Flatten())
    my_LSTM_model.add(Dropout(0.5))
    my_LSTM_model.add(Dense(units=128))
    my_LSTM_model.add(Dense(units=1))
    # my_LSTM_model.add(LSTM(units=1))
    return my_LSTM_model

def predictions(my_model, X_test, sc=None):
    LSTM_prediction = my_model.predict(X_test)
    if sc is not None:
        LSTM_prediction = sc.inverse_transform(LSTM_prediction)
    return LSTM_prediction

def earlyStopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
def learning_rate_scheduler(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0005
    elif epoch < 100:
        return 0.0001




from tensorflow.keras.optimizers import Adam
my_LSTM_model = LSTM_model()
my_LSTM_model.compile(
    optimizer=Adam(
        learning_rate=0.05
    ),
    # optimizer=SGD(
    #     lr=0.1,
    #     decay=1e-7,
    #     momentum=0.9,
    #     nesterov=False
    # ),
    loss='mean_squared_error',
    metrics=[MAE, MAPE],
)

history = my_LSTM_model.fit(
    X_train, y_train,
    # X_train, y_train,
    # epochs=50,
    epochs=1,
    steps_per_epoch=1,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1,
    shuffle=True,
    callbacks=[
        # earlyStopping(),
        LearningRateScheduler(learning_rate_scheduler),
        tf.keras.callbacks.TerminateOnNaN()
    ]
)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.ylim(0, 0.01)
plt.show()

my_LSTM_model.summary()
my_LSTM_model.layers[1].get_weights()[2].dtype

opt_learn_rate_plot(
    my_LSTM_model,
    X_train,
    y_train,
    10**-6,
    10**-2,
    100,
    batch_size=64,
    steps_per_epoch=1
)
# Compiling
# %%
# my_model = my_LSTM_model
# X_test = X_test
# sc = sc_target
LSTM_prediction = predictions(
    my_LSTM_model,
    X_test,
    sc_target
)
actual_pred_plot(LSTM_prediction[:, 0], sc_target.inverse_transform(y_test)[:, 0])
plt.show()

print('Test')
y = sc_target.inverse_transform(y_test)
# y = y_test
y_pred = LSTM_prediction = predictions(
    my_LSTM_model,
    X_test,
    sc_target
)
print(f'mse: {MSE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mae: {MAE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mape: {MAPE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mde: {mde(y.flatten(), y_pred.flatten())}')

print('Train')
y = sc_target.inverse_transform(y_train)
y_pred = predictions(
    my_LSTM_model,
    X_train,
    # sc_target
)
actual_pred_plot(y_pred[:, 0], sc_target.inverse_transform(y_train)[:, 0])
plt.show()
print(f'mse: {MSE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mae: {MAE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mape: {MAPE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mde: {mde(y.flatten(), y_pred.flatten())}')


# actual_pred_plot((LSTM_prediction- LSTM_prediction[:, 0].mean()) * 150, y_test, error=True)
# (((LSTM_prediction[:, 0] - LSTM_prediction[:, 0].mean()) * 150 - y_test[:, 0])**2).mean()
# plt.show()

def LSTM_model_regularization(X_train, y_train, X_test, sc):
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU, LSTM
    from keras.optimizers import SGD
    
    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    #my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    #my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dense(units=2))

    return my_LSTM_model

# df.iloc[:, target_column].plot()
# plt.show()

# for i in range(1, 2):
#     plt.scatter(
#         df.iloc[:-i, target_column],
#         df.iloc[i:, target_column]
#     )
# plt.xlabel('y_{t-1}')
# plt.ylabel('y_t')
# plt.show()