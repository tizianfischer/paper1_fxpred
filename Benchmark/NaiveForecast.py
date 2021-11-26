#!/usr/bin/env python
# coding: utf-8

# %% Loading Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import MSE, MAE, MAPE


import sys
import os
import sys
import platform
if platform.node() in ['msbq']:
    os.chdir('/home/ms/github/fxpred')
    # os.chdir('../.')
    sys.path.append(os.path.join(os.getcwd(), 'Transformer'))
from utils import data_read_dict, data_read_concat, data_merge
from utils import get_fx_and_metric_data_wo_weekend, mde

dtype = None  # np.float16
# %% read in data and adapt

df = get_fx_and_metric_data_wo_weekend(dtype=np.float32)
# df = df.loc[(df.index >= '2020-11-01') & (df.index < '2021-08-01'), :]


# %% 
def actual_pred_plot(preds, y_test, error=False):
    '''
    Plot the actual vs. prediction
    '''
    if not error:
        actual_pred = pd.DataFrame(columns = ['Adj. Close', 'prediction'])
        actual_pred['prediction'] = preds[:,0]
        actual_pred['Adj. Close'] = y_test[:,0]  #.loc['2019':,'Adj Close'][0:len(preds)]
    else:
        actual_pred = pd.DataFrame(columns = ['Error'])
        actual_pred['Error'] = preds[:,0] - y_test[:, 0]

    from tensorflow.keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(y_test[:,0]),np.array(preds[:,0]))
    return (m.result().numpy(), actual_pred.plot() )

# %% plot one closing currency pair
# df = df.loc[(df.index >= '2020-11-01') & (df.index < '2020-12-01'), :]
# df['EURUSD BGNE Curncy Bid Close'].plot()
plt.plot(df['EURUSD BGNE Curncy Bid Close'].values)
plt.show()

# %% Data split into train and test data
def ts_train_test_normalize(df,time_steps,for_periods, target_column=1):
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
    sc = StandardScaler().fit(ts_train)
    sc_target = StandardScaler().fit(ts_train.iloc[:, target_column:target_column+1])
    ts_train_scaled = ts_train.values  # sc.transform(ts_train)
    ts_val_scaled = ts_val.values  # sc.transform(ts_val)
    ts_test_scaled = ts_test.values  # sc.transform(ts_test)

    


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
        y_train.append(ts_train_scaled[i:i+for_periods, target_column])
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

    # Preparing X_test
    X_test = []
    y_test = []
    for i in range(time_steps,ts_test_len-for_periods):
        X_test.append(ts_test_scaled[i-time_steps:i])
        y_test.append(ts_test_scaled[i:i+for_periods,target_column])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    return X_train, y_train, X_val, y_val , X_test, y_test, sc, sc_target, index_train, index_val, index_test

target_column = list(df.columns).index('EURUSD BGNE Curncy Bid Close')
X_train, y_train, X_val, y_val , X_test, y_test, sc, sc_target, index_train, index_val, index_test = \
    ts_train_test_normalize(df, 64, 1, target_column)

X_train = X_train[:, :, target_column : target_column + 1]
X_val = X_val[:, :, target_column : target_column + 1]
X_test = X_test[:, :, target_column : target_column + 1]

y = y_train[1:]
y_pred = y_train[:-1]
print(f'mse: {MSE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mae: {MAE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mape: {MAPE(y.flatten(), y_pred.flatten()).numpy()}')
# print(f'mde: {1 - np.mean(np.diff(y.flatten()) * (y_pred.flatten() - y.flatten())[1:] >= 0)}')
print(f'mde: {mde(y, y_pred)}')


y = y_val[1:]
y_pred = y_val[:-1]
print(f'mse: {MSE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mae: {MAE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mape: {MAPE(y.flatten(), y_pred.flatten()).numpy()}')
# print(f'mde: {1 - np.mean(np.diff(y.flatten()) * (y_pred.flatten() - y.flatten())[1:] >= 0)}')
print(f'mde: {mde(y, y_pred)}')

print('test')
y = y_test[1:]
y_pred = y_test[:-1]
print(f'mse: {MSE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mae: {MAE(y.flatten(), y_pred.flatten()).numpy()}')
print(f'mape: {MAPE(y.flatten(), y_pred.flatten()).numpy()}')
# print(f'mde: {1 - np.mean(np.diff(y.flatten()) * (y_pred.flatten() - y.flatten())[1:] >= 0)}')
print(f'mde: {mde(y, y_pred)}')

