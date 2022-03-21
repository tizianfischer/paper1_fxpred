#!/usr/bin/env python

# coding: utf-8

# %% Loading Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error


def predictions(my_model, X_test, sc=None):
    LSTM_prediction = my_model.predict(X_test)
    if sc is not None:
        LSTM_prediction = sc.inverse_transform(LSTM_prediction)
    return LSTM_prediction
def predictions2(my_model, X_test, sc=None, bs=100):
    LSTM_prediction = np.concatenate(
        [my_model.predict_on_batch(X_test[i * bs : min((i+1)*bs, X_test.shape[0])]) for i in range(X_test.shape[0] // bs + 1)],
        axis = 0
    )
    if sc is not None:
        LSTM_prediction = sc.inverse_transform(LSTM_prediction)
    return LSTM_prediction

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
    return (m.result().numpy(), actual_pred.plot())

def ts_train_test_normalize(df, time_steps, for_periods, target_column=3, dtype=None):
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
    sc = StandardScaler().fit(ts_train)
    sc_target = StandardScaler().fit(ts_train.iloc[:, target_column:target_column+1])
    ts_train_scaled = sc.transform(ts_train)
    ts_val_scaled = sc.transform(ts_val)
    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    X_train_index = []
    y_train_index = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i])
        X_train_index.append(index_train[i-time_steps:i])
        y_train.append(ts_train_scaled[i:i+for_periods,target_column])
        y_train_index.append(index_train[i:i+for_periods])
    X_train, y_train = np.array(X_train, dtype=dtype), np.array(y_train, dtype=dtype)
    X_train_index, y_train_index = np.array(X_train_index, dtype=dtype), np.array(y_train_index, dtype=dtype)

    X_val = []
    y_val = []
    y_val_stacked = []
    for i in range(time_steps,ts_val_len-1): 
        X_val.append(ts_val_scaled[i-time_steps:i])
        y_val.append(ts_val_scaled[i:i+for_periods,target_column])
    X_val, y_val = np.array(X_val, dtype=dtype), np.array(y_val, dtype=dtype)
    # Reshaping X_train for efficient modelling
    #     X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    ts_test_scaled = sc.transform(ts_test)
    # Preparing X_test
    X_test = []
    y_test = []
    for i in range(time_steps,ts_test_len-for_periods):
        X_test.append(ts_test_scaled[i-time_steps:i])
        y_test.append(ts_test_scaled[i:i+for_periods,target_column])
    
    X_test, y_test = np.array(X_test, dtype=dtype), np.array(y_test, dtype=dtype)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    return X_train, y_train, X_val, y_val , X_test, y_test, sc, sc_target, index_train, index_val, index_test, X_train_index, y_train_index


def ts_train_test_normalize2(df, time_steps, for_periods = 1, target_columns_id=[3], dtype=None):
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
    if not isinstance(target_columns_id, (list, tuple, np.array)):
        target_columns_id = list(target_columns_id)
    sc = StandardScaler().fit(ts_train)
    sc_target = StandardScaler().fit(ts_train.iloc[:, target_columns_id])
    ts_val_scaled = sc.transform(ts_val)

    # create training data of s samples and t time steps
#     X_train = []
#     y_train = []
#     y_train_stacked = []
#     for i in range(time_steps,ts_train_len-for_periods): 
#         X_train.append(ts_train_scaled[i-time_steps:i].tolist())
#         y_train.append(ts_train_scaled[i:i+for_periods,target_columns_id].tolist())
#     X_train, y_train = np.array(X_train, dtype=dtype), np.array(y_train, dtype=dtype)
    X_train = []
    y_train = []
    X_train_index = []
    y_train_index = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i])
        X_train_index.append(index_train[i-time_steps:i])
        y_train.append(ts_train_scaled[i:i+for_periods,target_column])
        y_train_index.append(index_train[i:i+for_periods])
    X_train, y_train = np.array(X_train, dtype=dtype), np.array(y_train, dtype=dtype)
    X_train_index, y_train_index = np.array(X_train_index, dtype=dtype), np.array(y_train_index, dtype=dtype)
    
    X_val = []
    y_val = []
    y_val_stacked = []
    for i in range(time_steps,ts_val_len-for_periods): 
        X_val.append(ts_val_scaled[i-time_steps:i].tolist())
        y_val.append(ts_val_scaled[i:i+for_periods,target_columns_id].tolist())
    X_val, y_val = np.array(X_val, dtype=dtype), np.array(y_val, dtype=dtype)
    # Reshaping X_train for efficient modelling
    #     X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    ts_test_scaled = sc.transform(ts_test)
    # Preparing X_test
    X_test = []
    y_test = []
    for i in range(time_steps,ts_test_len-for_periods):
        X_test.append(ts_test_scaled[i-time_steps:i].tolist())
        y_test.append(ts_test_scaled[i:i+for_periods,target_columns_id].tolist())
    
    X_test, y_test = np.array(X_test, dtype=dtype), np.array(y_test, dtype=dtype)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    return X_train, y_train, X_val, y_val , X_test, y_test, sc, sc_target, index_train, index_val, index_test, X_train_index, y_train_index
