#!/usr/bin/env python
# coding: utf-8

# %% Loading Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import SGD

# %% read in data and adapt

path = 'data/10min Dataset.csv'
df = pd.read_csv(path, delimiter=';')
df['Dates'] = pd.to_datetime(df['Dates'], format='%d.%m.%y %H:%M')
df.set_index('Dates', inplace=True)
df = df.pct_change()[1:]
df.head(10)

df[df == np.infty] = 0 
df[df == -np.infty] = 0
df.dropna(how='all', axis=0, inplace=True) # Drop all rows with NaN values"
df.fillna(0, inplace=True)
print(set(np.diff(df.index.values)))


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
df['EURUSD BGNE Curncy Bid Close'].plot()
plt.show()

# %% Data split into train and test data
def ts_train_test_normalize(df,time_steps,for_periods):
    '''
    input: 
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2020/11/2-2020/12/31
      X_test:  data from 2021 -
      sc:      insantiated MinMaxScaler object fit to the training data
    '''    # create training and test set
    target_column = 3
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    ts_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    index_train = df[df.index < last_20pct].index
    ts_test = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    index_test = df[(df.index >= last_20pct) & (df.index < last_10pct)].index

    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    '''Normalize price columns'''
    #   df = (df - df.mean()) / (df.max() - df.min())
    #   df.columns[np.isnan(df).any(axis=0)]
    
    sc = MinMaxScaler((-1, 1)).fit(ts_train)
    sc_target = MinMaxScaler((-1, 1)).fit(ts_train.iloc[:, target_column:target_column+1])
    ts_train_scaled = sc.transform(ts_train)
    
    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i])
        y_train.append(ts_train_scaled[i:i+for_periods,target_column])
    X_train, y_train = np.array(X_train), np.array(y_train)

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

    return X_train, y_train , X_test, y_test, sc, sc_target, index_train, index_test

X_train, y_train, X_test, y_test, sc, sc_target, index_train, index_test = \
    ts_train_test_normalize(df, 12, 1)

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
    #my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    #my_rnn_model.add(SimpleRNN(32, return_sequences=True))
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

#%% LSTM model definition

def LSTM_model():
    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=128, activation='tanh', return_sequences=True))
    my_LSTM_model.add(LSTM(units= 64, activation='tanh', return_sequences=True))
    my_LSTM_model.add(LSTM(units= 64, activation='tanh'))
    my_LSTM_model.add(Dense(units=2))

    return my_LSTM_model

def train(my_model, X_train, y_train):
    my_model.compile(
        optimizer=SGD(
            lr=0.01,
            decay=1e-7,
            momentum=0.9,
            nesterov=False
        ),
        loss='mean_squared_error'
    )
    # Fitting to the training set


def predictions(my_model, X_test, sc):
    LSTM_prediction = my_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)

    return LSTM_prediction

def earlyStopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=20,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

from tensorflow.keras.optimizers import Adam
my_LSTM_model = LSTM_model()
my_LSTM_model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    # optimizer=SGD(
    #     lr=0.1,
    #     decay=1e-7,
    #     momentum=0.9,
    #     nesterov=False
    # ),
    loss='mean_squared_error'
)
history = my_LSTM_model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    callbacks=[earlyStopping()]
)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

my_LSTM_model.summary()

# Compiling
LSTM_prediction = predictions(my_LSTM_model, X_test, sc_target)
LSTM_prediction[1:10]
actual_pred_plot(LSTM_prediction * 500, y_test)
plt.show()

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

