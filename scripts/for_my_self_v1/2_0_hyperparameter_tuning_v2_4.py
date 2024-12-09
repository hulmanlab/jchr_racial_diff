#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:35:43 2024
"""

#%% 
import pandas as pd
import my_utils
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


file_path_df = r'../results/preprocessed_data/1_2_model_input_ws60min_ph60min.csv'
df = pd.read_csv(file_path_df)

group_name = "Race"

df.drop(columns=["EduLevel", "Gender"], inplace=True)
df.dropna(inplace=True)





#%%  Define hyperparameters for tuning


# lay1_neurons = [64, 32, 16, 8]  # First LSTM layer neurons
# lay2_neurons = [64, 32, 16, 8]  # Second LSTM layer neurons
# activation_function = ['relu', 'tanh']
# dropout_rates = [0.05, 0.1, 0.2, 0.25]      # Dropout rates
# batch_size = [64, 128, 256, 512, 1024, 2048, 4096]
# learning_rate = [0.01, 0.001]
# decay_rate = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]




# # Create all possible combinations of hyperparameters
# params = list(itertools.product(lay1_neurons, # 0
#                                 lay2_neurons, # 1
#                                 activation_function, # 2
#                                 dropout_rates,       # 3
#                                 batch_size,          # 4
#                                 learning_rate,       # 5
#                                 decay_rate))          # 6

file_path_df_params = r'../results/preprocessed_data/2_1_remaining_params_v1.csv'
df_params = pd.read_csv(file_path_df_params)

# Convert back to list of tuples if necessary
remaining_params = list(df_params.to_records(index=False))


part_size = len(remaining_params)//4

# params = remaining_params[:part_size]
# list2 = remaining_params[part_size:2*part_size]
# list3 = remaining_params[2*part_size:3*part_size]
params = remaining_params[3*part_size:]

#%% Loop
# Create an empty dataframe to store the results

results_df = pd.DataFrame(columns=['lay1_neurons',
                                   'lay2_neurons', 
                                   'activation_function',
                                    'dropout_rates', 
                                    'batch_size',
                                    'learning_rate',
                                    'decay_rate',
                                    'val_rmse',
                                    'test_rmse'])
    
i = 0
# Train and evaluate the model for each combination of hyperparameters
for my_params in params:
    
    i= i+1
    
    # split data
    xy_train_val, xy_test = my_utils.split_within_PtID(df, numb_values_to_remove=-672, seperate_target=False, group_column=group_name) # split witin  PtID: 4values/hour * 24hour/day * 7days/week = 672 values/week

    x_train, y_train, x_val, y_val = my_utils.split_within_PtID_ratio(xy_train_val,percentage_to_remove = 0.10, seperate_target=True, group_column=group_name)

    x_test, y_test = my_utils.seperate_the_target(xy_test, group_column=group_name)


    #

    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)

    x_train = scaler_x.transform(x_train)
    x_val = scaler_x.transform(x_val)
    x_test = scaler_x.transform(x_test)


    scaler_y = MinMaxScaler()

    y_train_reshaped = y_train.values.reshape(-1, 1)
    scaler_y.fit(y_train_reshaped)

    # Transform the datasets
    y_train = scaler_y.transform(y_train_reshaped)

    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))   
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))  


    # Transform the datasets
     


    #                    Transform input to the rnn
    x_train = pd.DataFrame(x_train)
    x_val = pd.DataFrame(x_val)
    x_test = pd.DataFrame(x_test)

    x_train = my_utils.get_rnn_input(x_train)
    x_val = my_utils.get_rnn_input(x_val)
    x_test = my_utils.get_rnn_input(x_test)

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],1)))
    model.add(LSTM(my_params[0], activation=my_params[2], return_sequences=True, kernel_initializer='glorot_uniform'))  # First LSTM layer
    model.add(LSTM(my_params[1], activation=my_params[2], kernel_initializer='glorot_uniform'))  # Second LSTM layer
    model.add(Dropout(my_params[3]))  # Dropout
    model.add(Dense(1, activation="relu", kernel_initializer='glorot_uniform', name='output'))  # Output layer

    print('------------------ ', i ,'-------------------')

    # Exponential decay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=my_params[5],
        decay_steps=1000,
        decay_rate=my_params[6],  # Convert linear decay rate to exponential form
    )

    # Optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])

# Define early stopping callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(x_train, y_train, epochs=100, batch_size=my_params[4], validation_data=(x_val, y_val), callbacks=[early_stop])
    
    
    # Validation
    
    # Done for validation
    y_pred_val = model.predict(x_val)
    y_pred_val2 = model.predict(x_val)
    y_val= scaler_y.inverse_transform(y_val)
    y_pred_val= scaler_y.inverse_transform(y_pred_val)
    
    rmse_val = my_utils.calculate_results(y_val, y_pred_val)
    
    
    # Done for testing
    y_pred_test = model.predict(x_test)
    y_pred_test2 = model.predict(x_test)
    y_test= scaler_y.inverse_transform(y_test)
    y_pred_test= scaler_y.inverse_transform(y_pred_test)
    
    rmse_test = my_utils.calculate_results(y_test, y_pred_test)
    
    new_row_df = pd.DataFrame([{'lay1_neurons': my_params[0],
                                    'lay2_neurons': my_params[1], 
                                    'activation_function': my_params[2],
                                    'dropout_rates': my_params[3], 
                                    'batch_size': my_params[4],
                                    'learning_rate': my_params[5],
                                    'decay_rate': my_params[6],
                                    'val_rmse': rmse_val,
                                    'test_rmse': rmse_test }])
    
    # Concatenate the new row with the existing DataFrame
    # results_df = pd.concat([results_df, new_row_df], ignore_index=True)
    
    import os
    
    # Define your directory and file path
    directory = '../results/preprocessed_data'
    file_path = os.path.join(directory, '2_0_hyperparametertuning_v2_4.csv')
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Now you can append to the CSV with the correct usage of os.path.isfile for checking the file existence
    new_row_df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)

    
