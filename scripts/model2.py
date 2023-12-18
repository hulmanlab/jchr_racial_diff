#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:19:13 2023

@author: au605715
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import my_utils
import pickle
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
import math 

#%%                     load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//1_2_cnn_ws60min_ph60min.csv')
df.dropna(inplace=True)

# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_3_data_split.pkl"

# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)



# ptid_test_b = {}
# ptid_training = {}
# ptid_ratio_w = {}

# for PtID, nested_dict in dictionary.items():
#     ptid_test_b[PtID] = nested_dict["test_b"]
#     ptid_training[PtID] = nested_dict["training"]
#     ptid_ratio_w[PtID] = nested_dict["ratio_w"]

#%%                     extract data from dictionary
PtID = 3
percentage = 100

iterations = 1

# histories_saved = []
dict_results = {}

for i in range(iterations):
    ptid_training_w = dictionary[PtID, percentage]['training_w']
    ptid_training_b = dictionary[PtID, percentage]['training_b']
    ptid_test_b = dictionary[PtID, percentage]['test_b']
    

    df_train_w = df[df['PtID'].isin(ptid_training_w)]
    df_train_b = df[df['PtID'].isin(ptid_training_b)]
    df_test_w = df[df['PtID']==PtID]
    df_test_b = df[df['PtID']==ptid_test_b]   
    
    df_train = pd.concat([df_train_w, df_train_b])
    df_train.reset_index(drop=True, inplace=True)


#%%                        split dataset

    # split based on PtID
    if percentage == 100: #  if there are no patients in group 2
    
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df_train_w, groups=df_train_w['PtID']))
    
        x_train_temp1 = df_train_w.iloc[train_idx]
        x_val_temp1 = df_train_w.iloc[test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp1, x_val_temp1)
    
    elif percentage == 0: # if there are no patients in group 1
     
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
        train_idx, test_idx = next(gss.split(df_train_b, groups=df_train_w['PtID']))
    
        x_train_temp2 = df_train_b.iloc[train_idx]
        x_val_temp2 = df_train_b.iloc[test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp2, x_val_temp2)
        
    else: # if there are patients in group 2
        x_train, y_train, x_val, y_val = my_utils.get_groupShuflesplit_equal_groups(df_train_w, df_train_b,test_size=0.2, seperate_target=True)     # split based on PtID
    
    
    # split test set
    x_test_w, y_test_w = my_utils.seperate_the_target(df_test_w)
    x_test_b, y_test_b = my_utils.seperate_the_target(df_test_b)
    
#%%                        Fine- tuning: split data

    # split within patients, train/test
    xy_train_tl_w, xy_test_tl_w = my_utils.split_within_PtID(df_test_w, numb_values_to_remove=-672, seperate_target=False)                                           # split witin  PtID
    xy_train_tl_b, xy_test_tl_b = my_utils.split_within_PtID(df_test_b, numb_values_to_remove=-672, seperate_target=False)                                           # 4values/hour * 24hour/day*7days/week = 672 values/week
    
    # split train in train/val with seperate targets
    x_train_tl_w, y_train_tl_w, x_val_tl_w, y_val_tl_w = my_utils.split_time_series_data(xy_train_tl_w, test_size=0.15)
    x_train_tl_b, y_train_tl_b, x_val_tl_b, y_val_tl_b = my_utils.split_time_series_data(xy_train_tl_b, test_size=0.15)
    
    # seperate target from test
    x_test_tl_w, y_test_tl_w = my_utils.seperate_the_target(xy_test_tl_w)
    x_test_tl_b, y_test_tl_b = my_utils.seperate_the_target(xy_test_tl_b)
    

#%%                     Scale data
    # min max normalization [0,1]
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    
    x_train_scal = scaler_x.transform(x_train)
    x_val_scal = scaler_x.transform(x_val)
    
    x_test_w_scal = scaler_x.transform(x_test_w)
    x_test_b_scal = scaler_x.transform(x_test_b)
    

    # finetuning: min max normalization
    scaler_tl_x = MinMaxScaler()
    scaler_tl_x.fit(x_train_tl_w)
    
    x_train_tl_w_scal = scaler_tl_x.transform(x_train_tl_w)
    x_train_tl_b_scal = scaler_tl_x.transform(x_train_tl_b)
    x_val_tl_w_scal = scaler_tl_x.transform(x_val_tl_w)
    x_val_tl_b_scal = scaler_tl_x.transform(x_val_tl_b)
    
    x_test_tl_w_scal = scaler_tl_x.transform(x_test_tl_w)
    x_test_tl_b_scal = scaler_tl_x.transform(x_test_tl_b)

#%%                 Scale y data

    
    scaler_y = MinMaxScaler()

    # Reshape and then fit
    y_train_reshaped = y_train.values.reshape(-1, 1)
    scaler_y.fit(y_train_reshaped)
    
    # Transform the datasets
    y_train = scaler_y.transform(y_train_reshaped)
    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))   
    y_test_w = scaler_y.transform(y_test_w.values.reshape(-1, 1))
    y_test_b = scaler_y.transform(y_test_b.values.reshape(-1, 1))


    
    scaler_tl_y_w = MinMaxScaler()
    # Reshape and fit the scaler on the training data
    y_train_tl_w_reshaped = y_train_tl_w.values.reshape(-1, 1)
    scaler_tl_y_w.fit(y_train_tl_w_reshaped)
    
    # Transform the datasets using the fitted scaler
    y_train_tl_w = scaler_tl_y_w.transform(y_train_tl_w_reshaped)
    y_val_tl_w = scaler_tl_y_w.transform(y_val_tl_w.values.reshape(-1, 1))
    y_test_tl_w = scaler_tl_y_w.transform(y_test_tl_w.values.reshape(-1, 1))
    
    
    
    scaler_tl_y_b = MinMaxScaler()
    # Reshape and fit the scaler on the training data
    y_train_tl_b_reshaped = y_train_tl_b.values.reshape(-1, 1)
    scaler_tl_y_b.fit(y_train_tl_b_reshaped)
    
    # Transform the datasets using the fitted scaler
    y_train_tl_b = scaler_tl_y_b.transform(y_train_tl_b.values.reshape(-1, 1))
    y_val_tl_b = scaler_tl_y_b.transform(y_val_tl_b.values.reshape(-1, 1))
    y_test_tl_b = scaler_tl_y_b.transform(y_test_tl_b.values.reshape(-1, 1))

    
#%%                     Transform input to the cnn
    x_train = pd.DataFrame(x_train_scal)
    x_val = pd.DataFrame(x_val_scal)
    
    x_test_w = pd.DataFrame(x_test_w_scal)
    x_test_b = pd.DataFrame(x_test_b_scal)
    
    x_train = my_utils.get_cnn1d_input(x_train)
    x_val = my_utils.get_cnn1d_input(x_val)
    
    x_test_w = my_utils.get_cnn1d_input(x_test_w)
    x_test_b = my_utils.get_cnn1d_input(x_test_b)
    
    
    # input to cnn_tl
    x_train_tl_w = pd.DataFrame(x_train_tl_w_scal)
    x_train_tl_b = pd.DataFrame(x_train_tl_b_scal)
    x_val_tl_w = pd.DataFrame(x_val_tl_w_scal)
    x_val_tl_b = pd.DataFrame(x_val_tl_b_scal)
    x_test_tl_w = pd.DataFrame(x_test_tl_w_scal)
    x_test_tl_b = pd.DataFrame(x_test_tl_b_scal)




#%%                     Model and evaluation

    model_base = my_utils.create_cnn((x_train.shape[1],1))
    # history = model.fit(x_train1, y_train1, epochs=60, batch_size=64, validation_data=(x_val1, y_val1))#, callbacks=[early_stop])
    
    # Compile the model
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model_base.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    
    # Define early stopping callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model_base.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stop])

    y_pred_test_w = model_base.predict(x_test_w)   
    y_pred_test_b = model_base.predict(x_test_b)

    
    baseline_test_w = df_test_w['Value_4']
    baseline_test_b = df_test_b['Value_4']
    
    baseline_test_w.reset_index(drop=True, inplace=True)
    baseline_test_b.reset_index(drop=True, inplace=True)
    
    
    
#%%                        Fine-tune the model   

    tl_learning_rate = 0.0001
    
    
    baseline_test_tl_w = xy_test_tl_w['Value_4']
    baseline_test_tl_b = xy_test_tl_b['Value_4']
    
    baseline_test_tl_w.reset_index(drop=True, inplace=True)
    baseline_test_tl_b.reset_index(drop=True, inplace=True)
    
    

    #%% white
    model_tl_w = model_base
    for layer in model_tl_w.layers[:-2]:  # This freezes all layers except the last two dense layers
        layer.trainable = False
    
    # The last two dense layers are left unfrozen for fine-tuning
    
    # Recompile the model with a smaller learning rate
    model_tl_w.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=tl_learning_rate),  # Use a smaller learning rate for fine-tuning
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    

    early_stop_tl_w = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history_tl_w = model_tl_w.fit(x_train_tl_w, y_train_tl_w, epochs=20, batch_size=64, validation_data=(x_val_tl_w, y_val_tl_w), callbacks=[early_stop_tl_w])



    y_pred_test_tlw_w = model_tl_w.predict(x_test_tl_w)
    y_pred_test_tlw_b = model_tl_w.predict(x_test_tl_b)


    #%% black
    model_tl_b = model_base
    for layer in model_tl_b.layers[:-2]:  # This freezes all layers except the last two dense layers
        layer.trainable = False
    
    # The last two dense layers are left unfrozen for fine-tuning
    
    # Recompile the model with a smaller learning rate
    model_tl_b.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=tl_learning_rate),  # Use a smaller learning rate for fine-tuning
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    

    early_stop_tl_b = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history_tl_b = model_tl_w.fit(x_train_tl_b, y_train_tl_b, epochs=20, batch_size=64, validation_data=(x_val_tl_b, y_val_tl_b), callbacks=[early_stop_tl_b])



    y_pred_test_tlb_w = model_tl_b.predict(x_test_tl_w)
    y_pred_test_tlb_b = model_tl_b.predict(x_test_tl_b)



#%% Scale y_label back

    
    #scale target back
    y_test_w = scaler_y.inverse_transform(y_test_w)
    y_test_b = scaler_y.inverse_transform(y_test_b)
    
    y_test_tl_w = scaler_tl_y_w.inverse_transform(y_test_tl_w)
    y_test_tl_b = scaler_tl_y_b.inverse_transform(y_test_tl_b)
    
    # Scale predicted back
    y_pred_test_w = scaler_y.inverse_transform(y_pred_test_w)
    y_pred_test_b = scaler_y.inverse_transform(y_pred_test_b)
    
    y_pred_test_tlw_w = scaler_tl_y_w.inverse_transform(y_pred_test_tlw_w)
    y_pred_test_tlw_b = scaler_tl_y_b.inverse_transform(y_pred_test_tlw_b)
    
    y_pred_test_tlb_w = scaler_tl_y_b.inverse_transform(y_pred_test_tlb_w)
    y_pred_test_tlb_b = scaler_tl_y_b.inverse_transform(y_pred_test_tlb_b)
    #%% My actual/true values and my baseline value
    y_actual_w = y_test_w
    y_actual_b = y_test_b 
    y_last_val_w = baseline_test_w.to_numpy()
    y_last_val_b = baseline_test_b.to_numpy()
    
    y_actual_tl_w = y_test_tl_w
    y_actual_tl_b = y_test_tl_b
    y_last_val_tl_w = baseline_test_tl_w.to_numpy()
    y_last_val_tl_b = baseline_test_tl_b.to_numpy()
        
    #%% calculating my values for each prediction base_model
    rmse_base_w, mae_base_w, mard_base_w  = my_utils.calculate_results(y_actual_w, y_last_val_w)
    rmse_base_b, mae_base_b, mard_base_b  = my_utils.calculate_results(y_actual_b, y_last_val_b)   
    
    #%%
    # y_pred_w = value['y_pred_w']
    rmse_w, mae_w, mard_w = my_utils.calculate_results(y_actual_w, y_pred_test_w)
    mse_w = my_utils.calculate_mse(y_actual_w,y_pred_test_w )
    mse2 = mean_squared_error(y_actual_w, y_pred_test_w)
    rmse2 = math.sqrt(mse2)
    
    # y_pred_b = value['y_pred_b']    
    rmse_b, mae_b, mard_b  = my_utils.calculate_results(y_actual_b, y_pred_test_b)
 
    
    #%% Transferlearned model
    
    # baseline values
    rmse_base_tl_w, mae_base_tl_w, mard_base_tl_w  = my_utils.calculate_results(y_actual_tl_w, y_last_val_tl_w)      
    rmse_base_tl_b, mae_base_tl_b, mard_base_tl_b  = my_utils.calculate_results(y_actual_tl_b, y_last_val_tl_b)   
    
    # transfer learned on white
    # y_pred_tlw_w = value['y_pred_tlw_w']
    rmse_tlw_w, mae_tlw_w, mard_tlw_w  = my_utils.calculate_results(y_actual_tl_w, y_pred_test_tlw_w)

    # y_pred_tlw_b = value['y_pred_tlw_b']
    rmse_tlw_b, mae_tlw_b, mard_tlw_b  = my_utils.calculate_results(y_actual_tl_b, y_pred_test_tlw_b)
 

    # transferlearned on black
    # y_pred_tlb_w = value['y_pred_tlb_w']
    rmse_tlb_w, mae_tlb_w, mard_tlb_w  = my_utils.calculate_results(y_actual_tl_w, y_pred_test_tlb_w)

    # y_pred_tlb_b = value['y_pred_tlb_b']
    rmse_tlb_b, mae_tlb_b, mard_tlb_b  = my_utils.calculate_results(y_actual_tl_b, y_pred_test_tlb_b)
    
    #%% save results

    dict_results[(PtID, percentage)] = {
        "y_last_val_w": baseline_test_w,
        "y_last_val_b": baseline_test_b,
        "y_test_w": y_test_w,
        "y_test_b": y_test_b,
        
        "y_pred_w": y_pred_test_w,
        "y_pred_b": y_pred_test_b,
        
        
        "y_last_val_tl_w": baseline_test_tl_w,
        "y_last_val_tl_b": baseline_test_tl_b,
        "y_test_tl_w": y_test_tl_w,
        "y_test_tl_b": y_test_tl_b,
        
        "y_pred_tlw_w": y_pred_test_tlw_w,
        "y_pred_tlw_b": y_pred_test_tlw_b,
        
        "y_pred_tlb_w": y_pred_test_tlb_w,
        "y_pred_tlb_b": y_pred_test_tlb_b,
        
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_mse': history.history['mean_squared_error'],
        'val_mse': history.history['val_mean_squared_error'],
        'train_mae': history.history['mean_absolute_error'],
        'val_mae': history.history['val_mean_absolute_error'],
        
        'loss_tlw': history_tl_w.history['loss'],
        'val_loss_tlw': history_tl_w.history['val_loss'],
        'train_mse_tlw': history_tl_w.history['mean_squared_error'],
        'val_mse_tlw': history_tl_w.history['val_mean_squared_error'],
        'train_mae_tlw': history_tl_w.history['mean_absolute_error'],
        'val_mae_tlw': history_tl_w.history['val_mean_absolute_error'],


        'loss_tlb': history_tl_b.history['loss'],
        'val_loss_tlb': history_tl_b.history['val_loss'],
        'train_mse_tlb': history_tl_b.history['mean_squared_error'],
        'val_mse_tlb': history_tl_b.history['val_mean_squared_error'],
        'train_mae_tlb': history_tl_b.history['mean_absolute_error'],
        'val_mae_tlb': history_tl_b.history['val_mean_absolute_error'],
        
        
        "rmse_base_w": rmse_base_w,
        "mae_base_w": mae_base_w, 
        "mard_base_w":mard_base_w, 
        "rmse_base_b": rmse_base_b, 
        "mae_base_b": mae_base_b, 
        "mard_base_b":mard_base_b, 
        
        "rmse_w": rmse_w, 
        "rmse2":rmse2,
        "mse_w": mse_w, 
        "mae_w": mae_w, 
        "mard_w":mard_w,  
        
        "rmse_b": rmse_b, 
        "mae_b": mae_b, 
        "mard_b":mard_b, 
        
        "rmse_base_tl_w": rmse_base_tl_w, # baseline for transferlearned white
        "mae_base_tl_w": mae_base_tl_w, 
        "mard_base_tl_w":mard_base_tl_w, 
        "rmse_base_tl_b": rmse_base_tl_b, # baseline for transferlearned black
        "mae_base_tl_b": mae_base_tl_b, 
        "mard_base_tl_b":mard_base_tl_b, 
        
        
        "rmse_tlw_w": rmse_tlw_w, # transferlearned white
        "mae_tlw_w": mae_tlw_w, 
        "mard_tlw_w":mard_tlw_w,   
        "rmse_tlw_b": rmse_tlw_b, 
        "mae_tlw_b": mae_tlw_b, 
        "mard_tlw_b":mard_tlw_b, 


        "rmse_tlb_w": rmse_tlb_w, # transfer learned black
        "mae_tlb_w": mae_tlb_w, 
        "mard_tlb_w":mard_tlb_w,         
        "rmse_tlb_b": rmse_tlb_b, 
        "mae_tlb_b": mae_tlb_b, 
        "mard_tlb_b":mard_tlb_b, 



        
    }
    

#%%


data = []

exclude_keys = {
    'y_last_val_w', 'y_last_val_b', 'y_test_w', 'y_test_b',
    'y_pred_w', 'y_pred_b', 'y_last_val_tl_w', 'y_last_val_tl_b',
    'y_test_tl_w', 'y_test_tl_b', 'y_pred_tlw_w', 'y_pred_tlw_b',
    'y_pred_tlb_w', 'y_pred_tlb_b', 'loss', 'val_loss', 'train_mse',
    'val_mse', 'train_mae', 'val_mae', 'loss_tlw', 'val_loss_tlw',
    'train_mse_tlw', 'val_mse_tlw', 'train_mae_tlw', 'val_mae_tlw',
    'loss_tlb', 'val_loss_tlb', 'train_mse_tlb', 'val_mse_tlb',
    'train_mae_tlb', 'val_mae_tlb'
}
data = []
for (PtID, percentage), metrics in dict_results.items():
    row = {'PtID': PtID, 'percentage': percentage}
    # Update row with keys not in the exclusion set
    row.update({k: metrics[k] for k in metrics if k not in exclude_keys})
    data.append(row)

df = pd.DataFrame(data)
df.rename(columns={'percentage': "ratio_w"}, inplace=True)



#%%


# # Specify the file path
# file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/slet1.pkl"

# # Write to file
# with open(file_path, 'wb') as file:
#     pickle.dump(dict_results, file)






# Specify the file path
# file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/predicted_results_cnn1_history_v1.pkl"




# with open(file_path, 'wb') as file:
#     # Serialize and save the list to the file
#     pickle.dump(histories_saved, file)