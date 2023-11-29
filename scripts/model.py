#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:19:13 2023

@author: au605715
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import my_utils
import pickle
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
#%%                     load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//cnn_ws60min_ph60min.csv')
df.dropna(inplace=True)

# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/data_split.pkl"

# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)


#%%                     extract data from dictionary


histories_saved = []
dict_results = {}

i = 0

for (PtID, percentage), value in dictionary.items():
    print(i)
    i=i+1
    ptid_training_w = value['training_w']
    ptid_training_b = value['training_b']
    ptid_test_b = value['test_b']
    

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
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    
    x_train_scal = scaler.transform(x_train)
    x_val_scal = scaler.transform(x_val)
    
    x_test_w_scal = scaler.transform(x_test_w)
    x_test_b_scal = scaler.transform(x_test_b)
    
    # finetuning: min max normalization
    scaler_tl = MinMaxScaler()
    scaler_tl.fit(x_train_tl_w)
    
    x_train_tl_w_scal = scaler_tl.transform(x_train_tl_w)
    x_train_tl_b_scal = scaler_tl.transform(x_train_tl_b)
    x_val_tl_w_scal = scaler_tl.transform(x_val_tl_w)
    x_val_tl_b_scal = scaler_tl.transform(x_val_tl_b)
    
    x_test_tl_w_scal = scaler_tl.transform(x_test_tl_w)
    x_test_tl_b_scal = scaler_tl.transform(x_test_tl_b)
#%%                     Transform input to the cnn
    x_train = pd.DataFrame(x_train_scal)
    x_val = pd.DataFrame(x_val_scal)
    x_test_w = pd.DataFrame(x_val_scal)
    x_test_b = pd.DataFrame(x_val_scal)
    
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
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
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
    model_tl_w.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tl_learning_rate),  # Use a smaller learning rate for fine-tuning
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
    model_tl_b.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tl_learning_rate),  # Use a smaller learning rate for fine-tuning
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    

    early_stop_tl_b = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history_tl_b = model_tl_w.fit(x_train_tl_b, y_train_tl_b, epochs=20, batch_size=64, validation_data=(x_val_tl_b, y_val_tl_b), callbacks=[early_stop_tl_b])



    y_pred_test_tlb_w = model_tl_b.predict(x_test_tl_w)
    y_pred_test_tlb_b = model_tl_b.predict(x_test_tl_b)


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


        
    }
    