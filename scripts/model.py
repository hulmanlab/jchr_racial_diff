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
    
    # ptid_training_w = dictionary[patient_id, percentage]['training_w']
    # ptid_training_b = dictionary[patient_id, percentage]['training_b']
    # ptid_test_b = dictionary[patient_id, percentage]['test_b']
    

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
    
    # split within patients
    # x_train2, y_train2, x_val2, y_val2 = my_utils.split_within_PtID(df_train, numb_values_to_remove=-672)                                           # split witin  PtID
                                                                                                                                                    # 4values/hour * 24hour/day*7days/week = 672 values/week

#%%                     Scale data
    # min max normalization [0,1]
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    
    x_train_scal = scaler.transform(x_train)
    x_val_scal = scaler.transform(x_val)
    
    x_test_w_scal = scaler.transform(x_test_w)
    x_test_b_scal = scaler.transform(x_test_b)
    
    
#%%                     Transform input to the cnn
    x_train = pd.DataFrame(x_train_scal)
    x_val = pd.DataFrame(x_val_scal)
    x_test_w = pd.DataFrame(x_test_w_scal)
    x_test_b = pd.DataFrame(x_test_b_scal)
    
    x_train = my_utils.get_cnn1d_input(x_train)
    x_val = my_utils.get_cnn1d_input(x_val)
    x_test_w = my_utils.get_cnn1d_input(x_test_w)
    x_test_b = my_utils.get_cnn1d_input(x_test_b)
    

#%%                     Define the model architecture
    
    # Define early stopping callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#%%                     model and evaluation

    model = my_utils.create_cnn((x_train.shape[1],1))

    
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stop])
    # history = model.fit(x_train1, y_train1, epochs=60, batch_size=64, validation_data=(x_val1, y_val1))#, callbacks=[early_stop])


# model2 = my_utils.create_rnn()
#%%                     Plot training & validation loss
    
    histories_saved.append(history)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    plt.show()
    

    y_pred_test_b = model.predict(x_test_b)
    y_pred_test_w = model.predict(x_test_w)
    
    baseline_test_w = df_test_w['Value_4']
    baseline_test_b = df_test_b['Value_4']
    
    baseline_test_w.reset_index(drop=True, inplace=True)
    baseline_test_b.reset_index(drop=True, inplace=True)


# save results
    dict_results[(PtID, percentage)] = {
        "y_test_w": y_test_w,
        "y_test_b": y_test_b,
        "y_pred_w": y_pred_test_w,
        "y_pred_b": y_pred_test_b,
        "y_last_val_w": baseline_test_w,
        "y_last_val_b": baseline_test_b
    }


# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/predicted_results_cnn1_v2.pkl"

# Write to file
with open(file_path, 'wb') as file:
    pickle.dump(dict_results, file)
