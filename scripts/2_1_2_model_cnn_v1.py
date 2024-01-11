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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras

def split_data_black_white_ratio_in_loop (df_group1, df_group2, ratio):

    """
    split data depending on the ratio of PtIDs from each group
    
    Parameters
    ----------
    df_group1 : Dataframe
        df containing data from group1
    df_group2 : Dataframe
        df containing data from group2.
    ratio : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train : TYPE
        data to train on.
    y_train : TYPE
        labels for data to train on.
    x_val : TYPE
        data to validate on (or test).
    y_val : TYPE
        labels for data to validate on (or test).

    """
    # split based on PtID
    if percentage == 100: #  if there are no patients in group 2
    
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_test_idx = next(gss.split(df_group1, groups=df_group1['PtID']))
    
        x_train_temp1 = df_group1.iloc[train_idx]
        x_val_temp1 = df_group1.iloc[val_test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp1, x_val_temp1, group_column ="Gender")
    
    elif percentage == 0: # if there are no patients in group 1
     
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
        train_idx, test_idx = next(gss.split(df_group2, groups=df_group2['PtID']))
    
        x_train_temp2 = df_group2.iloc[train_idx]
        x_val_temp2 = df_group2.iloc[test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp2, x_val_temp2, group_column ="Gender")
        
    else: # if there are patients in group 2
        x_train, y_train, x_val, y_val = my_utils.get_groupShuflesplit_equal_groups(df_group1, df_group2,test_size=0.2, seperate_target=True, group_column ="Gender")     # split based on PtID
    
    return x_train, y_train, x_val, y_val

#                     load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//1_2_2_cnn_gender_ws60min_ph60min.csv')
df.dropna(inplace=True)

# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_3_2_data_split_m_v1.pkl"

# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)


#%%                     extract data from dictionary

testing_mode = False
group1_is_male = True
histories_saved = []
dict_results = {}

i = 0

for (PtID, percentage), value in dictionary.items():
    print(i)
    i=i+1
    
#%%                     get data from dictionary
    ptid_training_w = value['training_m']
    ptid_training_b = value['training_f']
    ptid_test_b = value['test_f']
    ptid_test_w = value['test_m']
    

    df_train_w = df[df['PtID'].isin(ptid_training_w)]
    df_train_b = df[df['PtID'].isin(ptid_training_b)]
    df_test_w = df[df['PtID']==ptid_test_w]
    df_test_b = df[df['PtID']==ptid_test_b]

#%%                        split dataset
    if group1_is_male==True:
        x_train, y_train, x_val, y_val = split_data_black_white_ratio_in_loop(df_train_w, df_train_b, percentage)
    else: # if group 1 is black
        x_train, y_train, x_val, y_val = split_data_black_white_ratio_in_loop(df_train_b, df_train_w, percentage)  
    
    # split test set x and y
    x_test_w, y_test_w = my_utils.seperate_the_target(df_test_w, group_column ="Gender")
    x_test_b, y_test_b = my_utils.seperate_the_target(df_test_b, group_column ="Gender")
    
    #%%                     Fine- tuning: split data

    # split within patients, train/test
    xy_train_tl_w, xy_test_tl_w = my_utils.split_within_PtID(df_test_w, numb_values_to_remove=-672, seperate_target=False, group_column ="Gender") # split witin  PtID
    xy_train_tl_b, xy_test_tl_b = my_utils.split_within_PtID(df_test_b, numb_values_to_remove=-672, seperate_target=False, group_column ="Gender") # 4values/hour * 24hour/day*7days/week = 672 values/week
    
    # split train in train/val with seperate targets
    x_train_tl_w, y_train_tl_w, x_val_tl_w, y_val_tl_w = my_utils.split_time_series_data(xy_train_tl_w, test_size=0.15, group_column ="Gender")
    x_train_tl_b, y_train_tl_b, x_val_tl_b, y_val_tl_b = my_utils.split_time_series_data(xy_train_tl_b, test_size=0.15, group_column ="Gender")
    
    # seperate target from test
    x_test_tl_w, y_test_tl_w = my_utils.seperate_the_target(xy_test_tl_w, group_column ="Gender")
    x_test_tl_b, y_test_tl_b = my_utils.seperate_the_target(xy_test_tl_b, group_column ="Gender")
    
    


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
    
    
    x_train_tl_w = my_utils.get_cnn1d_input(x_train_tl_w)
    x_train_tl_b = my_utils.get_cnn1d_input(x_train_tl_b)
    
    x_val_tl_w = my_utils.get_cnn1d_input(x_val_tl_w)
    x_val_tl_b = my_utils.get_cnn1d_input(x_val_tl_b)
    
    x_test_tl_w = my_utils.get_cnn1d_input(x_test_tl_w)
    x_test_tl_b = my_utils.get_cnn1d_input(x_test_tl_b)
    

#%%                     Model and evaluation

    model_base = my_utils.create_cnn1(x_train.shape[1])
    # history = model.fit(x_train1, y_train1, epochs=60, batch_size=64, validation_data=(x_val1, y_val1))#, callbacks=[early_stop])
    
    # Compile the model
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model_base.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    
    # Define early stopping callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model_base.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stop])
    print ('now testing')
    y_pred_test_w = model_base.predict(x_test_w)   
    y_pred_test_b = model_base.predict(x_test_b)
    print ('now baseline')
    
    baseline_test_w = df_test_w['Value_12']
    baseline_test_b = df_test_b['Value_12']
    
    baseline_test_w.reset_index(drop=True, inplace=True)
    baseline_test_b.reset_index(drop=True, inplace=True)
    print ('now baseline')
    
    
#%%                        Fine-tune the model   

    tl_learning_rate = 0.0001
    
    
    baseline_test_tl_w = xy_test_tl_w['Value_12']
    baseline_test_tl_b = xy_test_tl_b['Value_12']
    
    baseline_test_tl_w.reset_index(drop=True, inplace=True)
    baseline_test_tl_b.reset_index(drop=True, inplace=True)
    
    print ('now finetuning')
 
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
    print('now testing')


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

    print('scale back 1')
    
    #scale target back
    y_test_w = scaler_y.inverse_transform(y_test_w)
    y_test_b = scaler_y.inverse_transform(y_test_b)
    print('scale back 2')  
    y_test_tl_w = scaler_tl_y_w.inverse_transform(y_test_tl_w)
    y_test_tl_b = scaler_tl_y_b.inverse_transform(y_test_tl_b)
    print('scale back 3')
    # Scale predicted back
    y_pred_test_w = scaler_y.inverse_transform(y_pred_test_w)
    y_pred_test_b = scaler_y.inverse_transform(y_pred_test_b)
    
    y_pred_test_tlw_w = scaler_tl_y_w.inverse_transform(y_pred_test_tlw_w)
    y_pred_test_tlw_b = scaler_tl_y_b.inverse_transform(y_pred_test_tlw_b)
    
    y_pred_test_tlb_w = scaler_tl_y_b.inverse_transform(y_pred_test_tlb_w)

    
    #%% My actual/true values and my baseline value
    print('transform everything into arrays')
    y_actual_w = y_test_w
    y_actual_b = y_test_b 
    y_last_val_w = baseline_test_w.to_numpy()
    y_last_val_b = baseline_test_b.to_numpy()
    
    y_actual_tl_w = y_test_tl_w
    y_actual_tl_b = y_test_tl_b
    y_last_val_tl_w = baseline_test_tl_w.to_numpy()
    y_last_val_tl_b = baseline_test_tl_b.to_numpy()
        
    #%% calculating my values for each prediction base_model
    # print('calc results')
    # rmse_base_w = my_utils.calculate_results(y_actual_w, y_last_val_w)
    # rmse_base_b = my_utils.calculate_results(y_actual_b, y_last_val_b)   
    
    # #%%
    # # y_pred_w = value['y_pred_w']
    # rmse_w = my_utils.calculate_results(y_actual_w, y_pred_test_w)

    
    # # y_pred_b = value['y_pred_b']    
    # rmse_b = my_utils.calculate_results(y_actual_b, y_pred_test_b)
 
    
    # #%% Transferlearned model
    
    # # baseline values
    # rmse_base_tl_w = my_utils.calculate_results(y_actual_tl_w, y_last_val_tl_w)      
    # rmse_base_tl_b = my_utils.calculate_results(y_actual_tl_b, y_last_val_tl_b)   
    
    # # transfer learned on white
    # # y_pred_tlw_w = value['y_pred_tlw_w']
    # rmse_tlw_w  = my_utils.calculate_results(y_actual_tl_w, y_pred_test_tlw_w)

    # # y_pred_tlw_b = value['y_pred_tlw_b']
    # rmse_tlw_b  = my_utils.calculate_results(y_actual_tl_b, y_pred_test_tlw_b)
 

    # # transferlearned on black
    # # y_pred_tlb_w = value['y_pred_tlb_w']
    # rmse_tlb_w = my_utils.calculate_results(y_actual_tl_w, y_pred_test_tlb_w)

    # # y_pred_tlb_b = value['y_pred_tlb_b']
    # rmse_tlb_b = my_utils.calculate_results(y_actual_tl_b, y_pred_test_tlb_b)
    
    
    #%% save results
    print('save in dict')
    dict_results[(PtID, percentage)] = {
        "y_last_val_m": baseline_test_w.to_numpy(),
        "y_last_val_f": baseline_test_b.to_numpy(),
        "y_test_m": y_test_w,
        "y_test_f": y_test_b,
        
        "y_pred_m": y_pred_test_w,
        "y_pred_f": y_pred_test_b,

        
        "y_last_val_tl_m": baseline_test_tl_w.to_numpy(),
        "y_last_val_tl_f": baseline_test_tl_b.to_numpy(),
        "y_test_tl_m": y_test_tl_w,
        "y_test_tl_f": y_test_tl_b,
        
        "y_pred_tlw_m": y_pred_test_tlw_w,
        "y_pred_tlw_f": y_pred_test_tlw_b,
        
        "y_pred_tlb_m": y_pred_test_tlb_w,
        "y_pred_tlb_f": y_pred_test_tlb_b,
        
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
        
                
        # "rmse_base_m": rmse_base_w,
        # "rmse_base_f": rmse_base_b, 
        
        # "rmse_m": rmse_w,        
        # "rmse_f": rmse_b, 

        
        # "rmse_base_tl_m": rmse_base_tl_w, # baseline for transferlearned white
        # "rmse_base_tl_f": rmse_base_tl_b, # baseline for transferlearned black

        # "rmse_tlw_m": rmse_tlw_w, # transferlearned white
        # "rmse_tlw_f": rmse_tlw_b, 
        # "rmse_tlb_m": rmse_tlb_w, # transfer learned black    
        # "rmse_tlb_f": rmse_tlb_b, 

        }
    
    
    if testing_mode==True:
        break
    
    
#%%
print('loop finished')

# data = []

# for (PtID, percentage), metrics in dictionary.items():
#     row = {'PtID': PtID, 'percentage': percentage}
#     # Update row with keys that contain 'rmse' and do not contain 'mae' or 'mard'
#     row.update({k: metrics[k] for k in metrics if 'rmse' in k and not ('mae' in k or 'mard' in k or 'loss' in k or 'val' in k or 'y_' in k or 'train' in k)})
#     data.append(row)


# df = pd.DataFrame(data)
# df.rename(columns={'percentage': "ratio_w"}, inplace=True)

    

#%% save results


# # Specify the file path

# if testing_mode == False:
#     file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/2_1_predicted_results_cnn1_v7.pkl"


#     with open(file_path, 'wb') as file:
#         # Serialize and save the list to the file
#         pickle.dump(dict_results, file)
    

    
    
    
