#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:19:13 2023

@author: au605715
"""
#%%
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
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp1, x_val_temp1)
    
    elif percentage == 0: # if there are no patients in group 1
     
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
        train_idx, test_idx = next(gss.split(df_group2, groups=df_group2['PtID']))
    
        x_train_temp2 = df_group2.iloc[train_idx]
        x_val_temp2 = df_group2.iloc[test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp2, x_val_temp2)
        
    else: # if there are patients in group 2
        x_train, y_train, x_val, y_val = my_utils.get_groupShuflesplit_equal_groups(df_group1, df_group2,test_size=0.2, seperate_target=True)     # split based on PtID
    
    return x_train, y_train, x_val, y_val


#                    load data
# df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//1_2_cnn_ws60min_ph60min.csv')
file_path_df = r'/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_2_cnn_ws120min_ph60min.csv'

df = pd.read_csv(file_path_df)

df.dropna(inplace=True)

# Specify the file path
# file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_3_1_data_split_wb_v3.pkl"
file_path = "/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_3_1_data_split_wb_v3.pkl"
# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)
    
#%%                     extract data from dictionary

testing_mode = False
dict_results = {}

i = 0

for (PtID, percentage), value in dictionary.items():
    print(i)
    i=i+1
    
#                     get data from dictionary

    ptid_training_w = value['training_w']
    ptid_training_b = value['training_b']
    ptid_test = value['PtID_test']
    ptid_race = value['race']

    df_train_w = df[df['PtID'].isin(ptid_training_w)]
    df_train_b = df[df['PtID'].isin(ptid_training_b)]
    df_test = df[df['PtID']==ptid_test]
    


#%%                     split dataset    
    
    x_train, y_train, x_val, y_val = split_data_black_white_ratio_in_loop(df_train_w, df_train_b, percentage)
   
    
    # split test set x and y
    x_test, y_test = my_utils.seperate_the_target(df_test)

    print('test1')
    #%%                     Fine- tuning: split data

    # split within patients, train/test
    xy_train_tl, xy_test_tl = my_utils.split_within_PtID(df_test, numb_values_to_remove=-672, seperate_target=False) # split witin  PtID: 4values/hour * 24hour/day*7days/week = 672 values/week
    
    # split train in train/val with seperate targets
    x_train_tl, y_train_tl, x_val_tl, y_val_tl = my_utils.split_time_series_data(xy_train_tl, test_size=0.15)

    
    # seperate target from test
    x_test_tl, y_test_tl = my_utils.seperate_the_target(xy_test_tl)
    

#%%                     Scale data
    # min max normalization [0,1]
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    
    x_train_scal = scaler_x.transform(x_train)
    x_val_scal = scaler_x.transform(x_val)
    
    x_test_scal = scaler_x.transform(x_test)
    

    # finetuning: min max normalization
    scaler_tl_x = MinMaxScaler()
    scaler_tl_x.fit(x_train_tl)
    
    x_train_tl_scal = scaler_tl_x.transform(x_train_tl)
    x_val_tl_scal = scaler_tl_x.transform(x_val_tl)
    
    x_test_tl_scal = scaler_tl_x.transform(x_test_tl)

#%%                 Scale y data
    scaler_y = MinMaxScaler()

    # Reshape and then fit
    y_train_reshaped = y_train.values.reshape(-1, 1)
    scaler_y.fit(y_train_reshaped)
    
    # Transform the datasets
    y_train = scaler_y.transform(y_train_reshaped)
    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))   
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))


    
    scaler_tl_y = MinMaxScaler()
    # Reshape and fit the scaler on the training data
    y_train_tl_reshaped = y_train_tl.values.reshape(-1, 1)
    scaler_tl_y.fit(y_train_tl_reshaped)
    
    # Transform the datasets using the fitted scaler
    y_train_tl = scaler_tl_y.transform(y_train_tl_reshaped)
    y_val_tl = scaler_tl_y.transform(y_val_tl.values.reshape(-1, 1))
    y_test_tl = scaler_tl_y.transform(y_test_tl.values.reshape(-1, 1))
    

#%%                     Transform input to the rnn
    x_train = pd.DataFrame(x_train_scal)
    x_val = pd.DataFrame(x_val_scal)
    x_test = pd.DataFrame(x_test_scal)
    
    x_train = my_utils.get_rnn_input(x_train)
    x_val = my_utils.get_rnn_input(x_val)
    
    x_test = my_utils.get_rnn_input(x_test)
    
    
    # input to cnn_tl
    x_train_tl = pd.DataFrame(x_train_tl_scal)

    x_val_tl = pd.DataFrame(x_val_tl_scal)
    
    x_test_tl_w = pd.DataFrame(x_test_tl_scal)

    x_train_tl = my_utils.get_rnn_input(x_train_tl)

    x_val_tl = my_utils.get_rnn_input(x_val_tl)
    
    x_test_tl = my_utils.get_rnn_input(x_test_tl)


#%%                     Model and evaluation

    model_base = my_utils.create_lstm_vanDoorn((x_train.shape[1]))
    
    # Compile the model
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay = 0.001)

    model_base.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    

    # Define early stopping callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model_base.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stop])

    y_pred_test = model_base.predict(x_test)   

    baseline_test = df_test['Value_4']
   
    baseline_test.reset_index(drop=True, inplace=True)

    
#%%                        Fine-tune the model   

    tl_learning_rate = 0.0001
    
    baseline_test_tl = xy_test_tl['Value_4']
    baseline_test_tl.reset_index(drop=True, inplace=True)

    model_tl = model_base
    for layer in model_tl.layers[:-2]:  # This freezes all layers except the last two dense layers
        layer.trainable = False
    
    # The last two dense layers are left unfrozen for fine-tuning
    
    # Recompile the model with a smaller learning rate
    model_tl.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=tl_learning_rate),  # Use a smaller learning rate for fine-tuning
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    

    early_stop_tl = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history_tl = model_tl.fit(x_train_tl, y_train_tl, epochs=100, batch_size=64, validation_data=(x_val_tl, y_val_tl), callbacks=[early_stop_tl])



    y_pred_test_tl = model_tl.predict(x_test_tl)



#%% Scale y_label back

    
    #scale target back
    y_test = scaler_y.inverse_transform(y_test)
  
    y_test_tl = scaler_tl_y.inverse_transform(y_test_tl)
    
    # Scale predicted back
    y_pred_test = scaler_y.inverse_transform(y_pred_test)
    
    y_pred_test_tl = scaler_tl_y.inverse_transform(y_pred_test_tl)
    
    #%% My actual/true values and my baseline value
    y_actual = y_test
    y_last_val = baseline_test.to_numpy()
 
    y_actual_tl = y_test_tl
    y_last_val_tl = baseline_test_tl.to_numpy()
        
    #%% calculating my values for each prediction base_model
    rmse_base = my_utils.calculate_results(y_actual, y_last_val)
    
    #%%
    # y_pred_w = value['y_pred_w']
    rmse = my_utils.calculate_results(y_actual, y_pred_test)
    
    #%% Transferlearned model
    
    # baseline values
    rmse_base_tl = my_utils.calculate_results(y_actual_tl, y_last_val_tl)      

    
    # transfer learned on white
    # y_pred_tlw_w = value['y_pred_tlw_w']
    rmse_tl  = my_utils.calculate_results(y_actual_tl, y_pred_test_tl)




    dict_results[(PtID, percentage)] = {
        "race": ptid_race,
        "y_last_val": baseline_test.to_numpy(),
        "y_actual": y_test,
        
        "y_pred": y_pred_test,

        "y_last_val_tl": baseline_test_tl.to_numpy(),
        "y_actual_tl": y_test_tl,

        "y_pred_tl": y_pred_test_tl,
        
        "rmse_base": rmse_base,
        "rmse": rmse,
        
        "rmse_base_tl": rmse_base_tl, # baseline for transferlearned
        "rmse_tl": rmse_tl, # transferlearned
        
        
        'epochs': len(history.history['loss']),
        'epochs_tl': len(history_tl.history['loss']),

        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_mse': history.history['mean_squared_error'],
        'val_mse': history.history['val_mean_squared_error'],
        'train_mae': history.history['mean_absolute_error'],
        'val_mae': history.history['val_mean_absolute_error'],
        
        'loss_tl': history_tl.history['loss'],
        'val_loss_tl': history_tl.history['val_loss'],
        'train_mse_tl': history_tl.history['mean_squared_error'],
        'val_mse_tl': history_tl.history['val_mean_squared_error'],
        'train_mae_tl': history_tl.history['mean_absolute_error'],
        'val_mae_tl': history_tl.history['val_mean_absolute_error'],
                


        }
    
    
    if testing_mode==True:
        break
    #%%
    if testing_mode == False:

        file_path = f"/home/hbt/jchr_data/jchr_racial_diff/results/processed_data/2_1_1_predicted_results_rnn_wb_2h/patient{PtID}_ratio{percentage}.pkl"


        with open(file_path, 'wb') as file:
            # Serialize and save the list to the file
            pickle.dump(dict_results, file)
        
    dict_results = {}
    
    
#%%


# data = []

# for (PtID, percentage), metrics in dictionary.items():
#     row = {'PtID': PtID, 'percentage': percentage}
#     # Update row with keys that contain 'rmse' and do not contain 'mae' or 'mard'
#     row.update({k: metrics[k] for k in metrics if 'rmse' in k and not ('mae' in k or 'mard' in k or 'loss' in k or 'val' in k or 'y_' in k or 'train' in k)})
#     data.append(row)


# df = pd.DataFrame(data)
# # df.rename(columns={'percentage': "ratio_w"}, inplace=True)
    
    
