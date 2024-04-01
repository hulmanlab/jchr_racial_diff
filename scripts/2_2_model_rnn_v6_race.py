#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:09:25 2023

@author: au605715
"""

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
import datetime
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
file_path_df = r'/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_2_model_input_ws60min_ph60min_v6.csv'
df = pd.read_csv(file_path_df)



# Specify the file path
file_path = "/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_3_data_split_race_v6_2.pkl"

# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)
    
#%%                     extract data from dictionary

testing_mode = False
group_name = "Race" #  Gender, EduLevel
df.drop(columns=['Gender','EduLevel','ageAtEnroll'], inplace = True)
df.dropna(inplace=True)


dict_results = {}

i = 0

for (PtID, percentage), value in dictionary.items():
    print(i)
    i=i+1
    
#                     get data from dictionary

    ptid_training_gr1 = value['training_gr1']
    ptid_training_gr2 = value['training_gr2']
    ptid_test = value['PtID_test']
    ptid_group = value['group']

    df_train_gr1 = df[df['PtID'].isin(ptid_training_gr1)]
    df_train_gr2 = df[df['PtID'].isin(ptid_training_gr2)]
    df_test = df[df['PtID']==ptid_test]
    


#%%                split dataset80/20 
    
    # x_train, y_train, x_val, y_val = split_data_group_ratio_in_loop(df_train_m, df_train_f, percentage)
   

#%%                     split dataset using one week of data to validate on the rest

    df_train_gr12 = pd.concat([df_train_gr1, df_train_gr2], ignore_index=True)
    df_train_gr12.reset_index

    # split within patients, train/val
    x_train, y_train, x_val, y_val = my_utils.split_within_PtID(df_train_gr12, numb_values_to_remove=-672, seperate_target=True, group_column=group_name) # split witin  PtID: 4values/hour * 24hour/day * 7days/week = 672 values/week

    print('ged 1')
    #%%                   Fine- tuning: split data
    

    # split within patients, train/test
    xy_train_tl, xy_test_tl = my_utils.split_within_PtID(df_test, numb_values_to_remove=-672, seperate_target=False, group_column=group_name) # split witin  PtID: 4values/hour * 24hour/day*7days/week = 672 values/week

    # baseline_test_tl = xy_test_tl['Value_4']
    print('ged 2')    

    # split train in train/val with seperate targets
    x_train_tl, y_train_tl, x_val_tl, y_val_tl = my_utils.split_time_series_data(xy_train_tl, test_size=0.15, group_column=group_name)
    print('ged 3')
    
    # seperate target from test
    x_test_tl, y_test_tl = my_utils.seperate_the_target(xy_test_tl, group_column=group_name)
    print('ged 4')
#%%                   Scale data
    # min max normalization [0,1]
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    
    x_train_scal = scaler_x.transform(x_train)
    x_val_scal = scaler_x.transform(x_val)
    

    # finetuning: min max normalization
    scaler_tl_x = MinMaxScaler()
    scaler_tl_x.fit(x_train_tl)
    
    x_train_tl_scal = scaler_tl_x.transform(x_train_tl)
    x_val_tl_scal = scaler_tl_x.transform(x_val_tl)
    x_test_tl_scal = scaler_tl_x.transform(x_test_tl)

#                 Scale y data
    scaler_y = MinMaxScaler()

    # Reshape and then fit
    y_train_reshaped = y_train.values.reshape(-1, 1)
    
    scaler_y.fit(y_train_reshaped)
    
    # Transform the datasets
    y_train = scaler_y.transform(y_train_reshaped)
    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))   

    
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
    
    x_train = my_utils.get_rnn_input(x_train)
    x_val = my_utils.get_rnn_input(x_val)
    
    
    # input to cnn_tl
    x_train_tl = pd.DataFrame(x_train_tl_scal)
    x_val_tl = pd.DataFrame(x_val_tl_scal)
    x_test_tl = pd.DataFrame(x_test_tl_scal)

    x_train_tl = my_utils.get_rnn_input(x_train_tl)
    x_val_tl = my_utils.get_rnn_input(x_val_tl)
    x_test_tl = my_utils.get_rnn_input(x_test_tl)

#%%                     Model and evaluation for ONLY test person

    model_single = my_utils.create_lstm_vanDoorn_updated((x_train.shape[1]))
    
    # Compile the model
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.95

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay = 0.001)

    model_single.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    

    # Define early stopping callbacks
    early_stop_single = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history_single = model_single.fit(x_train_tl, y_train_tl, epochs=100, batch_size=1024, validation_data=(x_val_tl, y_val_tl), callbacks=[early_stop_single])

    y_pred_single = model_single.predict(x_test_tl)


#%%                     Model and evaluation

    model_base = my_utils.create_lstm_vanDoorn_updated((x_train.shape[1]))
    

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay = 0.001)

    model_base.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    

    # Define early stopping callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model_base.fit(x_train, y_train, epochs=100, batch_size=1024, validation_data=(x_val, y_val), callbacks=[early_stop])

    y_pred_test = model_base.predict(x_test_tl)

    
#%%                        Fine-tune the model   
    lr_schedule_tl = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.0001,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    )
   

    model_tl = model_base
    for layer in model_tl.layers[:-2]:  # This freezes all layers except the last two layers
        layer.trainable = False

    
    # Recompile the model with a smaller learning rate
    model_tl.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule_tl),  # Use a smaller learning rate for fine-tuning
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    

    early_stop_tl = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history_tl = model_tl.fit(x_train_tl, y_train_tl, epochs=100, batch_size=1024, validation_data=(x_val_tl, y_val_tl), callbacks=[early_stop_tl])

    y_pred_test_tl = model_tl.predict(x_test_tl)



#%% Scale y_label back

    y_test_tl = scaler_tl_y.inverse_transform(y_test_tl) # The answer
    
    
    # Scale predicted back (The predicted to compare to the answer)
    y_pred_test = scaler_tl_y.inverse_transform(y_pred_test)
    
    y_pred_test_tl = scaler_tl_y.inverse_transform(y_pred_test_tl)
    
    y_pred_single = scaler_tl_y.inverse_transform(y_pred_single)
    
    #%% My actual/true values and my baseline value
    
    y_actual_tl = y_test_tl
    # y_last_val_tl = baseline_test_tl.to_numpy()
    

    dict_results[(PtID, percentage)] = {
        'timestamp': datetime.datetime.now(),
        "PtId_test": ptid_test,
        "ptid_group": ptid_group,
        "y_actual": y_test_tl,
        "y_last_val": xy_test_tl['Value_4'],
        
        "y_pred_single": y_pred_single,      
        "y_pred": y_pred_test,
        "y_pred_tl": y_pred_test_tl,
        
        'epochs': len(history.history['loss']),
        'epochs_tl': len(history_tl.history['loss']),
        'epochs_single': len(history_single.history['loss']),
        
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
        
        'loss_single': history_single.history['loss'],
        'val_loss_single': history_single.history['val_loss'],
        'train_mse_single': history_single.history['mean_squared_error'],
        'val_mse_single': history_single.history['val_mean_squared_error'],
        'train_mae_single': history_single.history['mean_absolute_error'],
        'val_mae_single': history_single.history['val_mean_absolute_error'],

        
        }
    
    
    if testing_mode==True:
        break
    if testing_mode == False:
    #%%
    

        file_path = f"/home/hbt/jchr_data/jchr_racial_diff/results/processed_data/2_1_1_predicted_results_rnn_v6_race2/patient{PtID}_ratio{percentage}.pkl"
        
        
        with open(file_path, 'wb') as file:
            # Serialize and save the list to the file
            pickle.dump(dict_results, file)
        
    dict_results = {}
    
    
    

# %%
