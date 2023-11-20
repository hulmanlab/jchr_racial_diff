#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:11:55 2023

@author: au605715
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helene is testing stuff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from my_utils import get_groupShuflesplit_equal_groups, change_trainingset, get_cnn1d_input, create_cnn
import tensorflow as tf
#%%                     load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//cnn_ws60min_ph60min.csv')
df.dropna(inplace=True)

#%%                     splitting patients in ethnicity
df_black = df[df['Race'] == 'black']
df_white = df[df['Race'] == 'white']

#%%                     numb of patients
df_unique = df['PtID'].unique()
df_unique_white = df_white['PtID'].unique()
df_unique_black = df_black['PtID'].unique()


#%%
race = "wb"
iterations = 1

# Initialize empty arrays to store results
test_loss_array_w = np.zeros(iterations)
test_mse_array_w = np.zeros(iterations)
custom_mae_array_w = np.zeros(iterations)
baseline_mae_array_w = np.zeros(iterations)
baseline_mse_array_w = np.zeros(iterations)

test_loss_array_b = np.zeros(iterations)
test_mse_array_b = np.zeros(iterations)
custom_mae_array_b = np.zeros(iterations)
baseline_mae_array_b = np.zeros(iterations)
baseline_mse_array_b = np.zeros(iterations)

histories_saved = []

for i in range(iterations):
    print("Iteration:", i + 1)
    #%%                     split data
    # split train-validate set and test set (hold-out set)
    train, test = get_groupShuflesplit_equal_groups(df_white, df_black, test_size=0.10, random_state=42, show_distribution=True, equal_PtID=True, seperate_target=False)
    test_w = test[test['Race']=='white']
    test_b = test[test['Race']=='black']
    
    x_test_w = test_w.drop(['Target', 'PtID', 'Race'], axis=1)
    x_test_b = test_b.drop(['Target', 'PtID', 'Race'], axis=1)
    
    y_test_w = test_w['Target']
    y_test_b = test_b['Target']
    
    # get data for training and validating
    x_train, y_train, x_val, y_val = change_trainingset(train, race=race, test_size=0.10, random_state=42, equal_PtID=True, show_distribution=False)
    
    #%%                     reshape input for model
    
    x_train = get_cnn1d_input(x_train)
    x_val = get_cnn1d_input(x_val)
    x_test_w = get_cnn1d_input(x_test_w)
    x_test_b = get_cnn1d_input(x_test_b)
    
    #%%                     Define the model architecture
    
    # Define early stopping callback¢
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    #%%                     model and evaluation
    model = create_cnn((x_train.shape[1],1))
    history = model.fit(x_train, y_train, epochs=60, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stop])
    
    
    # model_save_name = f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_{iterations}iter_{race}/cnn_{iterations}iter_{race}_CNN_{i}.h5'
    # model.save(model_save_name) 

    #%%                     Plot training & validation loss
    histories_saved.append(history)
    

    # print(history.history.keys()) # print history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    plt.show()


    #%%                     Get results
    # Evaluate the model
    test_loss_w, test_mse_w = model.evaluate(x_test_w, y_test_w)
    test_loss_b, test_mse_b = model.evaluate(x_test_b, y_test_b)
    # Calculate your custom MSE
    y_pred_test_w = model.predict(x_test_w)
    y_pred_test_b = model.predict(x_test_b)
    # my_mse = mean_squared_error(y_pred_test, y_test) # same as the one from evaluate test_mse_n
    my_mae_w = mean_absolute_error(y_pred_test_w, y_test_w)
    my_mae_b = mean_absolute_error(y_pred_test_b, y_test_b)
    
    
    # Calculate baseline metrics
    sliced_array_w = x_test_w[:, :, 0]
    sliced_array_b = x_test_b[:, :, 0]
    x_test_w_df = pd.DataFrame(sliced_array_w)
    x_test_b_df = pd.DataFrame(sliced_array_b)
    x_test_w_df.columns = ['Value_1', 'Value_2', 'Value_3', 'Value_4']
    x_test_b_df.columns = ['Value_1', 'Value_2', 'Value_3', 'Value_4']
    mse_w = mean_squared_error(x_test_w_df['Value_4'], y_test_w)
    mse_b = mean_squared_error(x_test_b_df['Value_4'], y_test_b)
    mae_w = mean_absolute_error(x_test_w_df['Value_4'], y_test_w)
    mae_b = mean_absolute_error(x_test_b_df['Value_4'], y_test_b)
  
    test_loss_array_w[i] = test_loss_w
    test_mse_array_w[i] = test_mse_w
    custom_mae_array_w[i] = my_mae_w
    baseline_mse_array_w[i] = mse_w
    baseline_mae_array_w[i] = mae_w
    
    test_loss_array_b[i] = test_loss_b
    test_mse_array_b[i] = test_mse_b
    custom_mae_array_b[i] = my_mae_b
    baseline_mse_array_b[i] = mse_b
    baseline_mae_array_b[i] = mae_b

df_test_results_w = pd.DataFrame({
  
    'Loss w': test_loss_array_w,
    'MSE w': test_mse_array_w,
    'MSE base w': baseline_mse_array_w,
    'MAE w': custom_mae_array_w,
    'MAE base w': baseline_mae_array_w
})

df_test_results_b = pd.DataFrame({
  
    'Loss B': test_loss_array_b,
    'MSE B': test_mse_array_b,
    'MSE base B': baseline_mse_array_b,
    'MAE B': custom_mae_array_b,
    'MAE base B': baseline_mae_array_b
})


#%%
# Save dataframe as CSV file

# Create empty lists to store data
epochs = []
loss_values = []
mse_values = []
val_loss_values = []
val_mse_values = []
epoch_id = []

# Initialize the epoch_id
current_epoch_id = 1

# Iterate through the list of History objects
for h in histories_saved:  # Assuming 'history_list' is your list of History objects
    # Extract data from the current History object
    current_epochs = np.arange(len(history.history['loss'])) + 1
    current_loss = history.history['loss']
    current_mse = history.history['mean_squared_error']
    current_val_loss = history.history['val_loss']
    current_val_mse = history.history['val_mean_squared_error']

    # Append the data to the respective lists
    epochs.extend(current_epochs)
    loss_values.extend(current_loss)
    mse_values.extend(current_mse)
    val_loss_values.extend(current_val_loss)
    val_mse_values.extend(current_val_mse)

    # Create the epoch_id array to mark the start of each epoch
    epoch_id.extend([current_epoch_id] * len(current_epochs))

    # Increment the epoch ID for the next epoch
    current_epoch_id += 1

# Create a DataFrame from the lists
df_loss = pd.DataFrame({
    'Epoch ID': epoch_id,
    'Epoch': epochs,
    'Loss': loss_values,
    'MSE': mse_values,
    'Val Loss': val_loss_values,
    'Val MSE': val_mse_values
})

#%%
# df_test_results_w.to_csv(f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_{iterations}iter_{race}/cnn_{iterations}iter_{race}_test_resulsts_w.csv', index=False)

# df_test_results_b.to_csv(f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_{iterations}iter_{race}/cnn_{iterations}iter_{race}_test_resulsts_b.csv', index=False) 

# df_loss.to_csv(f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_{iterations}iter_{race}/cnn_{iterations}iter_{race}_history_list.csv', index=False)

#%%

df_y_w = pd.DataFrame({
    'y_test_w': test_w['Target'],
})
df_y_w['y_pred_w'] = y_pred_test_w



df_y_b = pd.DataFrame({
    'y_test_b': test_b['Target'],
})
df_y_b['y_pred_b'] = y_pred_test_b

df_y_w.reset_index(drop=True, inplace=True)
df_y_b.reset_index(drop=True, inplace=True)

# df_y_w.to_csv(f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_{iterations}iter_{race}_test_w_y_pred_w.csv', index=False)
# df_y_b.to_csv(f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_{iterations}iter_{race}_test_b_y_pred_b.csv', index=False) 

#%%# Load the best weights from a file
# loaded_weights = np.load('/Users/au605715/Documents/GitHub/jchr_racial_diff/results/best_model_weights.npy', allow_pickle=True)

# # Set the model's weights to the loaded values
# model.set_weights(loaded_weights)



#%%
# f = open('/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_200iter_b_best_weights.txt', "a")
# f.write(str(best_weights_list))
# list_of_dicts = [{"column_name_1": arr} for arr in list_of_arrays]

# df = pd.DataFrame(list_of_dicts)

#%%
# f.close()

# #%%
# with open('/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_200iter_b_best_weights.txt', "r") as file:
#     # Read the content from the file
#     content = file.read()

# # Assuming the content is a string representing a list, you can use eval to convert it back to a Python list
# reloaded_list = eval(content)

# # Now you can use reloaded_list as a Python list
# print(reloaded_list)

#%%from keras.models import load_model

# model.save(f'/Users/au605715/Documents/GitHub/jchr_racial_diff/results/cnn_200iter_b_CNN_.h5')  # creates a HDF5 file 'my_model.h5'
