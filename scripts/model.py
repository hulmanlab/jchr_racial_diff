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
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from my_utils import get_groupShuflesplit_by_group
from sklearn.model_selection import GroupShuffleSplit
import logging
#%%                     load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/cnn_ws60min_ph60min.csv')
df.dropna(inplace=True)

#%%                     splitting patients in ethnicity
df_black = df[df['Race'] == 'black']
df_white = df[df['Race'] == 'white']

#%%                     numb of patients
df_unique = df['PtID'].unique()
df_unique_white = df_white['PtID'].unique()
df_unique_black = df_black['PtID'].unique()

#%%                     divide data
def get_X_y_groups_race(data):
    FEATURES = [
        "Value_1",
        "Value_2",
        "Value_3",
        "Value_4",
        "Target"
        ]
    
    GROUPS = "PtID"
    
    TARGET = "Target"
    
    RACE = "Race"
    
    X = data[FEATURES]
    y = data[TARGET]
    race = data[RACE]
    groups = data[GROUPS]
    
    return X, y, groups, race
df_X, df_y, df_groups, df_race = get_X_y_groups_race(df)

#%%                     split data

#split train-validate set and test set (hold-out set)
train, test = get_groupShuflesplit_by_group(df_white, df_black, test_size=0.10, random_state=42, show_distribution=True, equal_PtID=True)
x_test = test.drop(['Target','Race'], axis=1)

# FINE-TUNED MODEL split white people dataset for training
train_w = train[train['Race'] == 'white'].drop(labels = 'Race', axis = 1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(train_w, groups=train_w['PtID']))

# Create train and test sets
x_train, y_train = train_w.drop(['Target','PtID'], axis=1).iloc[train_idx], train_w['Target'].iloc[train_idx]
x_val, y_val = train_w.drop(['Target','PtID'], axis=1).iloc[test_idx], train_w['Target'].iloc[test_idx]


#%%                     Define the model architecture
def create_cnn(my_input_shape):
    model = tf.keras.Sequential([
        Conv1D(32, 3, activation='relu', input_shape=my_input_shape),
        MaxPooling1D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    return model


# Define early stopping callback¢
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#%%                     reshape input for model

#reshape input
x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_val = x_val.values.reshape((x_val.shape[0], x_val.shape[1], 1))

#%%                     model and evaluation
model = create_cnn((x_train.shape[1],1))

history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stop])


# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# if I want to log the training and validation loss
# logging.info("Training Loss:", history.history['loss'])
# logging.info("Validation Loss:", history.history['val_loss'])

x_test = test.drop(['Target', 'PtID', 'Race'], axis=1).values.reshape((test.shape[0], x_train.shape[1], 1))
y_test = test['Target']

test_loss, test_mae = model.evaluate(x_val, y_val)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")



#%%

for i in range(2):
    print("Iteration:", i + 1)


#%%                     reshaping feature matrix adding a dimension 


# # Evaluate the model on the test data
# test_mse, test_mae = model.evaluate(x_val, y_val)
# print('mean squarred error', test_mse)
# print('mean absolute error', test_mae)

# #%% simple baseline
# x_test, y_test = df.drop(['Target','PtID'], axis=1).iloc[test_idx], df['Target'].iloc[test_idx]
# mae = mean_absolute_error(x_test['Value_4'], y_test)
# mse = mean_squared_error(x_test['Value_4'], y_test)
# print('mean squarred error', mse)
# print('mean absolute error', mae)




# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()






 