#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:43:59 2023

@author: au605715
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_absolute_error, mean_squared_error

#%%
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/racial_diff_assemble.csv')

df.dropna(inplace=True)


#%% 
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['PtID']))

# Create train and test sets
x_train, y_train = df.drop(['Target','PtID'], axis=1).iloc[train_idx], df['Target'].iloc[train_idx]
x_test, y_test = df.drop(['Target','PtID'], axis=1).iloc[test_idx], df['Target'].iloc[test_idx]

#%% reshaping feature matrix adding a dimension 

# Convert DataFrames to numpy arrays and reshape
x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))


#%% Calculate class weights based on counts in y dataframe
# counter = Counter(y_train)
# max_count = max(counter.values())
# class_weights = {cls: max_count / count for cls, count in counter.items()}

#%% Define the model architecture
model = tf.keras.Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(x_train.shape[1],1)),
    BatchNormalization(),
    MaxPooling1D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# # Define early stopping callback¢
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with class weights and early stopping
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test), callbacks=[early_stop])

# Evaluate the model on the test data
test_mse, test_mae = model.evaluate(x_test, y_test)
print('mean squarred error', test_mse)
print('mean absolute error', test_mae)

#%% simple baseline
x_test, y_test = df.drop(['Target','PtID'], axis=1).iloc[test_idx], df['Target'].iloc[test_idx]
mae = mean_absolute_error(x_test['Value_4'], y_test)
mse = mean_squared_error(x_test['Value_4'], y_test)
print('mean squarred error', mse)
print('mean absolute error', mae)