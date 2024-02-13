import pandas as pd
import my_utils
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

#%% 


file_path_df = r'/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_2_model_input_ws60min_ph60min.csv'
# file_path_df = r'/Users/au605715/Documents/GitHub/study1/1_2_model_input_ws60min_ph60min.csv'
df = pd.read_csv(file_path_df)

group_name = "Race"

df.drop(columns=["EduLevel", "Gender"], inplace=True)
df.dropna(inplace=True)


#%% split data
xy_train_val, xy_test = my_utils.split_within_PtID(df, numb_values_to_remove=-672, seperate_target=False, group_column=group_name) # split witin  PtID: 4values/hour * 24hour/day * 7days/week = 672 values/week

x_train, y_train, x_val, y_val = my_utils.split_within_PtID_ratio(xy_train_val,percentage_to_remove = 0.10, seperate_target=True, group_column=group_name)

x_test, y_test = my_utils.seperate_the_target(xy_test, group_column=group_name)


#%%

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
 


#%%                     Transform input to the rnn
x_train = pd.DataFrame(x_train)
x_val = pd.DataFrame(x_val)
x_test = pd.DataFrame(x_test)

x_train = my_utils.get_rnn_input(x_train)
x_val = my_utils.get_rnn_input(x_val)
x_test = my_utils.get_rnn_input(x_test)

#%%

# search_space = {
#     "lr": [0.0001, 0.001, 0.01, 0.1, 1],
#     "decay_steps": [1000],
#     "decay_rate": [0.005],
#     "batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#     "epochs":
#     }

#%%

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Modify the model creation function to include optimizer and learning rate
def create_lstm_model(input_length, lstm_units_1=32, lstm_units_2=16, activation="relu" ,dropout_rate=0.1, optimizer='adam', learning_rate = 0.001, decay_steps = 1000, decay_rate = 0.001):
    input_layer = Input(shape=(input_length, 1))
    lstm_layer1 = LSTM(lstm_units_1, activation=activation, return_sequences=True)(input_layer)
    lstm_layer2 = LSTM(lstm_units_2, activation=activation)(lstm_layer1)
    dropout_layer = Dropout(dropout_rate)(lstm_layer2)
    output_layer = Dense(1, activation="relu")(dropout_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    
    # Parameters

    
    # Exponential decay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,  # Convert linear decay rate to exponential form
    )
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay = 0.001)
    
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error','mean_absolute_error'])
    print('goat')

    return model

# Early stopping callback defined outside the model creation function
# This won't be directly used in GridSearchCV but you can use it for final training
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Wrap the model with KerasRegressor
model = KerasRegressor(build_fn=create_lstm_model, input_length=100, verbose=0)

# Define the parameter grid, including optimizer and learning rate
param_grid = {
    'lstm_units_1': [8, 16, 32, 64, 128],
    'lstm_units_2': [8, 16, 32, 64, 128],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.05, 0.1, 0.2, 0.25],
    'learning_rate': [0.001, 0.0001, 0.00001, 0.000001],
    'decay_rate': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    'batch_size': [64, 128, 256, 512, 1024, 2048, 4096]
}

# Set up GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# You will need to fit the grid to your data
# grid.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stop])

# Note: GridSearchCV does not directly support passing validation_data or callbacks.
# For early stopping, you might need to integrate validation split within your training data
# or adjust the model fitting process to include early stopping logic.




#%%


# Perform the grid search as shown in the previous example...
grid_result = grid.fit(x_train, y_train)

# Get the results into a DataFrame
results = pd.DataFrame(grid_result.cv_results_)

# Save to a CSV file for later analysis
results.to_csv('grid_search_results.csv', index=False)

# Print the best parameters and the corresponding score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Get the detailed results for each combination of hyperparameters
print(results[['param_units', 'param_activation', 'param_dropout_rate', 'param_optimizer', 'mean_test_score', 'std_test_score', 'rank_test_score']])




# # y_pred_val = scaler_y(y_pred_val)

# #%% Validation
# y_val= scaler_y.inverse_transform(y_val)
# y_pred_val= scaler_y.inverse_transform(y_pred_val)

# rmse_val = my_utils.calculate_results(y_val, y_pred_val)

# print("val:",rmse_val)


# #%%

# y_pred_test = model_base.predict(x_test)

# y_test= scaler_y.inverse_transform(y_test)
# y_pred_test= scaler_y.inverse_transform(y_pred_test)

# rmse_test = my_utils.calculate_results(y_test, y_pred_test)

# print("test:",rmse_test)

# #%%

# import matplotlib.pyplot as plt

# # Extract loss values
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']

# # Extract epoch numbers
# epochs = range(1, len(train_loss) + 1)

# # Plot training and validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss, 'b-', label='Training Loss')
# plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()
