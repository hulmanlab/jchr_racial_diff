#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:48:02 2023
@author: au605715

contains all the functions made by me, Helene Thomsen
"""

#%% Preprocessing
def days_time_to_datetime(df, col_days_from_enroll, col_time,):
    """
    Converts days from enrollment and time into datetime
    Starts date is the 1970-01-01
    
    Parameters
    ----------
        df (pd.dataframe):
            Contains patient IDs, CGM data and time
        days_from_enroll (str):
            The name of the column .
        time (str):
            Name of the column containing time '%H:%M:%S'

    Returns  
    -------
        df['Datetime'] in the format

    """
    import datetime
    import pandas as pd
    df_temp = df.copy()
    df_temp['temp0'] = pd.to_datetime(df[col_time], format='%H:%M:%S')

    reference_date = pd.Timestamp('1970-01-01')
    df_temp['temp1'] = reference_date + pd.to_timedelta(df[col_days_from_enroll], unit='D')
    
    df_temp['Date']= df_temp['temp1'].dt.date
    df_temp['Time']= df_temp['temp0'].dt.time
    # merge date and time
    df_temp['Datetime'] = df_temp.apply(lambda row: datetime.datetime.combine(row['Date'], row['Time']), axis=1) 

    df_temp.reset_index(drop=True, inplace=True)

    df_temp.drop(['temp0','temp1','DeviceTm', 'Date', 'Time',col_days_from_enroll], axis=1, inplace=True)
    return df_temp


#%% feature extraction
def feature_extraction_cnn(df, window_size, prediction_horizon, col_patient_id ='PtID', col_glucose='CGM'):

    """
    Extracts features from a DataFrame containing glucose data for multiple patients.
    Works as a sliding window. Assuming no data is missing in data. Does not care about NaN in dataset

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing glucose data for multiple patients.
    window_size : int
        The size of the sliding window used for feature extraction in number of entries
        If f=1/15min -> 4 entries = 1hour
    prediction_horizon : int
        The size of the prediction horizon used for Target extraction in number of entries
        If f=1/15min 4 entries = 1hour and window_size = 4, then 4+4 =8, it is the 9th, just remem
    col_patient_id : str, optional
        The name of the column in the DataFrame that represents patient IDs. The default is 'PtID'.
    col_glucose : str, optional
        The name of the column in the DataFrame that contains glucose values. The default is 'CGM'.


    Returns
    -------
    df_final : pandas.DataFrame
        A DataFrame containing extracted features, with columns representing glucose values within sliding windows,
        associated patient IDs, and the target variable.

    Notes
    -----
    This function extracts features from glucose data by creating sliding windows of specified size for each patient.
    It then constructs a new DataFrame with features derived from these sliding windows.

    Example
    -------
    >>> import pandas as pd
    >>> data = {'PtID': [1, 1, 1, 2, 2, 2],
    ...         'CGM': [9.4, 9.3, 9.2, 9.1, 9, 8.7, 10.0]}
    >>> df = pd.DataFrame(data)
    >>> extracted_features = feature_extraction_cnn(df, window_size, prediction_horizon)
    >>> print(extracted_features)
       Value_1  Value_2  Value_3  Value_4  Target  PtID
    0      9.4      9.3      9.2       9     8.4    1
    1      9.3      9.2       9       9.1    8.1    1
    2      9.2       9       9.1      8.7   10.0    1
    3       9       9.1      8.7      8.6    NaN    1
    4      9.1      8.7      8.6      8.4    NaN    1
    5      NaN      NaN      5.6      6.2    5.6    2
    """
    import pandas as pd
    df_temp = df.copy()
    df_ptid = df_temp[col_patient_id].unique()
    
    df_hist_dict = {}  # Create a dictionary to store DataFrames for each PtID
    
    for i in range(len(df_ptid)):
        print('Patient: ', str(df_ptid[i]))
        
        df_one_temp = df_temp[df_temp[col_patient_id] == df_ptid[i]]
        temp_rows = []  # Reset temp_rows for each patient
        
        for j in range(len(df_one_temp) - window_size + 1):
            window = df_one_temp[col_glucose].iloc[j:j + window_size].tolist()  # Extract a window of data
            temp_rows.append(window)
        
        df_one_target_temp = df_one_temp[prediction_horizon:]
        df_one_target_temp.reset_index(drop=True, inplace=True)
        
        df_hist = pd.DataFrame(temp_rows, columns=[f'Value_{j+1}' for j in range(window_size)])
        df_hist['Target'] = df_one_target_temp[col_glucose]
        df_hist[col_patient_id] = df_ptid[i]
    
        df_hist_dict[i] = df_hist  # Store the DataFrame in the dictionary
    
    temp_list_of_dfs = [df.reset_index(drop=True) for df in df_hist_dict.values()]
    df_final = pd.concat(temp_list_of_dfs, ignore_index=True)
    return df_final


#%% 3.preprocess_splitting_PtID
def get_group_id(df,id_column, group_column, group1, group2):
    """
    get 2 lists with the unique PtIDs from you dataframe

    Parameters
    ----------
    df : data
    group_column : String

    group1 : String
        
    group2 : String


    Returns
    -------
    gr1_unique_id : int64array
    
    gr2_unique_id : int64array

    """

    df_gr1  = df[df[group_column] == group1]
    df_gr2 = df[df[group_column] == group2]
    
    gr1_unique_id  = df_gr1[id_column].unique()
    gr2_unique_id = df_gr2[id_column].unique()
    
    return gr1_unique_id, gr2_unique_id


def combine_arrays_w_percentage(array1, array2, percentage):
    """
    Takes the specified percentage of array1
    and the left over percentages from array2 randomly
    and combines them
    

    Parameters
    ----------
    array1 : Array of int64
        
    array2 : Array of int64
        
    percentage : integer
        0-100
  
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    import numpy as np

    # Calculate the number of elements to take from each array
    num_from_array1 = int(len(array1) * (percentage / 100))
    num_from_array2 = int(len(array2) * ((100 - percentage) / 100))

    # Randomly select elements from each array
    selected_from_array1 = np.random.choice(array1, num_from_array1, replace=False)
    selected_from_array2 = np.random.choice(array2, num_from_array2, replace=False)

    # Combine and return the result
    return np.concatenate((selected_from_array1, selected_from_array2))


#%% Split data

def get_groupShuflesplit_equal_groups(df_group1, df_group2, test_size=0.10, random_state=None, show_distribution=False, equal_PtID=True, seperate_target=True, group_column = 'Race'):
    """
    Description
    ----------
    Takes test_size percentage from each group and puts it in a testset
    Distribute on PtID are even can be done with setting equal_PtID = True
    
    Parameters
    ----------
    df_group1 : dataframe
        dataframe frome one group
    df_group2 : dataframe
        dataframe from the other group
    test_size : float, optional
        defines percentage of split that goes into test The default is 0.10.
    random_state : float, optional
        ensure the same split every time. The default is 42.
    show_distribution : TRUE/FALSE, optional
        Creates bar plot. The default is False.
    equal_PtID : TRUE/FALSE, optional
        makes equal distribution of each patientgroup in test set . The default is True.
    seperate_target: TRUE/FALSE, optional
        changes number of output variables. Divides train, test in x_train, y_train, x_test, y_test.
        The default is True


    Returns
    -------
    TYPE
        Dataframes
        * x_train = training
        * y_train = target to x_train
        * x_test = testing
        * y_test = target to x_test
        * test
    If show_distribution = TRUE you also get their distributions in datatype Series
    

    """
    from sklearn.model_selection import GroupShuffleSplit
    import pandas as pd
    import matplotlib.pyplot as plt
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_idx_1, test_idx_1 = next(gss.split(df_group1, groups=df_group1['PtID']))
    train_idx_2, test_idx_2 = next(gss.split(df_group2, groups=df_group2['PtID']))
    
    # Create train and test sets for white
    train_1 = df_group1.iloc[train_idx_1]
    test_1 = df_group1.iloc[test_idx_1]
    
    # Create train and test sets for black
    train_2 = df_group2.iloc[train_idx_2]
    test_2 = df_group2.iloc[test_idx_2]
    
    if equal_PtID:
        # Testing the length
        df_test_group1_unique = test_1['PtID'].unique()
        df_test_group2_unique = test_2['PtID'].unique()
        
        
        row_difference = len(df_test_group1_unique) - len(df_test_group2_unique)
        
        if row_difference > 0:
            # create equal number of PtIDs in df_test_group1_unique and df_test_group2_unique
            # do this by removing the PtIds that are too many from df_test_group1_unique
            removed_patient_ids_group1 = df_test_group1_unique[:row_difference]
            test_1 = test_1[~test_1['PtID'].isin(removed_patient_ids_group1)]
        elif row_difference < 0:
            # create equal number of PtIDs in df_test_group1_unique and df_test_group2_unique
            # do this by removing the PtIds that are too many from df_test_group2_unique
            removed_patient_ids_group2 = df_test_group2_unique[:abs(row_difference)]
            test_2 = test_2[~test_2['PtID'].isin(removed_patient_ids_group2)]
        else:
            pass
        
    # put together the train and test sets
    train = pd.concat([train_1, train_2])
    test = pd.concat([test_1, test_2])
    
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    # test if the distribution is correct and plot
    train_distribution = train[group_column].value_counts(normalize=True)
    test_distribution = test_2[group_column].value_counts(normalize=True)
    

    if show_distribution:
        # test if the distribution is correct and plot
        train_distribution = train[group_column].value_counts(normalize=True)
        test_distribution = test[group_column].value_counts(normalize=True)       
        
        # plot distribution
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # Add title for the entire plot
        fig.suptitle('Race Distribution')
        # for the titel of the subplots
        test_size = test_size*100
        train_size = 100-test_size
        
        # Plot training set race distribution
        bars_train = axs[0].bar(train_distribution.index, train_distribution, color=['blue', 'orange'])
        axs[0].set_title(f'{train_size}% Training & Finetuning Set') # subtitel

        # Plot testing set race distribution
        bars_test = axs[1].bar(test_distribution.index, test_distribution, color=['blue', 'orange'])
        
        axs[1].set_title(f'{test_size}% Testing Set') # subtitel

        # Rename the labels under the bars
        for ax in axs:
            ax.set_xticklabels(['nonHB', 'nonHW'], rotation=0)

        # Display exact values on top of the bars
        def add_values(bars, ax, labels):
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                percentage = round(yval * 100, 2)  # Convert to percentage and round to 2 decimal places, text on top of bars
                ax.text(bar.get_x() + bar.get_width()/2, yval, f'{percentage}%', ha='center', va='bottom')
                
                # Add two lines of text under each column
                ax.text(bar.get_x() + bar.get_width()/2, -0.1, labels[i][0], ha='center', va='bottom')
                ax.text(bar.get_x() + bar.get_width()/2, -0.15, labels[i][1], ha='center', va='bottom')

                
        # text under bars
        df_train_group1_unique = train_1['PtID'].unique()
        df_train_group2_unique = train_2['PtID'].unique()
        # Testing new lengththe length
        df_test_group1_unique = test_1['PtID'].unique()
        df_test_group2_unique = test_2['PtID'].unique()

        column_labels_train = [[(f'Patients:{len(df_train_group2_unique)}'), (f'Sample Size: {len(train_2)}')], [(f'Patients:{len(df_train_group1_unique)}'),  (f'Sample Size: {len(train_1)}')]]
        column_labels_test = [[(f'Patients:{len(df_test_group2_unique)}'),  (f'Sample Size: {len(test_2)}')], [(f'Patients:{len(df_test_group1_unique)}'),  (f'Sample Size: {len(test_1)}')]]


        # column_labels_train = [['Label1A', 'Label1B'], ['Label2A', 'Label2B']]
        # column_labels_test = [['Label3A', 'Label3B'], ['Label4A', 'Label4B']]

        add_values(bars_train, axs[0], column_labels_train)
        add_values(bars_test, axs[1], column_labels_test)

        # Show the plot
        plt.show()
        # return train, test, train_distribution, test_distribution
    if seperate_target:
        x_train = train.drop(['Target', 'PtID', group_column], axis=1)
        y_train = train['Target']
        x_test = test.drop(['Target', 'PtID', group_column], axis=1)
        y_test = test['Target']
    
        return  x_train, y_train, x_test, y_test
    return train, test

def split_within_PtID(df, numb_values_to_remove ,PtID_column = "PtID", seperate_target = True, group_column = "Race"):
    """
    Remove bottom values for each PtID

    Parameters
    ----------
    df : DataFrame

    numb_values_to_remove : int
        DESCRIPTION. number of rows to be removed
    PtID_column : String, optional
        DESCRIPTION. The default is "PtID".

    Returns
    -------
    x_train : DataFrame
        DESCRIPTION: first values woithout removed rows
        
    x_val : DataFrame
        DESCRIPTION: all the bottom rows that were removed

    """
    import pandas as pd

    
    # Group the data by PtId
    grouped = df.groupby('PtID')
    
    x_val = pd.DataFrame()
    x_train = pd.DataFrame()
    
    # Move the last 672 rows for each PtId to the new DataFrame
    for pt_id, group in grouped:
        last_672_rows = group.iloc[numb_values_to_remove:]
        first_values = group.iloc[:numb_values_to_remove]  # Remove the last 672 rows
        x_val = pd.concat([x_val, last_672_rows])
        x_train = pd.concat([x_train, first_values])
    
    if seperate_target:
        x_train2 = x_train.drop(['Target', 'PtID', group_column], axis=1)
        y_train = x_train['Target']
        x_test = x_val.drop(['Target', 'PtID', group_column], axis=1)
        y_test = x_val['Target']
    
        return  x_train2, y_train, x_test, y_test
    
    
    return  x_train, x_val

def seperate_the_target(x_train, x_val=None, group_column = "Race"):
    """
    Split training/validatio and label variable

    Parameters
    ----------
    x_train : DataFrame
        DESCRIPTION.
    x_val : DataFrame
        DESCRIPTION.

    Returns
    -------
    if x_val is provided:
        x_train2 : DataFrame
            DESCRIPTION.
        y_train : DataFrame
            DESCRIPTION.
        x_test : DataFrame
            DESCRIPTION.
        y_test : DataFrame
            DESCRIPTION.
    if x_val is NOT provided:
        x_train2 : DataFrame
            DESCRIPTION.
        y_train : DataFrame
            DESCRIPTION.
    """
  # Process training data
    x_train2 = x_train.drop(['Target', 'PtID', group_column], axis=1)
    y_train = x_train['Target']

    # Check if validation data is provided
    if x_val is not None:
        x_test = x_val.drop(['Target', 'PtID', group_column], axis=1)
        y_test = x_val['Target']
        return x_train2, y_train, x_test, y_test
    else:
        return x_train2, y_train

def change_trainingset(train, race, test_size, random_state=None, equal_PtID = True, show_distribution=False):
    """
    Description
    -----------
    split input in train and validation
    depending on race-based-input, different parts of input comes out

    Parameters
    ----------
    train : DataFrame
        DESCRIPTION.
    race : Boolean
        valid_inputs = ["w", "b", "wb"]
    test_size : Float
        can also be validation_size, the rest automatically turns into training
    random_state : keep same split?, optional
        default None, else a number eg. 42
    show_distribution : TYPE, optional
        Only relevant for wb
        Shows the difference between the two groups b and w
    equal_PtID : TYPE, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        Input not understood, see 'race' under 'parameters'

    Returns
    -------
    x_train DataFrame
    y_train DataFrame
    x_val   DataFrame
    y_val   DataFrame

    """
    from sklearn.model_selection import GroupShuffleSplit
    valid_inputs = ["w", "b", "wb"]
    if race == "w":                                                             # split white people dataset for training
        train_w = train[train['Race'] == 'white'].drop(labels = 'Race', axis = 1)

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(train_w, groups=train_w['PtID']))

        # Create train and test sets
        x_train_w, y_train_w = train_w.drop(['Target','PtID'], axis=1).iloc[train_idx], train_w['Target'].iloc[train_idx]
        x_val_w, y_val_w = train_w.drop(['Target','PtID'], axis=1).iloc[test_idx], train_w['Target'].iloc[test_idx]
        
        return x_train_w, y_train_w, x_val_w, y_val_w
    
    elif race == "b":                                                            # split black people dataset for training
        train_b = train[train['Race'] == 'black'].drop(labels = 'Race', axis = 1)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(train_b, groups=train_b['PtID']))

        # # Create train and test sets
        x_train_b, y_train_b = train_b.drop(['Target','PtID'], axis=1).iloc[train_idx], train_b['Target'].iloc[train_idx]
        x_val_b, y_val_b = train_b.drop(['Target','PtID'], axis=1).iloc[test_idx], train_b['Target'].iloc[test_idx]
        
        return x_train_b, y_train_b, x_val_b, y_val_b
    
    elif race == "wb":
        train_b = train[train['Race'] == 'black']
        train_w = train[train['Race'] == 'white']

        x_train_wb,y_train_wb, x_val_wb, y_val_wb  = get_groupShuflesplit_equal_groups(train_w, train_b, test_size=test_size, random_state=random_state, show_distribution=show_distribution, equal_PtID=equal_PtID)
        return x_train_wb,y_train_wb, x_val_wb, y_val_wb
    else:
        raise ValueError("Input not understood. Please provide one of the following options: " + ", ".join(valid_inputs))
        

def split_time_series_data(df, test_size, seperate_target=True, group_column ="Race"):
    """
    df: dataframe
    test_size: decimal % you want your test_size to be
    seperate_target: should target be split?
    Splits a DataFrame into training and validation sets without shuffling, preserving time order.
    
    :param df: Pandas DataFrame containing the time series data.
    :param train_size: Proportion of the dataset to include in the train split (between 0 and 1).
    :return: four datasets if seperate_target is default(True), the training set and the validation set and thei targets
    """

    train_size = 1-test_size
    total_rows = len(df)
    split_index = int(total_rows * train_size)  # 85% for training
    
    # Split the dataset
    train_df = df.iloc[:split_index]
    validation_df = df.iloc[split_index:]
    
    
    if seperate_target:
        x_train, y_train, x_test, y_test= seperate_the_target(train_df,validation_df, group_column = group_column)
    
        return  x_train, y_train, x_test, y_test
    
    return train_df, validation_df




#reshape input
def get_cnn1d_input (x_data):
    """
    Descriptions
    ------------
    reshapes time series input into a shape that cnn1D cant take
    Parameters
    ----------
    x_data : DataFrames
        training data

    Returns
    -------
    x_train_reshape : array of floats
        N x l_window x 1

    """
    x_train_reshape = x_data.values.reshape((x_data.shape[0], x_data.shape[1], 1))
    return x_train_reshape

def get_rnn_input (x_data):
    """
    Descriptions
    ------------
    reshapes time series input into a shape that cnn1D cant take
    Parameters
    ----------
    x_data : DataFrames
        training data

    Returns
    -------
    x_train_reshape : array of floats
        N x l_window x 1

    """
    x_train_reshape = x_data.values.reshape(-1, x_data.shape[1], 1)
    return x_train_reshape

def create_cnn1(my_input_shape):
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
    import tensorflow as tf
    from tensorflow import keras
    
    model = tf.keras.Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(my_input_shape,1)),
        BatchNormalization(),
        MaxPooling1D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    return model





def create_basic_rnn_vanDoorn(input_length):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, SimpleRNN, Bidirectional, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    # input layer
    input_layer = Input(batch_shape=(None, input_length, 1))

    # first layer with 32 neurons
    layer1 = Bidirectional(SimpleRNN(32, activation='relu'), input_shape=(input_length, 1))(input_layer)

    layer2 = Dense(16)(layer1)

    # Dropout
    layer3 = Dropout(0.1)(layer2)

    # pred horizons
    output_layer = Dense(1, name='output')(layer3)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model

    return model

def create_lstm_vanDoorn(input_length):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

    # input layer
    input_layer = Input(shape=(input_length, 1))

    # first LSTM layer with 32 neurons and returning sequences
    lstm_layer1 = LSTM(32, return_sequences=True)(input_layer)

    # second LSTM layer with 16 neurons
    lstm_layer2 = LSTM(16)(lstm_layer1)

    # Dropout
    dropout_layer = Dropout(0.05)(lstm_layer2)

    # output layer
    output_layer = Dense(1, name='output')(dropout_layer)
    
    # creating the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def create_rnn(my_input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense
    from tensorflow import keras
    model = Sequential([
        SimpleRNN(32, input_shape=(4, 1), activation='tanh'),
        # BatchNormalization(),
        Dense(1)
    ])
    # from tensorflow.keras.layers import LSTM, Dense
    # model = Sequential([
    #     LSTM(32, input_shape=(4, 1), activation='tanh'),
    #     Dense(1)
    # ])
    
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



#%% result calculations

def calculate_rmse(actual, predicted):
    """
    Calculate the Mean Squared Error between two numpy arrays.

    Parameters:
    actual (numpy.array): The array of actual values.
    predicted (numpy.array): The array of predicted values.

    Returns:
    float: The Mean Squared Error between array1 and array2.
    """
    import numpy as np
    # Ensuring both arrays are of the same length
    if len(actual) != len(predicted):
        raise ValueError("RMSE: Arrays must be of the same length")

    # Calculating the MSE
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mse(actual, predicted):
    """
    Calculate the Mean Squared Error between two numpy arrays.

    Parameters:
    actual (numpy.array): The array of actual values.
    predicted (numpy.array): The array of predicted values.

    Returns:
    float: The Mean Squared Error between array1 and array2.
    """
    import numpy as np
    # Ensuring both arrays are of the same length
    if len(actual) != len(predicted):
        raise ValueError("RMSE: Arrays must be of the same length")

    # Calculating the MSE
    mse = np.mean((actual - predicted) ** 2)

    return mse


def calculate_mae(predicted, actual):
    """
    Calculate the Mean Absolute Error between two numpy arrays.

    Parameters:
    actual (numpy.array): The array of actual values.
    predicted (numpy.array): The array of predicted values.

    Returns:
    float: The Mean Absolute Error between array1 and array2.
    """
    import numpy as np
    # Ensuring both arrays are of the same length
    if len(predicted) != len(actual):
        raise ValueError("mae: Arrays must be of the same length")

    # Calculating the MAE
    mae = np.mean(np.abs(predicted - actual))
    return mae

def calculate_mard(predicted, actual):
    """
    Calculate the Absolute Relative Difference between two numpy arrays.

    Parameters:
    actual (numpy.array): The array of actual values.
    predicted (numpy.array): The array of predicted values.

    Returns:
    numpy.array: An array of the Absolute Relative Differences for each element.
    """
    import numpy as np
    # Ensuring both arrays are of the same length
    if len(predicted) != len(actual):
        raise ValueError("mard: Arrays must be of the same length")
        # Flatten 'actual' if it's not already 1-dimensional
    if actual.ndim > 1:
        actual = actual.flatten()
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ard = np.abs((predicted - actual) / actual)
        # print("Shape of 'ard':", ard.shape)
        # print("Shape of 'actual':", actual.shape)
        ard[actual == 0] = 0  # Set ARD to 0 where the first array is 0
        mard = np.mean(ard)

    return mard


def calculate_r_squared(actual, predicted):
    """
    Calculate the coefficient of determination (R^2) for actual and predicted values.

    Parameters:
    actual (numpy.array): The array of actual values.
    predicted (numpy.array): The array of predicted values.

    Returns:
    float: The R^2 value.
    """
    import numpy as np

    # Ensure both arrays have the same length
    if len(actual) != len(predicted):
        raise ValueError("R2: Both arrays must be of the same length")


    # Sum of Squares of Residuals (SSR)
    ssr = np.sum((actual - predicted) ** 2)
    
    # Total Sum of Squares (SST)
    mean_actual = np.mean(actual)
    sst = np.sum((actual - mean_actual) ** 2)


    # Coefficient of Determination (R^2)
    r_squared = 1 - (ssr / sst)

    return r_squared


def calculate_results (actual,predicted):
    rmse = calculate_rmse(actual, predicted)
    # mae = calculate_mae(predicted, actual)
    # mard = calculate_mard(predicted, actual)
    # r2 = calculate_r_squared(actual, predicted)
    return rmse