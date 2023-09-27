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
        If f=1/15min -> 4 entries = 1hour
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
    df_one_temp = df_temp[df_temp[col_patient_id]==df_ptid[0]]
    
    
    temp_rows = []
    df_hist_dict = {}  # Create a dictionary to store DataFrames for each PtID
    for i in range(len(df_ptid)):
        print('Patient: ', str(df_ptid[i]))
    
        
        for j in range(len(df_one_temp) - window_size + 1):
            window = df_temp[col_glucose].iloc[j:j + window_size].tolist()  # Extract a window of data
            temp_rows.append(window)
        
        
        df_one_target_temp = df_one_temp[prediction_horizon :]
        df_one_target_temp.reset_index(drop=True,inplace =True)
        
        
        df_hist = pd.DataFrame(temp_rows, columns=[f'Value_{j+1}' for j in range(window_size)])
        df_hist['Target']=df_one_target_temp[col_glucose]
        df_hist[col_patient_id]=df_ptid[i]
    
        df_hist_dict[i] = df_hist  # Store the DataFrame in the dictionary
    
    temp_list_of_dfs = list(df_hist_dict.values())
    df_final = pd.concat(temp_list_of_dfs, ignore_index=True)
    return df_final

