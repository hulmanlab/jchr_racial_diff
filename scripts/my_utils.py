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

#%% Split data
def get_groupShuflesplit_by_group(df_group1, df_group2, test_size=0.10, random_state=42, show_distribution=False, equal_PtID=True):
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


    Returns
    -------
    TYPE
        Dataframes
        * training data
        * test data
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
    train_distribution = train['Race'].value_counts(normalize=True)
    test_distribution = test_2['Race'].value_counts(normalize=True)
    
    # print("Training Set Race Distribution:")
    # print(train_distribution)
    
    # print("\nTesting Set or Hold Out set Race Distribution:")
    # print(test_distribution)
    # print(len(test_1['PtID'].unique()))
    # print(len(test_2['PtID'].unique()))
    
    # print(len(test['PtID'].unique()))
    
    if show_distribution:
        # test if the distribution is correct and plot
        train_distribution = train['Race'].value_counts(normalize=True)
        test_distribution = test['Race'].value_counts(normalize=True)       
        
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
    return train, test
