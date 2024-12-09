#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:53:43 2023

@author: Helene

Load CGM data and fill in all the missing Datetime values for each patient with NaN


"""

import pandas as pd
import datetime
import numpy as np
from my_utils import days_time_to_datetime
#%% load data
file_path = r"../data/FDataCGM.txt"
df = pd.read_csv(file_path,sep='|')
# df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataCGM.txt',sep='|')
df.drop(columns=['RecID','SiteID','FileUniqueRecID'], inplace = True)

#%% from days and time to Datetime
# sort values
df= df.sort_values(by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm'])

# get a Datetime in the df
df = days_time_to_datetime(df,'DeviceDaysFromEnroll','DeviceTm')


#%% convert into mmol/L
df['CGM'] = df['Glucose']/18.018
df.drop(['Glucose'], axis=1, inplace=True)


#%% GAP DETECTION
 # create empty DataFrame for results
df_dataclean = pd.DataFrame(columns=['PtID', 'Datetime', 'CGM'])

# set initial datetime value to compare against
last_datetime = df['Datetime'].iloc[0]
last_patient = df['PtID'].iloc[0]

# create a list of dictionaries to append to df_dataclean
rows_to_add = []
#%%
# iterate through rows of df_sorted
for index, row in df.iterrows():
    print(row)

    # check if current patient ID is different from previous patient ID
    if row['PtID'] != last_patient:
        # set last datetime value to current datetime value
        last_datetime = row['Datetime']
    
    # calculate time difference between last datetime and current datetime
    time_difference = row['Datetime'] - last_datetime
    
    # check if time difference is more than 6 minutes
    if time_difference > datetime.timedelta(minutes=16):
        # calculate number of samples to add to df_dataclean
        samples_to_add = int(abs(time_difference.total_seconds() / 900)-1)
        # 900 seconds = 15 minutes
        
        # create a list of dictionaries to append to df_dataclean
        for i in range(samples_to_add):
            new_datetime = last_datetime + datetime.timedelta(seconds=(i+1)*900)
            new_row = {'PtID': row['PtID'], 'Datetime': new_datetime, 'CGM': np.nan,}
            rows_to_add.append(new_row)
            
    # add current row to rows_to_add
    rows_to_add.append(row.to_dict())
    
    # update last datetime and patient values
    last_datetime = row['Datetime']
    last_patient = row['PtID']
#%%
# append rows_to_add to df_dataclean
df_dataclean = pd.concat([df_dataclean, pd.DataFrame(rows_to_add)])

# reset index of df_dataclean
df_dataclean = df_dataclean.reset_index(drop=True)

#%% checking everything is as expected
# numb_total_days and numb_days_df_dataclean should be the same

nan_count = df_dataclean.isna().sum().sum()

#%% export files

df_dataclean.to_csv(r'../results/processed_data/1_1_racial_diff_dataclean.csv', index=False)
