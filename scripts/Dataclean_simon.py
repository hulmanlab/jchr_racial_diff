# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:51:01 2023

@author: Mikej
"""

import pandas as pd
import numpy as np
import datetime


#%% LOAD DATA 
file_path_replace_bg = r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/REPLACE-BG Dataset-79f6bdc8-3c51-4736-a39f-c4c0f71d45e5 (1)/Data Tables/HDeviceCGM.txt'
data = pd.read_csv(file_path_replace_bg, sep='|')
data = data.drop(['ParentHDeviceUploadsID','SiteID','DexInternalTm','RecordType','DexInternalDtTmDaysFromEnroll'], axis=1)
data = data.rename(columns={'DeviceDtTmDaysFromEnroll': 'DaysFromEnroll','DeviceTm':'time','GlucoseValue':'BG'})
data['BG'] = data.BG * 0.0555

#%% REMOVE RUN-IN PERIOD
# use boolean indexing to select rows where 'A' is negative
negative_rows = data['DaysFromEnroll'] < 1
# drop the selected rows using the 'drop()' method
data = data.loc[~negative_rows]

#%% COMBINING DATE AND TIME INTO ONE COLUMN
data['date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(data['DaysFromEnroll'] - 1, unit='d')

# Extract time from time column and combine with date from date column
data['date'] = data.apply(lambda row: datetime.datetime.combine(row['date'].date(), 
                                                       datetime.datetime.strptime(row['time'], '%H:%M:%S').time()), axis=1)
# Dropping old date related columns 
data = data.drop(['time','DaysFromEnroll'], axis=1)
data = data.sort_values(['PtID','date'])

#%% 
data['PtID'] = data['PtID'].rank(method='dense').astype(int)

# data = data.loc[data['PtID'].isin([1,2])]


#%% GAP DETECTION
 # create empty DataFrame for results
df2 = pd.DataFrame(columns=['PtID', 'date', 'BG', 'RecID'])

# set initial datetime value to compare against
last_datetime = data['date'].iloc[0]
last_patient = data['PtID'].iloc[0]

# create a list of dictionaries to append to df2
rows_to_add = []

# iterate through rows of data
for index, row in data.iterrows():
    # check if current patient ID is different from previous patient ID
    if row['PtID'] != last_patient:
        # set last datetime value to current datetime value
        last_datetime = row['date']
    
    # calculate time difference between last datetime and current datetime
    time_difference = row['date'] - last_datetime
    
    # check if time difference is more than 6 minutes
    if time_difference > datetime.timedelta(minutes=6):
        # calculate number of samples to add to df2
        samples_to_add = int(abs(time_difference.total_seconds() / 300)-1)
        
        # create a list of dictionaries to append to df2
        for i in range(samples_to_add):
            new_datetime = last_datetime + datetime.timedelta(seconds=(i+1)*300)
            new_row = {'PtID': row['PtID'], 'date': new_datetime, 'BG': np.nan, 'RecID': np.nan}
            rows_to_add.append(new_row)
            
    # add current row to rows_to_add
    rows_to_add.append(row.to_dict())
    
    # update last datetime and patient values
    last_datetime = row['date']
    last_patient = row['PtID']

# append rows_to_add to df2
df2 = pd.concat([df2, pd.DataFrame(rows_to_add)])

# reset index of df2
df2 = df2.reset_index(drop=True)

#%%
# df2.to_csv(r'C:\Users\Mike\Desktop\Cleaned_T1D_data.csv', index=False)


