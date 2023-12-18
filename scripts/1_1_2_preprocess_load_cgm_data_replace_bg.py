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
file_path_replace_bg = r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/REPLACE-BG Dataset-79f6bdc8-3c51-4736-a39f-c4c0f71d45e5 (1)/Data Tables/HDeviceCGM.txt'
df_cgm = pd.read_csv(file_path_replace_bg, sep='|')
#%%

columns_to_use = ['PtID', 'DeviceDtTmDaysFromEnroll', 'DeviceTm', 'GlucoseValue']
df = pd.DataFrame(df_cgm[columns_to_use])
#%% from days and time to Datetime

# sort values
df= df.sort_values(by=['PtID', 'DeviceDtTmDaysFromEnroll', 'DeviceTm'])

# get a Datetime in the df
df = days_time_to_datetime(df,'DeviceDtTmDaysFromEnroll','DeviceTm')


#%% convert into mmol/L
df['CGM'] = df['GlucoseValue']/18
df.drop(['GlucoseValue'], axis=1, inplace=True)


#%% GAP DETECTION
# if this is not working, outcomment the GAP detection section, load data and then run this cell and then save data_clean_file
  # create empty DataFrame for results
df_dataclean = pd.DataFrame(columns=['PtID', 'Datetime', 'CGM'])
last_patient = None
last_datetime = None
rows_to_add = []

print("Total rows in DataFrame:", len(df))

try:
    for index, row in df.iterrows():
        print(f"Processing index {index} with PtID {row['PtID']}")

        # Check if current patient ID is different from previous patient ID
        if row['PtID'] != last_patient:
            # Update last datetime value to current datetime value
            last_datetime = row['Datetime']
            
        # This condition will be true even if the patient ID hasn't changed
        if last_datetime is not None:
            # Calculate time difference between last datetime and current datetime
            time_difference = row['Datetime'] - last_datetime

            # Check if time difference is more than 5 minutes
            if time_difference > datetime.timedelta(minutes=5):
                # Calculate number of samples to add to df_dataclean
                samples_to_add = int(abs(time_difference.total_seconds() / 300) - 1)

                # Create a list of dictionaries to append to df_dataclean
                for i in range(samples_to_add):
                    new_datetime = last_datetime + datetime.timedelta(seconds=(i + 1) * 300)
                    new_row = {'PtID': row['PtID'], 'Datetime': new_datetime, 'CGM': np.nan}
                    rows_to_add.append(new_row)

        # Add current row to rows_to_add
        rows_to_add.append(row.to_dict())

        # Update last datetime and patient values
        last_datetime = row['Datetime']
        last_patient = row['PtID']

except Exception as e:
    print(f"An error occurred at index {index}: {e}")



# append rows_to_add to df_dataclean
df_dataclean = pd.concat([df_dataclean, pd.DataFrame(rows_to_add)])

# reset index of df_dataclean
df_dataclean = df_dataclean.reset_index(drop=True)

#%% checking everything is as expected
# numb_total_days and numb_days_df_dataclean should be the same

nan_count = df_dataclean.isna().sum().sum()

# #%% how much data per person?
df_value_counts = pd.DataFrame()
df_value_counts['Datapoints'] = df['PtID'].value_counts()

df_value_counts['Days'] = df_value_counts['Datapoints']/24/4

numb_total_days = df_value_counts['Days'].sum()
numb_days_df_dataclean = (len(df_dataclean)-nan_count)/24/4

df_value_counts['Weeks'] = df_value_counts['Datapoints']/24/4/7

df_value_counts=df_value_counts.reset_index()


#%% export files

# df_dataclean.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_1_2_gender_diff_dataclean_v2.csv', index=False)


#%%
