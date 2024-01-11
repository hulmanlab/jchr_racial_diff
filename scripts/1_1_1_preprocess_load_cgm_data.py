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
file_path = r"/home/hbt/jchr_data/_data/FDataCGM.txt"
df = pd.read_csv(file_path,sep='|')
# df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataCGM.txt',sep='|')
df.drop(columns=['RecID','SiteID','FileUniqueRecID'], inplace = True)

#%% from days and time to Datetime
# sort values
df= df.sort_values(by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm'])

# get a Datetime in the df
df = days_time_to_datetime(df,'DeviceDaysFromEnroll','DeviceTm')


#%% convert into mmol/L
df['CGM'] = df['Glucose']/18
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

#%% how much data per person?
df_value_counts = pd.DataFrame()
df_value_counts['Datapoints'] = df['PtID'].value_counts()

df_value_counts['Days'] = df_value_counts['Datapoints']/24/4

numb_total_days = df_value_counts['Days'].sum()
numb_days_df_dataclean = (len(df_dataclean)-nan_count)/24/4

df_value_counts['Weeks'] = df_value_counts['Datapoints']/24/4/7

df_value_counts=df_value_counts.reset_index()




df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
df_roster= df_roster[df_roster['FPtStatus'] != 'Dropped']
df_roster.drop(columns=['RecID','SiteID','FPtStatus'], inplace = True)
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)


df_counts = pd.merge(df_value_counts, df_roster, on='PtID')
df_counts['Days_round']=round(df_counts.Days)

df_count_w = df_counts [df_counts ['Race'] == 'white']
df_count_b = df_counts [df_counts ['Race'] == 'black']


value_counts = df_counts['Days_round'].value_counts().sort_index()

df_test = pd.DataFrame(value_counts)
import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(16, 6))  # You can adjust the dimensions as needed
value_counts.plot(kind='bar')
plt.xticks(rotation=45)  # Rotates labels to 45 degrees
plt.xlabel('Number of Days (rounded values)')
plt.ylabel('Number of Patients')
# plt.title('Frequency of Each Unique Value in the Column')
plt.show()

#%% export files

# df_dataclean.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/racial_diff_dataclean.csv', index=False)

# df_value_counts.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/racial_diff_numb_days.csv', index=False)


#%%
