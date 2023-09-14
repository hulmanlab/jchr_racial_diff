# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import datetime
import numpy as np
#%% load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataCGM.txt',sep='|')
df.drop(columns=['RecID','SiteID','FileUniqueRecID'], inplace = True)
# df.rename(columns={'DeviceDaysFromEnroll':"DaysFromEnroll"}, inplace=True)
#%% sort rows
df_sorted = df.sort_values(by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm'])

#%% convert my date and time into a datetime 

df_sorted ['temp0'] = pd.to_datetime(df['DeviceTm'], format='%H:%M:%S')

reference_date = pd.Timestamp('1970-01-01')
df_sorted['temp1'] = reference_date + pd.to_timedelta(df['DeviceDaysFromEnroll'], unit='D')
df_sorted ['Date']= df_sorted ['temp1'].dt.date
df_sorted ['Time']= df_sorted ['temp0'].dt.time
# merge date and time
df_sorted['Datetime'] = df_sorted.apply(lambda row: datetime.datetime.combine(row['Date'], row['Time']), axis=1) 

df_sorted.reset_index(drop=True, inplace=True)

df_sorted.drop(['temp0','temp1','DeviceTm', 'Date', 'Time','DeviceDaysFromEnroll'], axis=1, inplace=True)

#%% convert into mmol/L
df_sorted['CGM'] = df_sorted['Glucose']/18
df_sorted.drop(['Glucose'], axis=1, inplace=True)

#%% GAP DETECTION
 # create empty DataFrame for results
df_results = pd.DataFrame(columns=['PtID', 'Datetime', 'CGM'])

# set initial datetime value to compare against
last_datetime = df_sorted['Datetime'].iloc[0]
last_patient = df_sorted['PtID'].iloc[0]

# create a list of dictionaries to append to df_results
rows_to_add = []
#%%
# iterate through rows of df_sorted
for index, row in df_sorted.iterrows():
    print(row)

    # check if current patient ID is different from previous patient ID
    if row['PtID'] != last_patient:
        # set last datetime value to current datetime value
        last_datetime = row['Datetime']
    
    # calculate time difference between last datetime and current datetime
    time_difference = row['Datetime'] - last_datetime
    
    # check if time difference is more than 6 minutes
    if time_difference > datetime.timedelta(minutes=16):
        # calculate number of samples to add to df_results
        samples_to_add = int(abs(time_difference.total_seconds() / 900)-1)
        # 900 seconds = 15 minutes
        
        # create a list of dictionaries to append to df_results
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
# append rows_to_add to df_results
df_results = pd.concat([df_results, pd.DataFrame(rows_to_add)])

# reset index of df_results
df_results = df_results.reset_index(drop=True)

#%% checking everything is as expected
# numb_total_days and numb_days_df_results should be the same

nan_count = df_results.isna().sum().sum()

# how much data per person?
df_value_counts = pd.DataFrame()
df_value_counts['Datapoints'] = df_sorted['PtID'].value_counts()

df_value_counts['Days'] = df_value_counts['Datapoints']/24/4

numb_total_days = df_value_counts['Days'].sum()
numb_days_df_results = (len(df_results)-nan_count)/24/4

#%% export files

df_results.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/racial_diff_dataclean', index=False)
