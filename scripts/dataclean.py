# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import datetime
#%% load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataCGM.txt',sep='|')
df.drop(columns=['RecID','SiteID','FileUniqueRecID'], inplace = True)
# df.rename(columns={'DeviceDaysFromEnroll':"DaysFromEnroll"}, inplace=True)
df_sorted = df.sort_values(by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm'])

#%% Convert my date and time into a datetime

df_sorted ['temp0'] = pd.to_datetime(df['DeviceTm'], format='%H:%M:%S')

reference_date = pd.Timestamp('1970-01-01')
df_sorted['temp1'] = reference_date + pd.to_timedelta(df['DeviceDaysFromEnroll'], unit='D')
df_sorted ['Date']= df_sorted ['temp1'].dt.date
df_sorted ['Time']= df_sorted ['temp0'].dt.time
# merge date and time
# df_one['Datetime'] = df_one.apply(lambda row: datetime.datetime.combine(row['Date'], row['Time']), axis=1) 

df_sorted.drop(['temp0','temp1','DeviceTm'], axis=1, inplace=True)
df_sorted.reset_index(drop=True, inplace=True)

#%% convert into mmol/L
df_sorted['CGM'] = df_sorted['Glucose']/18
df_sorted.drop(['Glucose','DeviceDaysFromEnroll'], axis=1, inplace=True)

#%% how much data per person?
df_value_counts = pd.DataFrame()
df_value_counts['Datapoints'] = df_sorted['PtID'].value_counts()

df_value_counts['Days'] = df_value_counts['Datapoints']/24/4


#%% patient extraction
df_unique = df['PtID'].unique()
df_one = df[df['PtID']==df_unique[0]] #179


 
#%%
df_datatypes = df.dtypes # check for datatypes in df columns
















