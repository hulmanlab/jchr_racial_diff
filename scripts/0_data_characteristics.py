# -*- coding: utf-8 -*-
"""
- Load all data from the study.
- groups patients and counts them
- plots data

"""
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from my_utils import days_time_to_datetime
#%% load data
# excluded patients of the study
df_final_status = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtFinalStatus.txt', sep='|')
df_final_status.drop(columns=['RecID','SiteID'], inplace = True)

# ethnicity
df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
df_roster.drop(columns=['RecID','SiteID'], inplace = True)
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)
df_roster= df_roster[df_roster['FPtStatus'] != 'Dropped']
# baseline and first visit
df_baseline = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FBaseline.txt', sep='|')
df_baseline.drop(columns=['RecID','SiteID','EligChecklist','DomHand', 'EduLevel','EduLevelUnk','AnnualInc',
                          'AnnualIncDNA','AnnualIncUnk', 'InsDelPump', 'ConfStudyBGMCellPhMatch',
                          'ConfStudyBGMSyncPump','ConfPtRemNoAdjTime','NoExam','WeightNotDone',
                          'HeightNotDone','SensorPlaced','SensorLoc'], inplace = True)

# cgm
df_cgm = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataCGM.txt',sep='|')
df_cgm['CGM'] = df_cgm['Glucose']/18
df_cgm.drop(columns=['RecID','SiteID','FileUniqueRecID','Glucose'], inplace = True)
df_cgm = df_cgm.sort_values(by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm'])

df_unique = df_roster['PtID'].unique()


df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//1_2_cnn_ws60min_ph60min.csv')
df.dropna(inplace=True)

df2 = df[df['PtID'].isin(df_unique)]
df3 = df2[df2['Race'] != 'black']
#%% load data
# adverse reaction to CGM
df_advEvent = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FAdvEvent.txt', sep='|')

# acceleromenter data (179 patients)
df_acc = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataAccel.txt', sep='|')
df_acc_info = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataAccelSubjectInfo.txt', sep='|')
df_acc_info.drop(columns=['RecID'], inplace = True)
df_acc_info.drop_duplicates(keep='first', inplace=True)

# fingerprick
df_bgm = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDataBGM.txt', sep='|')
df_bgm['BGM'] = df_bgm['Glucose']/18
df_bgm.drop(columns=['Unnamed: 0','SiteID','Glucose','DeviceStorUnits'], inplace = True)

#%% loa data visits
# overview of all the visits and if patients showed up
df_visit_info = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FVisitInfo.txt', sep='|')

# can only be used for HbA1C
df_final_visit = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FFinalVisit.txt', sep=',')
df_final_visit.drop(columns=['RecID','SiteID'], inplace = True)

# only CGM device placement info (not relevant)
df_follow_up = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FFollowUpVisit.txt', sep=',')
df_follow_up.drop(columns=['RecID','SiteID'], inplace = True)

# all the results from the blood tests at the visits
df_sampleResults = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FSampleResults.txt', sep='|')

#%% load data (not relevant)
# # info of type of insulin each patient takes
# df_insulin = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FInsulin.txt', sep='|')
# df_insulin.drop(columns=['RecID','SiteID'], inplace = True)
# # hemoglobin and hematokrit values, not important
# df_lab_data = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FLabData.txt', sep='|')
# # other medical conditions beside diabetes
df_medical_conditions = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FMedicalConditions.txt', sep='|')
# # other medicine besides insulin
# df_medicine = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FMedications.txt', sep='|')
# patients the study has excluded
df_final_status = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtFinalStatus.txt', sep='|')
# # meal data
# df_daily_log = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FDailyLog.txt', sep='|')

#%% count groups

count_sex_female = (df_baseline['Gender'] == "F").sum()
count_sex_male = (df_baseline['Gender'] == "M").sum()
count_race_black = (df_roster['Race'] == "black").sum()
count_race_white = (df_roster['Race'] == "white").sum()

# I cannot find age anywhere (I have age for all accelerometer patients goes from 5-72yrs but not all cgm included has acc data)
# count_adults = (df_baseline['Age'] > 17).sum()

#%% test: does roster FPtStatus and final_status match? (yes they do)
# they both have information about excluded patients

df_ptid_included= df_roster[~df_roster['FPtStatus'].str.contains('Dropped')].reset_index(drop=True)
df_ptid_included = df_ptid_included.sort_values(by=['PtID'])


df_ptid_excluded = pd.DataFrame()
df_ptid_excluded = df_roster[~df_roster['FPtStatus'].str.contains('Completed')].reset_index(drop=True)
df_ptid_excluded = df_ptid_excluded.sort_values(by=['PtID'])
df_ptid_excluded = df_ptid_excluded.reset_index(drop=True)


df_final_status = df_final_status.sort_values(by=['PtID'])
df_final_status = df_final_status.reset_index(drop=True)


compare_temp = df_ptid_excluded ['PtID'] == df_final_status['PtID']

#%% How many do I need an age on?

df_cgm_unique = df_cgm['PtID'].unique()
df_included_unique = df_ptid_included['PtID'].unique()
df_acc_info_unique = df_acc_info['PtID'].unique()


df_cgm_unique = np.sort(df_cgm_unique)
df_acc_info_unique = np.sort(df_acc_info_unique )

compare_temp = [value in df_acc_info_unique for value in df_cgm_unique]
count_missing_age = compare_temp.count(False)

#%% how much data per person?
df_cgm = df_cgm[~df_cgm['PtID'].isin(df_ptid_excluded['PtID'])] # removes patients dropped from study
df_cgm_unique = df_cgm['PtID'].unique()

df_results = pd.DataFrame()
df_results['Datapoints'] = df_cgm['PtID'].value_counts()

df_results['Days'] = df_results['Datapoints']/24/4

# df_final_status.set_index('PtID', inplace = True)
# df_roster.set_index('PtID', inplace = True)

# df_exclude = df_value_counts.merge(df_roster,left_index=True, right_index=True)

numb_total_days = df_results['Days'].sum()
# numb_days_df_dataclean = (len(df_dataclean)-nan_count)/24/4

df_results['Weeks'] = df_results['Datapoints']/24/4/7

df_results=df_results.reset_index()


df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
df_roster= df_roster[df_roster['FPtStatus'] != 'Dropped']
df_roster.drop(columns=['RecID','SiteID','FPtStatus'], inplace = True)
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)



df_counts = pd.merge(df_results, df_roster, on='PtID')

df_counts['Days_round']=round(df_counts.Days)

df_count_w = df_counts [df_counts ['Race'] == 'white']
df_count_b = df_counts [df_counts ['Race'] == 'black']

df_results = df_counts
value_counts = df_counts['Days_round'].value_counts().sort_index()

df_test = pd.DataFrame(value_counts)

# Plotting
plt.figure(figsize=(16, 6))  # You can adjust the dimensions as needed
value_counts.plot(kind='bar')
plt.xticks(rotation=45)  # Rotates labels to 45 degrees
plt.xlabel('Number of Days (rounded values)')
plt.ylabel('Number of Patients')
# plt.title('Frequency of Each Unique Value in the Column')
plt.show()







#%% convert my date and time into a datetime 

# get a Datetime in the df
df_cgm = days_time_to_datetime(df_cgm,'DeviceDaysFromEnroll','DeviceTm')

#%% plot data
df_ptid = df_roster['PtID'].unique()
df_ptid_cgm = df_cgm['PtID'].unique()
df_cgm_one = df_cgm[df_cgm['PtID']==df_ptid[0]]


# Plotting the data

plt.plot(df_cgm_one['Datetime'], df_cgm_one['CGM'], marker='o', linestyle='-')
# plt.title('')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.grid(True)
# Format the x-axis labels to display only date and time (without year)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))

plt.xticks(rotation=45)

plt.show()



#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6), dpi=300)  # 10x6 inches figure with 300 DPI

# Creating the histogram
plt.hist(df_cgm['CGM'], bins=100, edgecolor='black')  # 'bins' defines the number of intervals or "buckets"

# Adding title and labels
plt.title('CGM Data Distribution')
plt.xlabel('Blood Glucose Level (mmol/L)')
plt.ylabel('Frequency')

# Customizing x-axis ticks
min_value = min(df_cgm['CGM'])
max_value = max(df_cgm['CGM'])
tick_interval = 1.0  # Adjust as needed for appropriate granularity
ticks = np.arange(min_value, max_value + tick_interval, tick_interval)
plt.xticks(ticks)


# Displaying the plot
plt.show()



df_results.Weeks.mean()

#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6), dpi=300)  # 10x6 inches figure with 300 DPI

# Creating the histogram
plt.hist(df_baseline['DiagT1DAge'], bins=100, edgecolor='black')  # 'bins' defines the number of intervals or "buckets"

# Adding title and labels
plt.title('diagt1d Data Distribution')
plt.xlabel('diagt1d')
plt.ylabel('Frequency')

# Customizing x-axis ticks
# min_value = min(df_cgm['CGM'])
# max_value = max(df_cgm['CGM'])
# tick_interval = 1.0  # Adjust as needed for appropriate granularity
# ticks = np.arange(min_value, max_value + tick_interval, tick_interval)
# plt.xticks(ticks)


# Displaying the plot
plt.show()



# df_results.Weeks.mean()



#%% Taken from load data

# how much data per person?
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