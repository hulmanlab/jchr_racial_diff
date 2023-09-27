# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#%% nload data (not relevant)
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

#%% count

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

