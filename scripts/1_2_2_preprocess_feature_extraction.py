#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:08:12 2023

@author: au605715
"""
import pandas as pd
from my_utils import feature_extraction_cnn

#%% import data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_1_2_gender_diff_dataclean_v2.csv')
file_gender = r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/REPLACE-BG Dataset-79f6bdc8-3c51-4736-a39f-c4c0f71d45e5 (1)/Data Tables/HScreening.txt'
df_gender = pd.read_csv(file_gender, sep='|')
df_gender =df_gender[['PtID', 'Gender']]
# df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
# df_roster.drop(columns=['RecID','SiteID'], inplace = True)
# df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
# df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
# df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)

#%% cnn

# create sample of 1 hour and prediction horizon of 1 hourho
df_feature_cnn = feature_extraction_cnn(df,window_size=12, prediction_horizon=24, col_patient_id='PtID', col_glucose='CGM')

#%% add gender
# df_race = pd.DataFrame(data=[df_roster['PtID'],df_roster['Race']]).transpose()
df_feature_cnn = pd.merge(df_feature_cnn, df_gender, on='PtID', how='left')

#%% For saving dataframes
# remember to change name to something meaningful
df_feature_cnn.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/cnn_gender_ws60min_ph60min.csv', index=False) # takes forever, maybe try with chunk_size = 10000 next time
print('finished')





