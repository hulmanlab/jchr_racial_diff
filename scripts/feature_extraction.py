#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:08:12 2023

@author: au605715
"""
import pandas as pd
from my_utils import feature_extraction_cnn

#%% import data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/racial_diff_dataclean.csv')

df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
df_roster.drop(columns=['RecID','SiteID'], inplace = True)
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)

#%% cnn

# create sample of 1 hour and prediction horizon of 1 hourho
df_feature_cnn = feature_extraction_cnn(df,window_size=4, prediction_horizon=8, col_patient_id='PtID', col_glucose='CGM')

# df_feature_cnn = feature_extraction_cnn(df,window_size=8, prediction_horizon=12, col_patient_id='PtID', col_glucose='CGM')
# df_feature_cnn_unique = df_feature_cnn.PtID.unique()
# df_one = df_feature_cnn[df_feature_cnn['PtID']==df_feature_cnn_unique[7]] #179
#%% add ethnicity
df_race = pd.DataFrame(data=[df_roster['PtID'],df_roster['Race']]).transpose()
df_feature_cnn = pd.merge(df_feature_cnn, df_race, on='PtID', how='left')

#%% For saving dataframes
# remember to change name to something meaningful
# df_feature_cnn.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/cnn_ws120min_ph60min.csv', index=False) # takes forever, maybe try with chunk_size = 10000 next time
# print('finished')





