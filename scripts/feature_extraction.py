#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:08:12 2023

@author: au605715
"""
import pandas as pd
from my_utils import feature_extraction_cnn

#%% import data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/racial_diff_dataclean.csv')

#%% cnn

# create sample of 1 hour and prediction horizon of 1 hourho
df_feature_cnn = feature_extraction_cnn(df,window_size=4, prediction_horizon=4, col_patient_id='PtID', col_glucose='CGM')
#%% For saving dataframes
# remember to change name to something meaningful
# df_feature_cnn.to_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/cnn_ws60min_ph60min.csv', index=False) # takes forever, maybe try with chunk_size = 10000 next time
print('finished')

