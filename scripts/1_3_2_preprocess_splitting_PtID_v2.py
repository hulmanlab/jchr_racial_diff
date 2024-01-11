#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:45:17 2023
Run after 2_preprocess
Gives a dictionary with patient IDs.
The script assigns PtIds to training and testing.
It systematically goes through all the IDs in group 1 and picks a random patient ID from group 2
group1 and group 2 can be switched around

@author: au605715
"""
#%% Load data
import pandas as pd
import my_utils
import random
import numpy as np
import pickle

file_path_roster = r'/home/hbt/jchr_data/jchr_racial_diff/data_replace_bg/Data Tables/HPtRoster.txt'
file_gender = r'/home/hbt/jchr_data/jchr_racial_diff/data_replace_bg/Data Tables/HScreening.txt'
df_roster = pd.read_csv(file_path_roster, sep='|')
df_gender = pd.read_csv(file_gender, sep='|')
df_roster= df_roster[df_roster['PtStatus'] != 'Dropped']

df_gender =df_gender[['PtID', 'Gender']]
df = df_gender[df_gender['PtID'].isin(df_roster['PtID'])]
#%%
group_column = "Gender"
id_column = "PtID"
group1 = "M"
group2 = "F"

ratio = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
df_unique_gr1, df_unique_gr2 = my_utils.get_group_id(df, id_column, group_column, group1, group2)
# df_unique = df[id_column].unique()
    
#%%

dictionary = {}

for current_ratio in ratio:
    df_unique = df[id_column].unique()
    print(current_ratio)
    
    for PtID in df_unique:
        print(PtID)
        PtID_w = df_unique_gr1.copy()
        PtID_b = df_unique_gr2.copy()
        
        print(f"Checking for PtID: {PtID}")
        if PtID in PtID_w: 
            # Remove test PtID from ID
            PtID_w = PtID_w[PtID_w != PtID]
            group= "male"
        elif PtID in PtID_b:
            # Remove test PtID from ID
            PtID_b = PtID_b[PtID_b != PtID]
            group = 'female'
        else:
            raise ValueError(f"{PtID} is not found in either group")
        
        # Calculate the number of elements to take from each array
        num_from_array1 = int(len(PtID_w) * (current_ratio / 100))
        num_from_array2 = int(len(PtID_b) * ((100 - current_ratio) / 100))

        # Randomly select elements from each array
        selected_from_array1 = np.random.choice(PtID_w, num_from_array1, replace=False)
        selected_from_array2 = np.random.choice(PtID_b, num_from_array2, replace=False)

        dictionary[(PtID, current_ratio)] = {
            "PtID_test": PtID,
            "gender": group,
            "ratio_m": current_ratio,
            "training_m": selected_from_array1,
            "training_f": selected_from_array2

        }
            



print(dictionary)


#%%  Specify the file path
file_path = "/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_3_2_data_split_mf_v3.pkl"

# Write to file
with open(file_path, 'wb') as file:
    pickle.dump(dictionary, file)


# %%
