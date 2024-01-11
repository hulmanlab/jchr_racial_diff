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
import pandas as pd
import my_utils
import random
import numpy as np
import pickle
#%% Load data

df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
df_roster= df_roster[df_roster['FPtStatus'] != 'Dropped']
df_roster.drop(columns=['RecID','SiteID','FPtStatus'], inplace = True)
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)
df = df_roster
#%%
group_column = "Race"
id_column = "PtID"

group1 = "black"
group2 = "white"
ratio = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
df_unique_gr1, df_unique_gr2 = my_utils.get_group_id(df, id_column, group_column, group1, group2)
# df_unique = df[id_column].unique()
    
#%%

dictionary = {}
for current_ratio in ratio:
    df_unique = df[id_column].unique()
    print(current_ratio)
    
    for PtID in df_unique_gr1:
        print(PtID)
        train_val_id_w = df_unique_gr1.copy()
        train_val_id_b = df_unique_gr2.copy()
        
        test_id_b = random.choice(train_val_id_b) # black PtID for testing

        # Remove test PtID from IDs
        filtered_array_w = train_val_id_w[train_val_id_w != PtID]
        filtered_array_b = train_val_id_b[train_val_id_b != test_id_b]
        
        # Calculate the number of elements to take from each array
        num_from_array1 = int(len(filtered_array_w) * (current_ratio / 100))
        num_from_array2 = int(len(filtered_array_b) * ((100 - current_ratio) / 100))

        # Randomly select elements from each array
        selected_from_array1 = np.random.choice(filtered_array_w, num_from_array1, replace=False)
        selected_from_array2 = np.random.choice(filtered_array_b, num_from_array2, replace=False)

        # Combine and return the result
        # combined_array = np.concatenate((selected_from_array1, selected_from_array2))
        if group1 == "white":
            dictionary[(PtID, current_ratio)] = {
                "test_w": PtID,
                "test_b": test_id_b,
                "training_w": selected_from_array1,
                "training_b": selected_from_array2,
                "ratio_w": current_ratio
            }
        if group1 == "black":
            dictionary[(PtID, current_ratio)] = {
                "test_b": PtID,
                "test_w": test_id_b,
                "training_w": selected_from_array2,
                "training_b": selected_from_array1,
                "ratio_b": current_ratio
            }

print(dictionary)


# # Specify the file path
# file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_3_data_split_b_v2.pkl"

# # Write to file
# with open(file_path, 'wb') as file:
#     pickle.dump(dictionary, file)