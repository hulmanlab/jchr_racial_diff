#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:45:17 2023
Run after 2_preprocess

@author: au605715
"""
import pandas as pd
import my_utils
import random
import numpy as np
import pickle
#%%
# df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//cnn_ws60min_ph60min.csv')
# df.dropna(inplace=True)

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

group1 = "white"
group2 = "black"
ratio = [90, 80, 70, 60, 50]
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
        
        test_id_b = random.choice(train_val_id_b)

        # Remove test PtID from IDs
        filtered_array_w = train_val_id_w[train_val_id_w != PtID]
        filtered_array_b = train_val_id_b[train_val_id_b != PtID]
        
        # Calculate the number of elements to take from each array
        num_from_array1 = int(len(filtered_array_w) * (current_ratio / 100))
        num_from_array2 = int(len(filtered_array_b) * ((100 - current_ratio) / 100))

        # Randomly select elements from each array
        selected_from_array1 = np.random.choice(filtered_array_w, num_from_array1, replace=False)
        selected_from_array2 = np.random.choice(filtered_array_b, num_from_array2, replace=False)

        # Combine and return the result
        combined_array = np.concatenate((selected_from_array1, selected_from_array2))
        
        dictionary[(PtID, current_ratio)] = {
            "test_b": test_id_b,
            "training": combined_array,
            "ratio_w": current_ratio
        }

print(dictionary)


# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/data_split.pkl"

# Write to file
with open(file_path, 'wb') as file:
    pickle.dump(dictionary, file)