#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:15:47 2024

@author: au605715
"""

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


# df_roster = pd.read_csv(r'/home/hbt/jchr_data/jchr_racial_diff/data/FPtRoster.txt', sep='|')
df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FPtRoster.txt', sep='|')
df_roster= df_roster[df_roster['FPtStatus'] != 'Dropped']
df_roster.drop(columns=['RecID','SiteID','FPtStatus'], inplace = True)
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)


df_baseline = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/Data Tables/FBaseline.txt', sep='|')
df_baseline = df_baseline[df_baseline['PtID'].isin(df_roster['PtID'])]
edu_mapping = {
    
    "Doctorate Degree": '2',
    "Master's Degree": '2',
    "Bachelor's Degree": '2',
    "Professional Degree": '2',
    "Associate Degree": '2',
    "Some college but no degree": '2',
    
    "High school graduate/diploma/GED": '1',
    "12th Grade - no diploma": '1',
    "11th Grade": '1',
    "10th Grade": '1',
    "9th Grade": '1',
    "7th or 8th Grade": '1',
    "5th or 6th grade": '1',
    "1st, 2nd, 3rd, or 4th grade": '1'
    
    
    
    }

# Use the mapping dictionary to replace values in the 'EduLevel' column
df_baseline['EduLevel'] = df_baseline['EduLevel'].replace(edu_mapping)

#%%
df_roster2 = pd.read_csv(r'/Users/au605715/Documents/GitHub/study1/FPtRosterNew.txt', sep='|')
df_roster2['ageAtEnroll'] = df_roster2['ageAtEnroll'].apply(lambda x: 2 if x > 17 else 1)


#%%



def split_ptid(df,group_column, group1, group2, id_column="PtID"):
    
    ratio = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    df_unique_gr1, df_unique_gr2 = my_utils.get_group_id(df, id_column, group_column, group1, group2)
    # df_unique = df[id_column].unique()
    
    dictionary = {}
    for current_ratio in ratio:
        df_unique = df[id_column].unique()
        print(current_ratio)

        for PtID in df_unique:
            print(PtID)
            PtID_gr1 = df_unique_gr1.copy()
            PtID_gr2 = df_unique_gr2.copy()
            print(f"Checking for PtID: {PtID}")
            
            
            # finding out which group are the biggest
            if len(PtID_gr1) > len(PtID_gr2):
                larger_group = PtID_gr1.copy()
                smaller_group = PtID_gr2.copy()
            else:
                larger_group = PtID_gr2.copy()
                smaller_group = PtID_gr1.copy()
            
            # removing a PtID from the smalles group when the PtID occurs in the biggest group
            # this is done to make sure there are equal amounts of PtIDs in training no matter which group the test person is from
            if PtID in larger_group:
                print('removing a PtID from the smalles group when the PtID occurs in the biggest group')
                random_index=np.random.randint(0, len(smaller_group))
                smaller_group = np.delete(smaller_group, random_index)
                
                
            if len(PtID_gr1) > len(PtID_gr2):
                print('gr1 is biggest')
                PtID_gr1 = larger_group
                PtID_gr2 = smaller_group
      
            else:
                print('gr2 is biggest')
                PtID_gr2 = larger_group
                PtID_gr1 = smaller_group
                print(PtID_gr1)
                print('goat')
    
            # Remove test PtID from ID
            if PtID in PtID_gr1: 
                PtID_gr1 = PtID_gr1[PtID_gr1 != PtID]
                group= "child"
            elif PtID in PtID_gr2:
                PtID_gr2 = PtID_gr2[PtID_gr2 != PtID]
                group = 'adult'
            else:
                raise ValueError(f"PtID: {PtID} is not found in either group")
            
                
                # Randomly remove PtID from the longer array to make them the same length
            while len(PtID_gr1) > len(PtID_gr2):
                random_index = np.random.randint(0, len(PtID_gr1))
                PtID_gr1 = np.delete(PtID_gr1, random_index)
                
                
            while len(PtID_gr2) > len(PtID_gr1):
                random_index = np.random.randint(0, len(PtID_gr2))
                PtID_gr2 = np.delete(PtID_gr2, random_index)
                
    
                
            
            
            # Calculate the number of elements to take from each array
            num_from_array1 = int(len(PtID_gr1) * (current_ratio / 100))
            num_from_array2 = int(len(PtID_gr2) * ((100 - current_ratio) / 100))
            
    
            # Randomly select elements from each array
            selected_from_array1 = np.random.choice(PtID_gr1, num_from_array1, replace=False)
            selected_from_array2 = np.random.choice(PtID_gr2, num_from_array2, replace=False)
    
            dictionary[(PtID, current_ratio)] = {
                "PtID_test": PtID,
                "group": group,
                "ratio_gr1": current_ratio,
                "training_gr1": selected_from_array1,
                "training_gr2": selected_from_array2
    
            }
    return dictionary            

dictionary_race = split_ptid(df=df_roster,group_column="Race", group1="white", group2="black")
dictionary_gender = split_ptid(df=df_baseline, group_column="Gender", group1="M", group2="F")

df_baseline = df_baseline[df_baseline['EduLevelUnk'] != 1]
dictionary_edulevel = split_ptid(df=df_baseline, group_column="EduLevel", group1="1", group2="2")

dictionary_age = split_ptid(df=df_roster2, group_column="ageAtEnroll", group1=1, group2=2)
# print(dictionary)


#%%  Specify the file path
file_path_race = "/Users/au605715/Documents/GitHub/study1/1_3_data_split_race_v6.pkl"
file_path_gender = "/Users/au605715/Documents/GitHub/study1/1_3_data_split_gender_v6.pkl"
file_path_edulevel = "/Users/au605715/Documents/GitHub/study1/1_3_data_split_edulvl_v6.pkl"
file_path_age = "/Users/au605715/Documents/GitHub/study1/1_3_data_split_age_v6.pkl"
# Write to file
with open(file_path_race, 'wb') as file:
    pickle.dump(dictionary_race, file)
    
with open(file_path_gender, 'wb') as file:
    pickle.dump(dictionary_gender, file)
    
with open(file_path_edulevel, 'wb') as file:
    pickle.dump(dictionary_edulevel, file)
    
    
with open(file_path_age, 'wb') as file:
    pickle.dump(dictionary_age, file)