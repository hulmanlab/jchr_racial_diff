
#%%
import pandas as pd
import pickle


# Specify the file path
file_path = "/home/hbt/jchr_data/jchr_racial_diff/results/preprocessed_data/1_3_1_data_split_wb_v3.pkl"
# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)
    
    
#%%    
PtID =  1
Percentage = 0
for (PtID, percentage), value in dictionary.items():
    ptid_training_w = value['training_w']
    ptid_training_b = value['training_b']

    
    break
    
# %%
