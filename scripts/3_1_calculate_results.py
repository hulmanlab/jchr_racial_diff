#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:56:35 2023

@author: au605715
"""

# from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import pickle
import pandas as pd
import my_utils


# Specify the file path
# file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/data_split.pkl"
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/2_1_predicted_results_cnn1_v6_1.pkl"

# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)


#%%

iteration = 1
dict_results = {}


for (PtID, percentage), value in dictionary.items():


    print('Iteration:',iteration,'PtID:',PtID,'ratio:' ,percentage)
    iteration=iteration+1
    
    # My actual/true values and my baseline value
    y_actual_w = value['y_test_w']
    y_actual_b = value['y_test_b']
    y_last_value_w = value['y_last_val_w']
    y_last_value_b = value['y_last_val_b']
    
    y_actual_tl_w = value['y_test_tl_w']
    y_actual_tl_b = value['y_test_tl_b']
    y__last_value_w = value['y_last_val_tl_w']
    y__last_value_b = value['y_last_val_tl_b']
        
    # calculating my values for each prediction base_model
    rmse_base_w = my_utils.calculate_results(y_actual_w, y_last_value_w)
    rmse_base_b = my_utils.calculate_results(y_actual_b, y_last_value_b)   
     
    y_pred_w = value['y_pred_w']
    rmse_w = my_utils.calculate_results(y_actual_w, y_pred_w)
    # mse_w = my_utils.calculate_mse(y_actual_w,y_pred_w )

     
    y_pred_b = value['y_pred_b']    
    rmse_b = my_utils.calculate_results(y_actual_b, y_pred_b)
    rmse_base_b = my_utils.calculate_results(y_actual_b, y_last_value_b)   
    
    # Transferlearned model
    
    # baseline values
    rmse_base_tl_w = my_utils.calculate_results(y_actual_tl_w, y__last_value_w)      
    rmse_base_tl_b = my_utils.calculate_results(y_actual_tl_b, y__last_value_b)   
    
    # transfer learned on white
    y_pred_tlw_w = value['y_pred_tlw_w']
    rmse_tlw_w = my_utils.calculate_results(y_actual_tl_w, y_pred_tlw_w)

    y_pred_tlw_b = value['y_pred_tlw_b']
    rmse_tlw_b = my_utils.calculate_results(y_actual_tl_b, y_pred_tlw_b)
 

    # transferlearned on black
    y_pred_tlb_w = value['y_pred_tlb_w']
    rmse_tlb_w = my_utils.calculate_results(y_actual_tl_w, y_pred_tlb_w)

    y_pred_tlb_b = value['y_pred_tlb_b']
    rmse_tlb_b = my_utils.calculate_results(y_actual_tl_b, y_pred_tlb_b)

    
    
    
    dict_results[(PtID, percentage)] = {
        
        "rmse_base_w": rmse_base_w,
        "rmse_base_b": rmse_base_b, 
        "rmse_w": rmse_w, 
        "rmse_b": rmse_b, 


        "rmse_base_tl_w": rmse_base_tl_w, # baseline for transferlearned white
        "rmse_base_tl_b": rmse_base_tl_b, # baseline for transferlearned black        
        "rmse_tlw_w": rmse_tlw_w, # transferlearned white 
        "rmse_tlw_b": rmse_tlw_b, 
        "rmse_tlb_w": rmse_tlb_w, # transfer learned black     
        "rmse_tlb_b": rmse_tlb_b, 


    }





#%%


# Assuming 'dictionary' is your input dictionary with dicts
data = []

for (PtID, percentage), metrics in dict_results.items():
    row = {'PtID': PtID, 'percentage': percentage}
    # Update row with keys that contain 'rmse' and do not contain 'mae' or 'mard'
    row.update({k: metrics[k] for k in metrics if 'rmse' in k and not ('mae' in k or 'mard' in k or 'loss' in k or 'val' in k or 'y_' in k or 'train' in k)})
    data.append(row)

df = pd.DataFrame(data)
df.rename(columns={'percentage': "ratio"}, inplace=True)


rmse_columns_ordered = ['PtID', 'ratio', 'rmse_base_w','rmse_w', 'rmse_base_b', 'rmse_b', 'rmse_base_tl_w','rmse_base_tl_b','rmse_tlw_w', 'rmse_tlw_b','rmse_tlb_w', 'rmse_tlb_b'] 
df = df[rmse_columns_ordered]


#%% save dataframe
# df.to_csv( "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/3_1_calculated_results_cnn1_v6.csv", index=False) 

