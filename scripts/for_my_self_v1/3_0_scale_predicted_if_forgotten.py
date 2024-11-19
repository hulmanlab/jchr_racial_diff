
"""
To scale y_pred when forgotten in 2_1_model
Remember to check if random seed is the same as when generated the code



"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import my_utils
import pickle
from sklearn.model_selection import GroupShuffleSplit


def split_data_black_white_ratio_in_loop (df_group1, df_group2, ratio):
    """
    split data depending on the ratio of PtIDs from each group
    
    Parameters
    ----------
    df_group1 : Dataframe
        df containing data from group1
    df_group2 : Dataframe
        df containing data from group2.
    ratio : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train : TYPE
        data to train on.
    y_train : TYPE
        labels for data to train on.
    x_val : TYPE
        data to validate on (or test).
    y_val : TYPE
        labels for data to validate on (or test).

    """
    # split based on PtID
    if percentage == 100: #  if there are no patients in group 2
    
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_test_idx = next(gss.split(df_group1, groups=df_group1['PtID']))
    
        x_train_temp1 = df_group1.iloc[train_idx]
        x_val_temp1 = df_group1.iloc[val_test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp1, x_val_temp1)
    
    elif percentage == 0: # if there are no patients in group 1
     
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
        train_idx, test_idx = next(gss.split(df_group2, groups=df_group2['PtID']))
    
        x_train_temp2 = df_group2.iloc[train_idx]
        x_val_temp2 = df_group2.iloc[test_idx]
    
        x_train, y_train, x_val, y_val = my_utils.seperate_the_target(x_train_temp2, x_val_temp2)
        
    else: # if there are patients in group 2
        x_train, y_train, x_val, y_val = my_utils.get_groupShuflesplit_equal_groups(df_group1, df_group2,test_size=0.2, seperate_target=True)     # split based on PtID
    
    return x_train, y_train, x_val, y_val

#%%                     load data
df = pd.read_csv(r'/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data//1_2_cnn_ws60min_ph60min.csv')
df.dropna(inplace=True)

# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/1_3_data_split_b_v1.pkl"


# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)
    
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/2_1_predicted_results_cnn1_v6.pkl"
with open(file_path, 'rb') as file:
    dict_results = pickle.load(file)
    
    
#%%                     extract data from dictionary

testing_mode = False
group1_is_white = False

i = 0


for (PtID, percentage), value in dictionary.items():
    print(i)
    i=i+1
    
#%%                     get data from dictionary
    ptid_training_w = value['training_w']
    ptid_training_b = value['training_b']
    ptid_test_b = value['test_b']
    ptid_test_w = value['test_w']
    

    df_train_w = df[df['PtID'].isin(ptid_training_w)]
    df_train_b = df[df['PtID'].isin(ptid_training_b)]
    df_test_w = df[df['PtID']==ptid_test_w]
    df_test_b = df[df['PtID']==ptid_test_b]

#%%                        split dataset
    if group1_is_white==True:
        x_train, y_train, x_val, y_val = split_data_black_white_ratio_in_loop(df_train_w, df_train_b, percentage)
    else: # if group 1 is black
        x_train, y_train, x_val, y_val = split_data_black_white_ratio_in_loop(df_train_b, df_train_w, percentage)  
    
    # split test set x and y
    x_test_w, y_test_w = my_utils.seperate_the_target(df_test_w)
    x_test_b, y_test_b = my_utils.seperate_the_target(df_test_b)
    
    #%%                     Fine- tuning: split data

    # split within patients, train/test
    xy_train_tl_w, xy_test_tl_w = my_utils.split_within_PtID(df_test_w, numb_values_to_remove=-672, seperate_target=False) # split witin  PtID
    xy_train_tl_b, xy_test_tl_b = my_utils.split_within_PtID(df_test_b, numb_values_to_remove=-672, seperate_target=False) # 4values/hour * 24hour/day*7days/week = 672 values/week
    
    # split train in train/val with seperate targets
    x_train_tl_w, y_train_tl_w, x_val_tl_w, y_val_tl_w = my_utils.split_time_series_data(xy_train_tl_w, test_size=0.15)
    x_train_tl_b, y_train_tl_b, x_val_tl_b, y_val_tl_b = my_utils.split_time_series_data(xy_train_tl_b, test_size=0.15)
    
    # seperate target from test
    x_test_tl_w, y_test_tl_w = my_utils.seperate_the_target(xy_test_tl_w)
    x_test_tl_b, y_test_tl_b = my_utils.seperate_the_target(xy_test_tl_b)
    
    


#%%                     Scale data
    # min max normalization [0,1]
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    
    x_train_scal = scaler_x.transform(x_train)
    x_val_scal = scaler_x.transform(x_val)
    
    x_test_w_scal = scaler_x.transform(x_test_w)
    x_test_b_scal = scaler_x.transform(x_test_b)
    

    # finetuning: min max normalization
    scaler_tl_x = MinMaxScaler()
    scaler_tl_x.fit(x_train_tl_w)
    
    x_train_tl_w_scal = scaler_tl_x.transform(x_train_tl_w)
    x_train_tl_b_scal = scaler_tl_x.transform(x_train_tl_b)
    x_val_tl_w_scal = scaler_tl_x.transform(x_val_tl_w)
    x_val_tl_b_scal = scaler_tl_x.transform(x_val_tl_b)
    
    x_test_tl_w_scal = scaler_tl_x.transform(x_test_tl_w)
    x_test_tl_b_scal = scaler_tl_x.transform(x_test_tl_b)

#%%                 Scale y data
    scaler_y = MinMaxScaler()

    # Reshape and then fit
    y_train_reshaped = y_train.values.reshape(-1, 1)
    scaler_y.fit(y_train_reshaped)
    
    # Transform the datasets
    y_train = scaler_y.transform(y_train_reshaped)
    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))   
    y_test_w = scaler_y.transform(y_test_w.values.reshape(-1, 1))
    y_test_b = scaler_y.transform(y_test_b.values.reshape(-1, 1))


    
    scaler_tl_y_w = MinMaxScaler()
    # Reshape and fit the scaler on the training data
    y_train_tl_w_reshaped = y_train_tl_w.values.reshape(-1, 1)
    scaler_tl_y_w.fit(y_train_tl_w_reshaped)
    
    # Transform the datasets using the fitted scaler
    y_train_tl_w = scaler_tl_y_w.transform(y_train_tl_w_reshaped)
    y_val_tl_w = scaler_tl_y_w.transform(y_val_tl_w.values.reshape(-1, 1))
    y_test_tl_w = scaler_tl_y_w.transform(y_test_tl_w.values.reshape(-1, 1))
    
    
    
    scaler_tl_y_b = MinMaxScaler()
    # Reshape and fit the scaler on the training data
    y_train_tl_b_reshaped = y_train_tl_b.values.reshape(-1, 1)
    scaler_tl_y_b.fit(y_train_tl_b_reshaped)
    
    # Transform the datasets using the fitted scaler
    y_train_tl_b = scaler_tl_y_b.transform(y_train_tl_b.values.reshape(-1, 1))
    y_val_tl_b = scaler_tl_y_b.transform(y_val_tl_b.values.reshape(-1, 1))
    y_test_tl_b = scaler_tl_y_b.transform(y_test_tl_b.values.reshape(-1, 1))



#%% extract values from dict_results

    result_value = dict_results[PtID, percentage]  # or any key you are using

    y_pred_test_w = result_value['y_pred_w']
    y_pred_test_b = result_value['y_pred_b']
    
    y_pred_test_tlw_w = result_value['y_pred_tlw_w']
    y_pred_test_tlw_b = result_value['y_pred_tlw_b']
    
    y_pred_test_tlb_w = result_value['y_pred_tlb_w']
    y_pred_test_tlb_b = result_value['y_pred_tlb_b']
    
    
    y_pred_test_tlb_b = result_value['y_pred_tlb_b']
    
    
    baseline_test_w = df_test_w['Value_4']
    baseline_test_b = df_test_b['Value_4']
    
    baseline_test_w.reset_index(drop=True, inplace=True)
    baseline_test_b.reset_index(drop=True, inplace=True)
    
    
    
    baseline_test_tl_w = xy_test_tl_w['Value_4']
    baseline_test_tl_b = xy_test_tl_b['Value_4']
    
    baseline_test_tl_w.reset_index(drop=True, inplace=True)
    baseline_test_tl_b.reset_index(drop=True, inplace=True)
    
    
    # Used to turn the y_last values into an array float 32 instead of being a series. Should be corrected in 2_1_model now, but just in case
    # y_last_val_w = result_value['y_last_val_w']
    # y_last_val_b = result_value['y_last_val_b']
    
    # y_last_val_tl_w = result_value['y_last_val_tl_w']   
    # y_last_val_tl_b = result_value['y_last_val_tl_b']
    


#%% Scale y_label back

    
    #scale target back
    y_test_w = scaler_y.inverse_transform(y_test_w)
    y_test_b = scaler_y.inverse_transform(y_test_b)
    
    y_test_tl_w = scaler_tl_y_w.inverse_transform(y_test_tl_w)
    y_test_tl_b = scaler_tl_y_b.inverse_transform(y_test_tl_b)
    
    # Scale predicted back
    y_pred_test_w = scaler_y.inverse_transform(y_pred_test_w)
    y_pred_test_b = scaler_y.inverse_transform(y_pred_test_b)
    
    y_pred_test_tlw_w = scaler_tl_y_w.inverse_transform(y_pred_test_tlw_w)
    y_pred_test_tlw_b = scaler_tl_y_b.inverse_transform(y_pred_test_tlw_b)
    
    y_pred_test_tlb_w = scaler_tl_y_b.inverse_transform(y_pred_test_tlb_w)
    y_pred_test_tlb_b = scaler_tl_y_b.inverse_transform(y_pred_test_tlb_b)
    #%% Calculate results
    
    
    
    #%% My actual/true values and my baseline value
    y_actual_w = y_test_w
    y_actual_b = y_test_b 
    y_last_val_w = baseline_test_w.to_numpy()
    y_last_val_b = baseline_test_b.to_numpy()
    
    y_actual_tl_w = y_test_tl_w
    y_actual_tl_b = y_test_tl_b
    y_last_val_tl_w = baseline_test_tl_w.to_numpy()
    y_last_val_tl_b = baseline_test_tl_b.to_numpy()
        
    #%% calculating my values for each prediction base_model
    rmse_base_w = my_utils.calculate_results(y_actual_w, y_last_val_w)
    rmse_base_b = my_utils.calculate_results(y_actual_b, y_last_val_b)   
    
    #%%
    # y_pred_w = value['y_pred_w']
    rmse_w = my_utils.calculate_results(y_actual_w, y_pred_test_w)

    
    # y_pred_b = value['y_pred_b']    
    rmse_b = my_utils.calculate_results(y_actual_b, y_pred_test_b)
 
    
    #%% Transferlearned model
    
    # baseline values
    rmse_base_tl_w = my_utils.calculate_results(y_actual_tl_w, y_last_val_tl_w)      
    rmse_base_tl_b = my_utils.calculate_results(y_actual_tl_b, y_last_val_tl_b)   
    
    # transfer learned on white
    # y_pred_tlw_w = value['y_pred_tlw_w']
    rmse_tlw_w  = my_utils.calculate_results(y_actual_tl_w, y_pred_test_tlw_w)

    # y_pred_tlw_b = value['y_pred_tlw_b']
    rmse_tlw_b  = my_utils.calculate_results(y_actual_tl_b, y_pred_test_tlw_b)
 

    # transferlearned on black
    # y_pred_tlb_w = value['y_pred_tlb_w']
    rmse_tlb_w = my_utils.calculate_results(y_actual_tl_w, y_pred_test_tlb_w)

    # y_pred_tlb_b = value['y_pred_tlb_b']
    rmse_tlb_b = my_utils.calculate_results(y_actual_tl_b, y_pred_test_tlb_b)
    
    

    result_value['y_pred_w'] = y_pred_test_w
    result_value['y_pred_b'] = y_pred_test_b 
    
    result_value['y_pred_tlw_w'] = y_pred_test_tlw_w
    result_value['y_pred_tlw_b'] = y_pred_test_tlw_b
    
    result_value['y_pred_tlb_w'] = y_pred_test_tlb_w
    result_value['y_pred_tlb_b'] = y_pred_test_tlb_b
    
    
            
    # result_value['y_last_val_w'] = y_last_val_w.astype('float32')
    # result_value['y_last_val_b'] = y_last_val_b.astype('float32')
    
    # result_value['y_last_val_tl_w'] = y_last_val_tl_w.astype('float32')
    # result_value['y_last_val_tl_b'] = y_last_val_tl_b.astype('float32')
    
    
    #%% save results

    if testing_mode==True:
        break
    
    
#%%


data = []

for (PtID, percentage), metrics in dict_results.items():
    row = {'PtID': PtID, 'percentage': percentage}
    # Update row with keys that contain 'rmse' and do not contain 'mae' or 'mard'
    row.update({k: metrics[k] for k in metrics if 'rmse' in k and not ('mae' in k or 'mard' in k or 'loss' in k or 'val' in k or 'y_' in k or 'train' in k)})
    data.append(row)


df = pd.DataFrame(data)
# df.rename(columns={'percentage': "ratio_w"}, inplace=True)

    

#%% save results


# Specify the file path

if testing_mode == False:
    file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/2_1_predicted_results_cnn1_v6_1.pkl"


    with open(file_path, 'wb') as file:
        # Serialize and save the list to the file
        pickle.dump(dict_results, file)
    

    
    
    
