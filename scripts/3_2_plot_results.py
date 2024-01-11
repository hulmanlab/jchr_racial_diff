#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:02:33 2023

@author: au605715
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle


file_path_csv = r"/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/3_1_calculated_results_cnn1_v6.csv"
df = pd.read_csv(file_path_csv)


file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/2_1_predicted_results_cnn1_v6_1.pkl"

# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)



#%%
# List of RMSE columns
rmse_columns = [col for col in df.columns if 'rmse_' in col]

# Initialize a list to store each row of the final DataFrame
rows_list = []

# Loop over each ratio_w value
for ratio in df['ratio'].unique():
    # Filter the DataFrame for the current ratio
    df_ratio = df[df['ratio'] == ratio]
    
    # Initialize a dictionary to store the stats for the current ratio
    stats_dict = {'ratio': ratio}
    
    # Calculate the statistics for each RMSE column
    for col in rmse_columns:
        col_data = df_ratio[col].dropna()  # Exclude NaN values for the calculation
        mean = col_data.mean()
        median = col_data.median()
        std = col_data.std()
        # Confidence interval calculation
        ci_lower = mean - 1.96 * (std / np.sqrt(len(col_data)))
        ci_upper = mean + 1.96 * (std / np.sqrt(len(col_data)))
        
        # Store the statistics in the dictionary
        stats_dict[f'{col}_mean'] = mean
        stats_dict[f'{col}_median'] = median
        stats_dict[f'{col}_std'] = std
        stats_dict[f'{col}_95p_CI_Lower'] = ci_lower
        stats_dict[f'{col}_95p_CI_Upper'] = ci_upper

    # Append the stats for the current ratio to the rows_list
    rows_list.append(stats_dict)

# Create the final DataFrame
df_stats = pd.DataFrame(rows_list)



#%%

file_name_with_extension = file_path_csv.split("/")[-1]
file_name = file_name_with_extension.split(".")[0]

sns.set_style("whitegrid")
plt.figure(figsize=(12, 10))


rmse_col_w = 'rmse_w'
rmse_col_b = 'rmse_b'

rmse_col_tlw_w = 'rmse_tlw_w'
rmse_col_tlb_b = 'rmse_tlb_b'

gr1_color = "#5ec962"
gr2_color = "#440154"
alpha_value = 0.5  # saturation
marker1 = "o"
marker2 = "^"
y_group1 = df_stats['rmse_w_mean']
y_group2 = df_stats['rmse_b_mean']

y_group1_CI95_l = df_stats['rmse_w_95p_CI_Lower']
y_group1_CI95_u = df_stats['rmse_w_95p_CI_Upper']
y_group2_CI95_l = df_stats['rmse_b_95p_CI_Lower']
y_group2_CI95_u = df_stats['rmse_b_95p_CI_Upper']



y_group1_tl = df_stats['rmse_tlw_w_mean']
y_group2_tl = df_stats['rmse_tlb_b_mean']

y_group1_CI95_l_tl = df_stats['rmse_tlw_w_95p_CI_Lower']
y_group1_CI95_u_tl = df_stats['rmse_tlw_w_95p_CI_Upper']
y_group2_CI95_l_tl = df_stats['rmse_tlb_b_95p_CI_Lower']
y_group2_CI95_u_tl = df_stats['rmse_tlb_b_95p_CI_Upper']

x_values = df_stats['ratio']
x_tl_values = df_stats['ratio']

# Calculate the errors from the means to the confidence interval limits
yerr_group1 = [y_group1 - y_group1_CI95_l, y_group1_CI95_u - y_group1]
yerr_group2 = [y_group2 - y_group2_CI95_l, y_group2_CI95_u - y_group2]

yerr_group1_tl = [y_group1_tl - y_group1_CI95_l_tl, y_group1_CI95_u_tl - y_group1_tl]
yerr_group2_tl = [y_group2_tl - y_group2_CI95_l_tl, y_group2_CI95_u_tl - y_group2_tl]

# Plot the means with error bars
plt.errorbar(x=x_values, y=y_group1, yerr=yerr_group1, fmt=marker1, color=gr1_color, label='rmse_w', capsize=5) # , elinewidth=10, markersize=15, capthick=10)
plt.errorbar(x=x_values, y=y_group2, yerr=yerr_group2, fmt=marker1, color=gr2_color, label='rmse_b', capsize=5) # elinewidth=10, markersize=14, capthick=10)


# Plot the transfer learning means with error bars
plt.errorbar(x=x_tl_values, y=y_group1_tl, yerr=yerr_group1_tl, fmt=marker2, color=gr1_color, label='rmse_tlw_w', capsize=5, markersize=8)
plt.errorbar(x=x_tl_values, y=y_group2_tl, yerr=yerr_group2_tl, fmt=marker2, color=gr2_color, label='rmse_tlb_b', capsize=5, markersize=8)


# Set the legend and labels
plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='20')
plt.title(f'{file_name}: \n Mean RMSE and 95%CI', fontsize=30)
plt.xlabel('Ratio [%]', fontsize=24)
plt.ylabel('Mean RMSE \n [mmol/L]', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(df_stats['ratio'])
# plt.ylim(0, 3)
# Show the plot
plt.show()

y_diff1 = y_group1-y_group1_tl
y_diff2 = y_diff1.mean()


y_diff3 = y_group2-y_group2_tl
y_diff4 = y_diff3.mean()


#%% Plot to compare to baseline


# Define a function to create the plot based on the RMSE column names
def plot_rmse(df, df_stats, rmse_col_w, rmse_col_b, file_name, gr1_color = 'b', gr2_color = 'r'):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # Group by ratio and calculate means
    df_mean = df.groupby('ratio')[[rmse_col_w, rmse_col_b]].mean().reset_index()

    # Get the mean and standard deviation values
    y_group1 = df_mean[rmse_col_w]
    y_group2 = df_mean[rmse_col_b]
    y_group1_CI95 = df_stats[rmse_col_w + '_std']
    y_group2_CI95 = df_stats[rmse_col_b + '_std']

    # Plot settings
    my_s = 150
    my_s2 = 100

    # Plot the means
    plt.scatter(x=df_mean['ratio'], y=y_group1, label=rmse_col_w, color=gr1_color, s=my_s)
    plt.scatter(x=df_mean['ratio'], y=y_group2, label=rmse_col_b, color=gr2_color, s=my_s)

    # Plot the standard deviations
    plt.scatter(x=df_mean['ratio'], y=y_group1 - y_group1_CI95, color=gr1_color, marker='_', s=my_s2, label = f'std of {rmse_col_w}:')
    plt.scatter(x=df_mean['ratio'], y=y_group1 + y_group1_CI95, color=gr1_color, marker='_', s=my_s2)
    plt.scatter(x=df_mean['ratio'], y=y_group2 - y_group2_CI95, color=gr2_color, marker='_', s=my_s2, label = f'std of {rmse_col_b}:')
    plt.scatter(x=df_mean['ratio'], y=y_group2 + y_group2_CI95, color=gr2_color, marker='_', s=my_s2)

    # Set the legend and labels
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='20')
    plt.title(f'{file_name}: \n Mean RMSE and Standard Deviation of {rmse_col_w} and {rmse_col_b}', fontsize=30)
    plt.xlabel('Ratio', fontsize=24)
    plt.ylabel('Mean RMSE', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Show the plot
    plt.show()



# plotting baselines to see if my model performs better then the naive approach
# rmse_pairs_base = [
#     ('rmse_base_w', 'rmse_w'),
#     ('rmse_base_b', 'rmse_b'),
#     ('rmse_base_tl_w', 'rmse_tlw_w'),
#     ('rmse_base_tl_b', 'rmse_tlw_b'),
#     ('rmse_base_tl_w', 'rmse_tlb_w'),
#     ('rmse_base_tl_b', 'rmse_tlb_b')
# ]

# for rmse_w, rmse_b in rmse_pairs_base:
#     plot_rmse(df, df_stats, rmse_w, rmse_b, file_name, gr1_color='g')



# #%% plot predictions

# # Assuming you want values for a specific PtID
# desired_ptID = 'your_target_ptID'

# # Accessing the values directly
# percentage, value = dictionary[desired_ptID]

# # Accessing the specific values for the desired PtID
# y_actual_w = value['y_test_w']


#%% calculate the mean and CI for all RMSE

# val_rmse_w_mean = df.rmse_w.mean()
df_used = df.rmse_tlb_b
val_rmse_mean = df_used.mean()

val_rmse_std = df_used.std()

val_ci_lower = val_rmse_mean - 1.96*(val_rmse_std / np.sqrt(len(df_used)))
val_ci_upper = val_rmse_mean + 1.96*(val_rmse_std / np.sqrt(len(df_used)))

