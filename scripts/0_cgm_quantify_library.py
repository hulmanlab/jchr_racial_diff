

# I haven't downloaded cgm quantify to this library yet and values still needs to be checked
# this script also plots cgm data qith plotly, this is also only downloaded to the environment used in spyder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cgmquantify as cgm
import plotly.express as px
from datetime import datetime
from matplotlib.dates import date2num
# %matplotlib inline

def days_time_to_datetime(df, col_days_from_enroll, col_time,):
    """
    Converts days from enrollment and time into datetime
    Starts date is the 1970-01-01
    
    Parameters
    ----------
        df (pd.dataframe):
            Contains patient IDs, CGM data and time
        days_from_enroll (str):
            The name of the column .
        time (str):
            Name of the column containing time '%H:%M:%S'

    Returns  
    -------
        df['Datetime'] in the format

    """
    import datetime
    
    import pandas as pd
    df_temp = df.copy()
    df_temp['temp0'] = pd.to_datetime(df[col_time], format='%H:%M:%S')

    reference_date = pd.Timestamp('1970-01-01')
    df_temp['temp1'] = reference_date + pd.to_timedelta(df[col_days_from_enroll], unit='D')
    
    df_temp['Date']= df_temp['temp1'].dt.date
    df_temp['Time']= df_temp['temp0'].dt.time
    # merge date and time
    df_temp['Datetime'] = df_temp.apply(lambda row: datetime.datetime.combine(row['Date'], row['Time']), axis=1) 

    df_temp.reset_index(drop=True, inplace=True)

    df_temp.drop(['temp0','temp1','DeviceTm', 'Date', 'Time',col_days_from_enroll], axis=1, inplace=True)
    return df_temp


df = pd.read_csv(r'/Users/au605715/Documents/GitHub/study1/1_1_racial_diff_dataclean.csv')
df_cgm = pd.read_csv(r'/Users/au605715/Documents/GitHub/study1/Data/FDataCGM.txt', sep='|')

# ethnicity
df_roster = pd.read_csv(r'/Users/au605715/Documents/GitHub/study1/Data//FPtRoster.txt', sep='|')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic Black', 'black')
df_roster['RaceProtF'] = df_roster['RaceProtF'].replace('Non-Hispanic White', 'white')
df_roster.rename(columns={'RaceProtF':"Race"}, inplace=True)
df_roster= df_roster[df_roster['FPtStatus'] != 'Dropped']
df_roster.drop(columns=['RecID','SiteID', 'FPtStatus'], inplace = True)

df_roster_unique = df_roster['PtID'].unique()

df = df[df['PtID'].isin(df_roster_unique)]
df['Glucose']=df['CGM']*18


df_cgm = df_cgm[df_cgm['PtID'].isin(df_roster_unique)]
df_cgm = df_cgm.sort_values(by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm'])
df_cgm = days_time_to_datetime(df_cgm,'DeviceDaysFromEnroll','DeviceTm')
df_unique = df['PtID'].unique()
#%% Date

# Assuming 'Datetime' is already in a datetime-like format
df['Datetime'] = pd.to_datetime(df['Datetime'])
# Convert to the ISO 8601 format
df['Time'] = df['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
df['Time'] = pd.to_datetime(df['Time'])
# Extract just the date portion and store it in 'Day'
df['Day'] = df['Datetime'].dt.date
# Reset the index
df = df.reset_index(drop=True)


# Assuming 'Datetime' is already in a datetime-like format
df_cgm['Datetime'] = pd.to_datetime(df_cgm['Datetime'])
# Convert to the ISO 8601 format
df_cgm['Time'] = df_cgm['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
# Extract just the date portion and store it in 'Day'
df_cgm['Day'] = df_cgm['Datetime'].dt.date
# Reset the index
df_cgm = df_cgm.reset_index(drop=True)

#%%

df_one = df[df['PtID']==df_roster_unique[1]]

TIR_one = cgm.TIR(df_one)
print(TIR_one)
#%% Calculate values - for testing

def intradaysd(df):
    """
        Computes and returns the intraday standard deviation of glucose 
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            intradaysd_mean (float): intraday standard deviation averaged over all days
            intradaysd_medan (float): intraday standard deviation median over all days
            intradaysd_sd (float): intraday standard deviation standard deviation over all days
            
    """
    intradaysd =[]

    for i in pd.unique(df['Day']):
        intradaysd.append(np.std(df[df['Day'] == i]['Glucose']))


    intradaysd_mean = np.mean(intradaysd)
    intradaysd_median = np.median(intradaysd)
    intradaysd_sd = np.std(intradaysd)
    
    return intradaysd_mean, intradaysd_median, intradaysd_sd



df_one1 = df[df['PtID'] == df_unique[5]]
df_one2 = df_cgm[df_cgm['PtID'] == df_unique[5]]

# df_one1['Time'] = pd.to_datetime(df_one1['Time'])
# intradaysd.append(np.std(day_data))
# test_one = cgm.intradaysd(df_one1)
df_one1['Day_dt'] = pd.to_datetime(df_one1['Day'])
df_one1['Day'] = pd.to_datetime(df_one1['Day'])
df_one1['Day'] = df_one1['Day'].dt.day

print(df_one1.dtypes)
test_one = intradaysd(df_one1)


#%% Calculate values 2
# Create an empty DataFrame to store results
df_results = pd.DataFrame(columns=['PtID', 'TIR','PIR', 'TOR' , 'POR', 'GMI', 'MAGE', 'MODD', 'CONGA24',
                                   'LBGI', 'HBGI', 'ADRR', 'interday_cv','interday_sd',
                                   'intraday_cv', 'intraday_sd', 'NaN_sum'])
# add TOR and POR
for i in range(len(df_unique)):
    
    df_one = df[df['PtID'] == df_unique[i]]
    # df_one['Time'] = pd.to_datetime(df_one['Time'])
    df_one['Day'] = pd.to_datetime(df_one['Day'])
    df_one['Day'] = pd.to_datetime(df_one['Day'])
    df_one['Day'] = df_one['Day'].dt.day
    TIR_one = cgm.TIR(df_one, sr=15) # return results in minutes
    PIR_one = cgm.PIR(df_one, sr=15)
    TOR_one = cgm.TOR(df_one, sr=15)
    POR_one = cgm.POR(df_one, sr=15)
    GMI_one = cgm.GMI(df_one) # something is off with this value and the units
    MAGE_one = cgm.MAGE(df_one)
    MODD_one = cgm.MODD(df_one)
    LBGI_one = cgm.LBGI(df_one)
    HBGI_one = cgm.HBGI(df_one)
    ADRR_one = cgm.ADRR(df_one)
    CONGA24_one = cgm.CONGA24(df_one)

    interdaycv_one = cgm.interdaycv(df_one)
    intradaycv_one = cgm.intradaycv(df_one)

    interdaysd_one = cgm.interdaysd(df_one)
    print('###### GOAT1 #####')
    df_one['Time'] = df_one['Time'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    intradaysd_one = cgm.intradaysd(df_one)
    print('###### GOAT2 #####')
    # Convert the 'Time' column from datetime.time to a string format


    isna_one = df_one['Glucose'].isna().sum()
    
    
    # Create a temporary DataFrame to append
    # Uncomment the next line if MAGE and MODD are needed
    temp_df = pd.DataFrame({'PtID': [df_unique[i]], 'TIR': [TIR_one], 'PIR':[PIR_one],
                            'TOR': [TOR_one], 'POR': [POR_one], 'GMI': [GMI_one],
                            'MAGE': [MAGE_one], 'MODD': [MODD_one], 'CONGA24': [CONGA24_one],
                            'LBGI': [LBGI_one], 'HBGI': [HBGI_one], 'ADRR': [ADRR_one],
                            'interday_cv': [interdaycv_one],'interday_sd': [interdaysd_one],
                            'intraday_cv': [intradaycv_one], 'intraday_sd': [intradaysd_one],
                            'NaN_sum': [isna_one]})

    
    # Use pd.concat instead of append
    df_results = pd.concat([df_results, temp_df], ignore_index=True)
    print(df_unique[i])
    
df_merged = pd.merge(df_results, df_roster, on='PtID', how='inner')
df_merged.drop(columns=['interday_cv', 'interday_sd', 'intraday_cv', 'intraday_sd'], axis=1, inplace=True)

#%% Printing out results

df_merged['TIR'] = df_merged['TIR'].astype(float)
df_merged['TOR'] = df_merged['TOR'].astype(float)
df_merged['NaN_sum'] = df_merged['NaN_sum'].astype(float)

df_results_w = df_merged[df_merged['Race']=='white']
df_results_b = df_merged[df_merged['Race']=='black']
df_results_w.drop(columns=['Race'], axis=1, inplace=True)
df_results_b.drop(columns=['Race'], axis=1, inplace=True)

df_results_summary_w = df_results_w.describe()
df_results_summary_b = df_results_b.describe()
print ('goat')

# Calculate the median separately
median_w = df_results_w.median()
median_b = df_results_b.median()

# Add the median to the descriptive statistics
df_results_summary_w.loc['median'] = median_w
df_results_summary_b.loc['median'] = median_b

#%% Does it matter if i calculate in mg/dL and then convert to mmol/LBGI_one
# or should i convert from mg/dL to mmol/L and then calculate metrics?



df_one_cgm = 12.71 + 4.70587* df_one['CGM']
df_one_glucose = (3.31 + 0.02392 * df_one['Glucose'])
    # print(df_unique[i]


# print(dir(cgm))


# print(cgm.interdaysd(df_one))

# https://github.com/brinnaebent/cgmquantify/wiki/User-Guide

# cgm.plotglucosesd(df_one)
# cgm.plotglucosebounds(df_one, upperbound=20, lowerbound=0)
# cgm.plotglucosesmooth(df_one)

#%%

import plotly.io as pio
pio.renderers.default = 'browser'



# df_one = df[df['PtID']==11]
# df_one2 = df_cgm[df_cgm['PtID']==11]


bool_val = 1
i = 5

if bool_val == 1:
    df_one = df[df['PtID']==df_roster_unique[i]]
    dataset = "dataclean"
    df_one2 = df_cgm[df_cgm['PtID']==df_roster_unique[i]]
else:
    df_one = df_cgm[df_cgm['PtID']==df_roster_unique[i]]
    dataset = "raw"






df_one['Aggregate']=[10 if glucose_val >= 10 else 3.9 if glucose_val <=3.9 else 7 for glucose_val
                     in df_one['Glucose']]
df_one[['Datetime', 'Glucose', 'Aggregate']]

fig = px.line(df_one, x="Datetime", y=["Glucose"], # , "Aggregate"],
              title=f'{dataset}, PtId {df_roster_unique[i]}: GlucoseValue vs Time.', labels={'Glucose Value (mg/dL)':'Glucose Fluctuation'})
                                                    #,'Aggregate': 'Aggregate Blood Glucose'})

# Add horizontal line at y=3.9
fig.add_shape(
    type="line",
    x0=df_one["Datetime"].min(),
    y0=3.9,
    x1=df_one["Datetime"].max(),
    y1=3.9,
    line=dict(color="red", width=2, dash="dash"),  # Line style
)

# Add horizontal line at y=10
fig.add_shape(
    type="line",
    x0=df_one["Datetime"].min(),  # Start of the line (min datetime)
    y0=10,  # Y position of the line
    x1=df_one["Datetime"].max(),  # End of the line (max datetime)
    y1=10,  # Same Y position of the line
    line=dict(color="red", width=2, dash="dash"),  # Line style
)

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
            ])
    )
)

fig.show()


