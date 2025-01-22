# Import packages

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from missforest import MissForest
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


# Define paths

# Define the base directory dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the CSV file
APPLICATION_TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'application_train.csv')

# Bureau table path 
BUREAU_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bureau.csv')

# Previous_application table path 
PREVIOUS_APPLICATION_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'previous_application.csv')

# Processed data directory
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Model Directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# Load data

df = pd.read_csv(APPLICATION_TRAIN_DATA_PATH)
df_bureau = pd.read_csv(BUREAU_DATA_PATH)
df_previous_application = pd.read_csv(PREVIOUS_APPLICATION_DATA_PATH)


# Preprocess data 

# Remove the four rows with 'XNA' gender due to insufficient data and imbalance
df = df[(df['CODE_GENDER'] == 'F') | (df["CODE_GENDER"] == 'M')]

# Select candidate columns that contain meaningful data and drop others
cleaned_df = df.loc[:, ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
                    'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
                    'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                    'OWN_CAR_AGE','OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'ORGANIZATION_TYPE',
                    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'LIVINGAREA_AVG','TOTALAREA_MODE',
                    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                    'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','FLAG_MOBIL',
                    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE','FLAG_EMAIL',
                    'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
                    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                    'LIVE_REGION_NOT_WORK_REGION','DAYS_LAST_PHONE_CHANGE'
                  ]
              ]

# Split the dataframe into numerical data and categorical data
cat = cleaned_df.select_dtypes('object')
num = cleaned_df.select_dtypes(include=['int64','float64'])

# Rename specific columns related to number of days
num = num.rename(columns={"DAYS_BIRTH": "YEARS_BIRTH",'DAYS_EMPLOYED':'YEARS_EMPLOYED','DAYS_REGISTRATION':'YEARS_REGISTRATION','DAYS_ID_PUBLISH':'YEARS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE':'YEARS_LAST_PHONE_CHANGE'})

# Calculate the number of years for each column in days
num['YEARS_BIRTH'] = num['YEARS_BIRTH'].abs()/365
num['YEARS_EMPLOYED'] = num['YEARS_EMPLOYED'].abs()/365
num['YEARS_REGISTRATION'] = num['YEARS_REGISTRATION'].abs()/365
num['YEARS_ID_PUBLISH'] = num['YEARS_ID_PUBLISH'].abs()/365
num['YEARS_LAST_PHONE_CHANGE'] = num['YEARS_LAST_PHONE_CHANGE'].abs()/365

# Change the noise values to the median, as the current values are nonsensical.
num["YEARS_EMPLOYED"][num["YEARS_EMPLOYED"]>1000] = num["YEARS_EMPLOYED"].median()

# Dropping unnecessary columns
num.drop(['AMT_REQ_CREDIT_BUREAU_QRT','FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'YEARS_LAST_PHONE_CHANGE'], axis=1, inplace=True)

# Fill five missing values of 'OWN_CAR_AGE' column with the mode value
num[(num['OWN_CAR_AGE'].isna()) & (cat["FLAG_OWN_CAR"] == 'Y')] = num[(num['OWN_CAR_AGE'].isna()) & (cat["FLAG_OWN_CAR"] == 'Y')].fillna(7)

# Fill nulls with median for specific columns based on observations and null percentages
num[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_REQ_CREDIT_BUREAU_YEAR']] = num[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_REQ_CREDIT_BUREAU_YEAR']].fillna(num[['DEF_60_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2', 'AMT_GOODS_PRICE','AMT_ANNUITY','AMT_REQ_CREDIT_BUREAU_YEAR']].median())

# Imputing some columns with missforest regression technique
mf = MissForest(rgr=RandomForestRegressor())
num_imputed = mf.fit_transform(num[['LIVINGAREA_AVG', 'TOTALAREA_MODE', 'EXT_SOURCE_3']])

# Update num dataframe with the imputed values using the RandomForestAlgorithm
num[['LIVINGAREA_AVG']] = num_imputed[['LIVINGAREA_AVG']]
num[['TOTALAREA_MODE']] = num_imputed[['TOTALAREA_MODE']]
num[['EXT_SOURCE_3']] = num_imputed[['EXT_SOURCE_3']]

# Concat numerical and categorical dataframes
df_application_train = pd.concat([num, cat], axis=1)

# Fill the remaining nulls
df_application_train['OCCUPATION_TYPE'].fillna(value='XNA', inplace=True)
df_application_train['OWN_CAR_AGE'].fillna(value=0, inplace=True)

# Feature engineering
''' 
Create a new feature by combining two existing features, 
calculating an overall rating for a specific region that includes the city. 
'''
df_application_train['OVERALL_REGION_RATING'] = df_application_train['REGION_RATING_CLIENT_W_CITY']/df_application_train['REGION_POPULATION_RELATIVE']
df_application_train['LOAN_REPAYMENT_PERIOD'] = df_application_train['AMT_CREDIT'] / df_application_train['AMT_ANNUITY']

# Dropping unnecessary columns
df_application_train = df_application_train.drop(['CNT_FAM_MEMBERS', 'AMT_CREDIT',
                                                'LIVINGAREA_AVG', 'OBS_30_CNT_SOCIAL_CIRCLE',
                                                'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                                                'REGION_RATING_CLIENT','LIVE_REGION_NOT_WORK_REGION','NAME_TYPE_SUITE',
                                                'REG_REGION_NOT_LIVE_REGION', 'YEARS_REGISTRATION', 'YEARS_ID_PUBLISH', 'REG_REGION_NOT_WORK_REGION'], axis=1)

# Handling outliers

# Drop the row having an anomaly in the 'AMT_INCOME_TOTAL' column
df_application_train.drop(df_application_train[df_application_train.AMT_INCOME_TOTAL > 100000000].index, inplace=True)


# Aggregate secondary tables

# Bureau table
total_loans = df_bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().reset_index()
total_loans.rename(columns={'SK_ID_BUREAU': 'TOTAL_LOANS'}, inplace=True)

average_delay = df_bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].mean().reset_index()
average_delay.rename(columns={'CREDIT_DAY_OVERDUE': 'AVG_REPAYMENT_DELAY'}, inplace=True)

# Previous_application table
count_previous_loans = df_previous_application.groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index()
count_previous_loans.rename(columns={'SK_ID_PREV': 'COUNT_PREVIOUS_LOANS'}, inplace=True)

average_loan_amount = df_previous_application.groupby('SK_ID_CURR')['AMT_APPLICATION'].mean().reset_index()
average_loan_amount.rename(columns={'AMT_APPLICATION': 'AVG_LOAN_AMOUNT'}, inplace=True)

# Merging tables

# Merging total loans
df_application_train = df_application_train.merge(total_loans, on='SK_ID_CURR', how='left')

# Merging average repayment delay
df_application_train = df_application_train.merge(average_delay, on='SK_ID_CURR', how='left')

# Merging count of previous loans
df_application_train = df_application_train.merge(count_previous_loans, on='SK_ID_CURR', how='left')

# Merging average loan amount
df_application_train = df_application_train.merge(average_loan_amount, on='SK_ID_CURR', how='left')


# Cleaning After Merging

# Filling missing values (if no loans exist for a customer)
df_application_train['TOTAL_LOANS'].fillna(0, inplace=True)
df_application_train['AVG_REPAYMENT_DELAY'].fillna(0, inplace=True)

# Filling missing values (if no previous loans exist for a customer)
df_application_train['COUNT_PREVIOUS_LOANS'].fillna(0, inplace=True)
df_application_train['AVG_LOAN_AMOUNT'].fillna(0, inplace=True)


# Normalize data

# Dividing dataframe into numerical and categorical data
numerical = df_application_train.select_dtypes(['int64','float64'])
categorical = df_application_train.select_dtypes('object')

numerical_columns = list(numerical.columns)
categorical_columns = list(categorical.columns)

# Utilize a robust scaler since the data contains significant outliers that should not be clipped.
scaler = RobustScaler()
scaled_data = scaler.fit_transform(numerical)
df_scaled = pd.DataFrame(scaled_data, columns= numerical_columns)

# Prepare data
df_scaled.drop(['SK_ID_CURR', 'REGION_RATING_CLIENT_W_CITY'], axis=1, inplace=True)

# Concat numerical and categorical dataframes
df_processed = pd.concat([df_scaled, categorical], axis=1)


# Save data
df_processed.to_csv(f'{PROCESSED_DIR}\df_processed_v0.3.csv')