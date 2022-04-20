import pandas as pd
import numpy as np
import math

def read_data(data_path, column_names):
    """
    
    Processes a data file into Pandas.
    
    Params:
    data_path = file location for a dataset
    column_names = desired column names for the dataset
    
    Return:
    Pandas dataframe of specified file and column names. 
    
    """
    dataframe = pd.read_csv(data_path)
    dataframe.columns = column_names
    return dataframe

heart_failure_data_path = 'datasets/heart_disease.csv'
heart_failure_column_names = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_bp',
    'cholesterol',
    'fasting_bp',
    'resting_ecg',
    'max_hr',
    'exercise_angina',
    'oldpeak',
    'st_slope',
    'heart_disease'
]

heart_failure_df = read_data(
    heart_failure_data_path,
    heart_failure_column_names
)

print("Heart Failure Dataframe Value Counts: ")
print(heart_failure_df['age'].value_counts())

