'''
Developer: Rhys Hewer
Project: Data Science - Credit Scoring
Version: 1
Purpose: Clean data
Output: preprocessed .pkl file
'''

#import packages
import pandas as pd
import matplotlib.pyplot as plt

#load data
data = pd.read_pickle('output/dataframes/data.pkl')

##### DATA CLEANING ###########################################################
#missing values
data.isnull().values.any()

#data types
data.info()

#incorrect zero values
data_describe = data.describe()

#outliers
numeric_features = list(data.columns)
non_numeric = ['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'default']
numeric_list = [x for x in numeric_features if x not in non_numeric]
for i in numeric_list:
    plt.figure(i)
    plt.boxplot(data[i])  
    
    
##### OUTPUT ##################################################################
data.to_pickle('output/dataframes/preprocessed.pkl')
