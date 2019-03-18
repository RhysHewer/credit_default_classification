'''
Developer: Rhys Hewer
Project: Data Science - Credit Scoring
Version: 1
Purpose: Import and explore data
Output: .pkl file with non-numeric features as categoricals
'''

#import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

#load data
data = pd.read_csv('data/default of credit card clients.csv', header=1)


##### PRELIMINARY DATA EXPLORATION ############################################
#preliminary data exploration
data_head = data.head()
data.shape
data.columns
data.info()
data_summary = data.describe(include = 'all')

##### EXPLORE DEPENDENT VARIABLE ##############################################
#rename dependant variable
data.rename(columns={'default payment next month': 'default'}, inplace=True)

#Dependent variable exploration
data.default.value_counts().plot(kind = 'bar')
plt.title('Dependent Variable distribution')
plt.xlabel('default')
plt.ylabel('count')
plt.show()

data.default.value_counts(normalize=True)*100

##### EXPLORING FEATURES ######################################################
#exploring features
numeric_features = list(data.columns)
non_numeric = ['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'default']
numeric_list = [x for x in numeric_features if x not in non_numeric]

#recoding non-numeric as factors
data[non_numeric] = data[non_numeric].astype('category')

##### PLOTTING FEATURES #######################################################
#histograms
for i in numeric_list:
    plt.figure(i)
    plt.hist(data[i])

#boxplots/outliers    
for i in numeric_list:
    plt.figure(i)
    plt.boxplot(data[i])   

#exploring non-numeric features
for i in non_numeric:
    plt.figure(i)
    plt.hist(data[i])  
    
#numeric feature relationship with dependent variable
for i in numeric_list:
    plt.figure(i)
    sns.violinplot(data.default, data[i]) 

#non-numeric feature relationship with dependent variable
mosaic_list = [x for x in non_numeric if x not in ['ID', 'default']]
for i in mosaic_list:
    mosaic(data, ['default', i], title = i)

##### SAVE AS PICKLE ##########################################################
data.to_pickle('output/dataframes/data.pkl')
