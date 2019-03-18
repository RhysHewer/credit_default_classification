'''
Developer: Rhys Hewer
Project: Data Science - Credit Scoring
Version: 1
Purpose: model data
Output: .pkl of test set with predictions
'''

#import packages
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

import eli5
from eli5.sklearn import PermutationImportance


#load data
data = pd.read_pickle('output/dataframes/preprocessed.pkl')

##### DATA PREPARATION ########################################################
dep_variable = 'default'
non_model_variables = 'ID'
remove_list = [dep_variable, non_model_variables]

y = data[dep_variable].astype('int64')

data = data.drop(columns=remove_list)

#dummify categorical variables
category_columns = list(data.select_dtypes(include='category'))
category_list = [x for x in category_columns if x not in remove_list]

X = pd.get_dummies(data, drop_first=True, columns=category_list, prefix=category_list)

#split data for modelling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)

##### MODELLING ###############################################################
#instantiate the estimator (create the model)
mod = RandomForestClassifier(random_state=111) 

#cross validation
CV_score = cross_val_score(mod, X, y, cv = 5)

#fit (train) the model
mod.fit(X_train, y_train)

#make predictions on the testing set
y_pred = mod.predict(X_test)

#metrics
print('accuracy =', accuracy_score(y_test, y_pred)*100, '%')

#confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted']))

conf_mat_plot = pd.DataFrame(conf_mat)
sns.set(font_scale=1.4)
sns.heatmap(conf_mat_plot, annot=True, fmt="d", linewidths=.5, cbar=False)

#feature importance
perm = PermutationImportance(mod, random_state=1).fit(X_test, y_test)
imp_data = eli5.explain_weights_df(perm, feature_names = X_test.columns.tolist())

##### OUTPUT ##################################################################
#recombine to test set df
test_set = X_test
test_set['default'] = y_test
test_set['preds'] = y_pred

test_set.to_pickle('output/dataframes/test_set.pkl')
