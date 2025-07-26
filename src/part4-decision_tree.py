'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
import os
if not os.path.exists('data/df_arrests_train.csv'):
    print("df_arrests_train.csv not found. Please run part 3 first.")
    exit(1)

if not os.path.exists('data/df_arrests_test.csv'):
    print("df_arrests_test.csv not found. Please run part 3 first.")
    exit(1)


#read dataframes
df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
df_arrests_test = pd.read_csv('data/df_arrests_test.csv')


#create parameter grid 
param_grid_dt = {'max_depth': [2, 5, 10]}

#initalize decision tree
dt_model= DTC(random_state=42)

#intialize grid search
gs_cv_dt = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid_dt,
    cv=5,
    scoring='accuracy'
)

#run model
feat = ['num_fel_arrests_last_year', 'current_charge_felony']
gs_cv_dt.fit(df_arrests_train[feat], df_arrests_train['y'])

#optimal max depth
opt_depth = gs_cv_dt.best_params_['max_depth']
print(f"Optimal value for max_depth: {opt_depth}")
#regulatization
if opt_depth == min(param_grid_dt['max_depth']):
    reg_strength = "most regularization"
elif opt_depth == max(param_grid_dt['max_depth']):
    reg_strength = "least regularization"
else:
    reg_strength = "medium regularization"
print(f"Did it have the most, least, or medium regularization? {reg_strength}")

#predict test 
df_arrests_test['pred_dt'] = gs_cv_dt.predict(df_arrests_test[feat])

#save for part5 
df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)
