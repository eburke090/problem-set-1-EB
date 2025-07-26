'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr
# Your code here

#read in 
df_arrests = pd.read_csv('data/df_arrests.csv')

#create 2 dataframes
df_arrests_train, df_arrests_test = train_test_split(
    df_arrests,
    test_size=0.3,
    shuffle=True,
    stratify = df_arrests['y'],
    random_state=42
)

# Print class distribution in splits
print("\nClass distribution in training set:")
print(df_arrests_train['y'].value_counts())

print("\nClass distribution in test set:")
print(df_arrests_test['y'].value_counts())

#create features list
features = ['num_fel_arrests_last_year', 'current_charge_felony']

#create parameter grid 
param_grid = {'C': [0.01, 1, 100]}

#initalize regression model
lr_model = lr(random_state=42, solver='liblinear')

#intialize grid search 
gs_cv = GridSearchCV(
    estimator = lr_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)


#run model
gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])

#optimal c value 
opt_c = gs_cv.best_params_['C']
print(f"\nOptimal value for C: {opt_c }")

if opt_c  == min(param_grid['C']):
    reg_strength = "most regularization"
elif opt_c  == max(param_grid['C']):
    reg_strength = "least regularization"
else:
    reg_strength = "medium regularization"
print(f"Did it have the most, least, or medium regularization? {reg_strength}")

#predict test
df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features])

#save 
df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

