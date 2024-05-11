# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 09:42:15 2023

@author: mhabayeb
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30) # set the maximum width
# Load the dataset in a dataframe object 
df = pd.read_csv(r'C:\Users\ashish.gupta\Downloads\Flask Activity Titanic Project\titanic3.csv')
# Explore the data check the column values
print(df.columns.values)
print (df.head())
print (df.info())
# Check for missing values
print(df.isnull().sum())
#or
print(len(df) - df.count())
# Check the strongest relationship with class from numeric fields
#Choose columns
include = ['age','sex', 'embarked', 'survived','pclass']
df_ = df[include]
#print(df_.corr())
"""
Explore again
"""

#Check for missing values for included columns
print(df_.isnull().sum())
#or
print(len(df_) - df.count())
print(df_['sex'].unique())
print(df_['embarked'].unique())
print(df_['survived'].unique())
print(df_['pclass'].unique())
print(df_['survived'].value_counts())   # unbalanced
"""
Split
"""
#Split the data using stratifed sampling

df_ = df_[df_['survived'].notna()]  # We need to drop one row whose class value is Nan


from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

for train_index, test_index in sss.split(df_, df_['survived']):
    train_set = df_.iloc[train_index]
    test_set = df_.iloc[test_index]

print(train_set.shape)
print(test_set['survived'].value_counts())
print(train_set['survived'].value_counts())
#We have the same percentage values for the class in both train and test
"""
Prepare the data for machine learning using pipelines and transformers
fill in missing values for age column with median age
handdle caterogical values in columns sex and embarked
scale the data
"""
# separate target from features
dependent_variable = 'survived'
cols = ['age', 'sex', 'pclass','embarked']

y_train = train_set[dependent_variable]
x_train = train_set[cols]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Get column names
cat_attribs = ['sex','embarked']
num_attribs = ['age']
ord_attribs = ['pclass']



cat_pipeline = Pipeline(steps=[('mode_cat',SimpleImputer(strategy="most_frequent")),
                           ("one_hot_encode",OneHotEncoder(drop='first'))])

ord_pipeline = Pipeline(steps=[('mode_ord',SimpleImputer(strategy="most_frequent")),
                           ("ordinal_encode",OrdinalEncoder())])

num_pipeline = Pipeline(steps=[('mode_num',SimpleImputer(strategy="median")),
                           ("num_encode",StandardScaler())])

col_transform = ColumnTransformer(transformers=[("cat",cat_pipeline,cat_attribs),("pclass", ord_pipeline,ord_attribs),("age",num_pipeline,num_attribs)])

#x_train.to_csv('test1.csv',index=False)
x_train_prepared=col_transform.fit_transform(x_train)

cols = ['sex', 'embarked_s', 'embarked_q', 'pclass','age']
x_train_prepared = pd.DataFrame(x_train_prepared,columns=cols)

#x_train_prepared.to_csv('test.csv',index=False)
"""
Assume we are doing Logistic regression
Let us build one with cross validation
"""
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(solver='lbfgs')
#build a model
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr,x_train_prepared, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)
"""
Hyperparameter tune
"""
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
C = randint(low=1, high=20)
penalty = ["l1", "l2"]
pram_grid = dict(C=C, penalty=penalty)

clf = RandomizedSearchCV(lr, pram_grid, random_state=42, n_iter=100,
                             cv=3, verbose=0, n_jobs=-1)
clf.fit(x_train_prepared, y_train)

print(clf.best_params_)
best_model = clf.best_estimator_ 

#{'C': 7, 'penalty': 'l2'}
"""
Dump the model, pilepline & scalar
"""

import joblib 
joblib.dump(best_model, r'C:\Users\ashish.gupta\Downloads\Flask Activity Titanic Project\my_model_titanic.pkl')
print("Model dumped!")
joblib.dump(col_transform,r'C:\Users\ashish.gupta\Downloads\Flask Activity Titanic Project\my_pipe_titanic.pkl')
print("Pipe dumped!")


