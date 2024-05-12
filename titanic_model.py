# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 09:42:15 2023

@author: mhabayeb
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30) # set the maximum width
# Load the dataset in a dataframe object 
df = pd.read_csv('C:/courses_centennial/COMP309 new/data/titanic3.csv')
# Explore the data check the column values
print(df.columns.values)
print (df.head())
print (df.info())
# Check for missing values
print(df.isnull().sum())
#or
print(len(df) - df.count())
# Check the strongest relationship with class from numeric fields
print(df.corr())
#Choose columns
include = ['age','sex', 'embarked', 'survived']
df_ = df[include]
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
y_train = train_set[dependent_variable]
x_train = train_set[train_set.columns.difference([dependent_variable])]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Get column names
cat_attribs = ['sex','embarked']
num_attribs = ['age']
model_columns = list(train_set.columns)
cat_pipeline = Pipeline(steps=[('mode_cat',SimpleImputer(strategy="most_frequent")),
                           ("one_hot_encode",OneHotEncoder())])
col_transform = ColumnTransformer(transformers=[("cat",cat_pipeline,[1,2]),
                                               ("age",SimpleImputer(strategy="median"),[0])],
                                               
                                  )

x_train_prepared1=col_transform.fit_transform(x_train)

# still data is not scaled

scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train_prepared1)
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
score = np.mean(cross_val_score(lr,x_train_scaled, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
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
clf.fit(x_train_scaled, y_train)

print(clf.best_params_)
best_model = clf.best_estimator_ 

#{'C': 7, 'penalty': 'l2'}
"""
Dump the model, pilepline & scalar
"""

import joblib 
joblib.dump(best_model, 'C:/courses_centennial/COMP 247/Titanic_project/my_model_titanic.pkl')
print("Model dumped!")
joblib.dump(col_transform,'C:/courses_centennial/COMP 247/Titanic_project/my_pipe_titanic.pkl')
print("Pipe dumped!")
joblib.dump(scalar, 'C:/courses_centennial/COMP 247/Titanic_project/my_scalar_titanic.pkl')
print("Scalar dumped!")

