#data preprocessing

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
 
 
 
# splitting the dataset in to training set and the testing set 
 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)
import matplotlib.pyplot as plt
x_train[x_train.dtypes[(x_train.dtypes=="float64")|(x_train.dtypes=="int64")].index.values].hist(figsize=[11,11])


 # Encoding the categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
le =  LabelEncoder()

for col in x_test.columns.values:
       # Encoding only categorical variables
       if x_test[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
           data=x_train[col].append(x_test[col])
           le.fit(data.values)
           x_train[col]=le.transform(x_train[col])
           x_test[col]=le.transform(x_test[col])
           
#ohe = OneHotEncoder()
#smoking_status=ohe.fit_transform(x_train[['smoking_status']]).toarray().astype(int)
#smoking_status[0]
#x_train.iloc[:,-1]
oh = pd.get_dummies(x_train['smoking_status'])
x_train = x_train.drop('smoking_status', axis=1)
x_train = x_train.join(oh) 
x_train.rename(columns={0: 'formerly smoked'}, inplace=True)
x_train.rename(columns={1: 'never smoked'}, inplace=True)
x_train.rename(columns={2: 'smokes'}, inplace=True)

oh = pd.get_dummies(x_train['gender'])
x_train = x_train.drop('gender', axis=1)
x_train = x_train.join(oh)
x_train.rename(columns={0: 'female'}, inplace=True)
x_train.rename(columns={1: 'male'}, inplace=True)
x_train.rename(columns={2: 'others'}, inplace=True)

oh = pd.get_dummies(x_train['work_type'])
x_train = x_train.drop('work_type', axis=1)
x_train = x_train.join(oh)
x_train.rename(columns={0: 'children'}, inplace=True)
x_train.rename(columns={3: 'Private'}, inplace=True)
x_train.rename(columns={2: 'Never_worked'}, inplace=True)
x_train.rename(columns={4: 'Self-employed'}, inplace=True)
x_train.rename(columns={1: 'Govt jobs'}, inplace=True)

oh = pd.get_dummies(x_test['smoking_status'])
x_test = x_test.drop('smoking_status', axis=1)
x_test = x_test.join(oh) 
x_test.rename(columns={0: 'formerly smoked'}, inplace=True)
x_test.rename(columns={1: 'never smoked'}, inplace=True)
x_test.rename(columns={2: 'smokes'}, inplace=True)

oh = pd.get_dummies(x_test['gender'])
x_test = x_test.drop('gender', axis=1)
x_test = x_test.join(oh)
x_test.rename(columns={0: 'female'}, inplace=True)
x_test.rename(columns={1: 'male'}, inplace=True)
x_test.rename(columns={2: 'others'}, inplace=True)

oh = pd.get_dummies(x_test['work_type'])
x_test = x_test.drop('work_type', axis=1)
x_test = x_test.join(oh)
x_test.rename(columns={0: 'children'}, inplace=True)
x_test.rename(columns={3: 'Private'}, inplace=True)
x_test.rename(columns={2: 'Never_worked'}, inplace=True)
x_test.rename(columns={4: 'Self-employed'}, inplace=True)
x_test.rename(columns={1: 'Govt jobs'}, inplace=True)