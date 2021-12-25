# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data=pd.read_csv("train-data.csv")
train_data.head()

train_data.shape

train_data.isnull().sum()

len(train_data["Name"].unique())

train_data.drop(['New_Price','Unnamed: 0'],axis=1,inplace=True)

train_data.shape

train_data.dtypes

train_data['Engine']=train_data['Engine'].str.replace("CC"," ")
train_data['Power']=train_data['Engine'].str.replace("bhp"," ")
train_data['Mileage']=train_data['Mileage'].str.replace("kmpl"," ")
train_data['Mileage']=train_data['Mileage'].str.replace("km/kg"," ")

train_data['Engine']=train_data['Engine'].astype(float)
train_data['Mileage']=train_data['Mileage'].astype(float)

train_data.head()

train_data['Owner_Type'].value_counts()

train_data.corr()['Price']

train_data.plot(x='Engine',y='Price',kind='scatter')

from sklearn.model_selection import train_test_split

X=train_data.drop(['Price'],axis=1)
y=train_data['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

num_pipe1=make_pipeline(StandardScaler(),SimpleImputer(strategy='mean'))

cat_pipe2=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore'))

ct=make_column_transformer((num_pipe1,['Year','Kilometers_Driven','Mileage','Engine','Seats']),
                          (cat_pipe2,['Name','Location','Fuel_Type','Transmission','Owner_Type','Power']),
                          remainder='passthrough')

from sklearn.linear_model import LinearRegression

LR_pipeline=make_pipeline(ct,LinearRegression())
LR_pipeline.fit(X_train,y_train)

LR_pipeline.score(X_train,y_train),LR_pipeline.score(X_test,y_test)

print(X_test)

import xgboost as xgb

xg_pipeline=make_pipeline(ct,xgb.XGBRegressor(max_depth=9,n_estimators=600,learning_rate=0.06,colsample_bytree=0.38))
xg_pipeline.fit(X_train,y_train)

xg_pipeline.score(X_train,y_train)

xg_pipeline.score(X_test,y_test)

xg_pipeline.predict(X_test)

y_test