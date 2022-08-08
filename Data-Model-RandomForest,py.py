# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 02:55:29 2022

@author: taran
"""

#Importing data set
import pandas as pd

df = pd.read_csv("D:/Hackathon_On_Email_Clicks/train_data.csv")

#Checking Missing Values
df.isnull().sum().sort_values(ascending=False)/df.shape[0]

#Checking Variance of each column
df['sender'].value_counts() #Very Low Variance
df['subject_len'].value_counts() #High variance
df['body_len'].value_counts() #High variance
df['mean_paragraph_len'].value_counts() #High Variance
df['day_of_week'].value_counts() #High Variance
df['is_weekend'].value_counts() #Medium Variance
df['times_of_day'].value_counts() #Medium Variance
df['category'].value_counts() #Medium Variance
df['product'].value_counts()  
df['no_of_CTA'].value_counts()
df['mean_CTA_len'].value_counts()
df['is_image'].value_counts()
df['is_personalised'].value_counts()#Very low variance
df['is_quote'].value_counts()
df['is_timer'].value_counts() #No variance
df['is_emoticons'].value_counts() #Very Low Variance
df['is_discount'].value_counts() #Very low variance
df['is_price'].value_counts() #very low variance
df['is_urgency'].value_counts() #very Low Variance
df['target_audience'].value_counts()
df['click_rate'].value_counts()

#Exploratory Data Analysis
import seaborn as sns
import matplotlib as plt
#Explore different plots
sns.pairplot(df, hue ='sex') 

df['is_weekend'].value_counts().plot.bar()
#Likewise for each column

sns.heatmap(df[['subject_len', 'body_len', 'mean_paragraph_len', 'mean_CTA_len']].corr(), cmap='Blues', annot=True)
plt.show()


#Encoding Categorical Data Variables 
#One hot encoding
sender = pd.get_dummies(df.sender, prefix='sender')
df = pd.concat([df,sender],axis=1)
df.drop(columns='sender', inplace=True)

#One hot encoding
day_of_week = pd.get_dummies(df.day_of_week, prefix='day_of_week')
df = pd.concat([df,day_of_week],axis=1)
df.drop(columns='day_of_week', inplace=True)

#One hot encoding
times_of_day = pd.get_dummies(df.times_of_day, prefix='times_of_day')
df = pd.concat([df,times_of_day],axis=1)
df.drop(columns='times_of_day', inplace=True)

#One hot encoding
category = pd.get_dummies(df.category, prefix='category')
df = pd.concat([df,category],axis=1)
df.drop(columns='category', inplace=True)

#One hot encoding
target_audience = pd.get_dummies(df.target_audience, prefix='target_audience')
df = pd.concat([df,target_audience],axis=1)
df.drop(columns='target_audience', inplace=True)

#One hot encoding
product = pd.get_dummies(df['product'])
df = pd.concat([df,product], axis=1)
df = df.drop(['product'], axis=1)


#Feature Engineering

df['is_image'].value_counts()
df['more_than_1_image']=df.is_image.apply(lambda x:1 if x>1 else 0)
df = df.drop(['is_image'], axis=1)

df['is_quote'].value_counts()
df['more_than_1_quote']=df.is_quote.apply(lambda x:1 if x>1 else 0)
df = df.drop(['is_quote'], axis=1)

df['is_emoticons'].value_counts()
df['more_than_1_emoticons']=df.is_emoticons.apply(lambda x:1 if x>1 else 0)
df = df.drop(['is_emoticons'], axis=1)

df['is_price'].value_counts()
df['more_than_1_price']=df.is_price.apply(lambda x:1 if x>1 else 0)
df = df.drop(['is_price'], axis=1)


df = df.drop(['campaign_id'], axis=1)

df = df.drop(['is_timer'], axis=1)

#separate the other attributes from the predicting attribute
x = df.drop('click_rate',axis=1)
#separte the predicting attribute into Y for model training 
y = df['click_rate']

# importing train_df_split from sklearn
from sklearn.model_selection import train_test_split

#Split data into df and train
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

numeric=['subject_len', 'body_len', 'mean_paragraph_len','no_of_CTA',
         'mean_CTA_len']
sc = StandardScaler()
X_train[numeric]=sc.fit_transform(X_train[numeric])
X_test[numeric]=sc.transform(X_test[numeric])

import numpy as np

#Model developement
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

#Metrics to evaluate your model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

model = RandomForestRegressor()
#transforming target variable through quantile transformer
ttr = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
ttr.fit(X_train, y_train)
yhat = ttr.predict(X_test)
r2_score(y_test, yhat), mean_absolute_error(y_test, yhat), np.sqrt(mean_squared_error(y_test, yhat))


from sklearn.ensemble import GradientBoostingRegressor
def boost_models(x):
    regr_trans = TransformedTargetRegressor(regressor=x, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans.fit(X_train, y_train)
    yhat = regr_trans.predict(X_test)
    algoname= x.__class__.__name__
    return algoname, round(r2_score(y_test, yhat),3), round(mean_absolute_error(y_test, yhat),1), round(np.sqrt(mean_squared_error(y_test, yhat)),1)

import xgboost as xg
import lightgbm as lgbm

algo=[GradientBoostingRegressor(), lgbm.LGBMRegressor(), xg.XGBRFRegressor()]
score=[]
for a in algo:
    score.append(boost_models(a))
pd.DataFrame(score, columns=['Model', 'Score', 'MAE', 'RMSE'])



#Prediction on test data
test = pd.read_csv("D:/Hackathon_On_Email_Clicks/test_data.csv")

train_len = len(df)
test_len = len(test)

df = pd.concat([df,test],axis=0)
df.reset_index(drop=True,inplace=True)

train = df.iloc[:train_len,:]
test = df.iloc[train_len:,:]


from sklearn.preprocessing import StandardScaler

numeric=['subject_len', 'body_len', 'mean_paragraph_len','no_of_CTA',
         'mean_CTA_len']
sc = StandardScaler()
train[numeric]=sc.fit_transform(train[numeric])
test[numeric]=sc.transform(test[numeric])

#Split data into df and train
X = train.drop('click_rate',axis=1)
y = train['click_rate']

ttr.fit(X, y)


test = test.drop('click_rate',axis=1)

pred = ttr.predict(test)

submission = pd.DataFrame()
submission['id'] = test['campaign_id']
submission['click_target']=pred

submission.to_csv('D://Hackathon_On_Email_Clicks//my-submission.csv',index=False)





