#libraries/engines involved
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder,TargetEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Data Cleaning and Preprocessing"""
FRA= pd.read_csv("C:/Users/PC/Documents/DATABASE/financial_risk_assessment.csv")
RA= FRA.drop(['City', 'State', 'Country'], axis=1)#manual selecting features to be used by model


#encoding nominal category
RA= pd.get_dummies(RA, columns=['Gender', 'Marital Status', 'Loan Purpose'], prefix=['Gender', 'Marital Status', 'Loan Purpose'])


#encoding ordinal category
OE= OrdinalEncoder(categories=[['Low', 'Medium', 'High']])#target is an ordinal data
RA['Risk Rating']= OE.fit_transform(RA['Risk Rating'].values.reshape(-1,1).astype(str))
TE= TargetEncoder() #to remove ant form of bias towrds target
RA['Education Level']= TE.fit_transform(RA['Education Level'].values.reshape(-1,1).astype(str), RA['Risk Rating'])
RA['Employment Status']= TE.fit_transform(RA['Employment Status'].values.reshape(-1,1).astype(str), RA['Risk Rating'])
RA['Payment History']= TE.fit_transform(RA['Payment History'].values.reshape(-1,1).astype(str), RA['Risk Rating'])


#encoding of sparse features(interval data)
bin= [16,24,30,40,60,70]
val=[1,2,3,4,5]
RA['Age']= pd.cut(RA['Age'], bins=bin, labels=val)


#cleaning
for i in RA.columns:
    RA[i]= RA[i].astype(float)
RA= RA.fillna(0)

#Data Selection for training and testing
x=RA.drop(['Risk Rating'], axis=1)
y= RA['Risk Rating']
x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.75, random_state=50)


#Tensorising data for gpu and torch framework
x_train= torch.FloatTensor(x_train.values)
x_test= torch.FloatTensor(x_test.values)
y_train= torch.LongTensor(y_train.values)
y_test= torch.LongTensor(y_test.values)

#data packaging for training and testing modules
train_frame= pd.DataFrame(['x_train', 'y_train'])
test_frame= pd.DataFrame(['x_test', 'y_test'])