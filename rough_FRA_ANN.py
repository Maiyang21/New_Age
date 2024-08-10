#libraries involved
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder,TargetEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

#Data Cleaning and Preprocessing
FRA= pd.read_csv("C:/Users/PC/Documents/DATABASE/financial_risk_assessment.csv")
RA= FRA.drop(['City', 'State', 'Country'], axis=1)
#encoding nominal category
RA= pd.get_dummies(RA, columns=['Gender', 'Marital Status', 'Loan Purpose'], prefix=['Gender', 'Marital Status', 'Loan Purpose'])
#encoding ordinal category
OE= OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
RA['Risk Rating']= OE.fit_transform(RA['Risk Rating'].values.reshape(-1,1).astype(str))
TE= TargetEncoder()
RA['Education Level']= TE.fit_transform(RA['Education Level'].values.reshape(-1,1).astype(str), RA['Risk Rating'])
RA['Employment Status']= TE.fit_transform(RA['Employment Status'].values.reshape(-1,1).astype(str), RA['Risk Rating'])
RA['Payment History']= TE.fit_transform(RA['Payment History'].values.reshape(-1,1).astype(str), RA['Risk Rating'])
#encoding of sparse features
bin= [16,24,30,40,60,70]
val=[1,2,3,4,5]
RA['Age']= pd.cut(RA['Age'], bins=bin, labels=val)
#cleaning
for i in RA.columns:
    RA[i]= RA[i].astype(float)
RA= RA.fillna(0)
"""
#unit checking data
j=[]
for i in RA.columns:
    j.append(i)
print(len(j))"""

#Data Selection
x=RA.drop(['Risk Rating'], axis=1)
y= RA['Risk Rating']
x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.75, random_state=50)
#Tensorising data
x_train= torch.FloatTensor(x_train.values)
x_test= torch.FloatTensor(x_test.values)
y_train= torch.LongTensor(y_train.values)
y_test= torch.LongTensor(y_test.values)

#Neural Network Architecture
class model(nn.Module):
    #neuron connections
    def __init__(self, input_l=24, h1=20, h2=15, h3=8, output_l=3) -> None:
        super().__init__()
        self.fc1= nn.Linear(input_l, h1)
        self.fc2= nn.Linear(h1, h2)
        self.fc3= nn.Linear(h2, h3)
        self.out= nn.Linear(h3, output_l)

    #neuron functions
    def forward(self,x):
        x= F.tanh(self.fc1(x))
        x= F.tanh(self.fc2(x))
        x= F.tanh(self.fc3(x))
        x= F.softmax(self.out(x), dim=-1)
        # output result
        return x

#Network train setup
torch.manual_seed(40)
Model= model()
epochs=100
losses=[]
optimizer= torch.optim.Adam(Model.parameters(), lr=0.01)
criterion= nn.CrossEntropyLoss()

#Network Training
Model.forward(x_train)
for i in range(epochs):
    y_prd= Model.forward(x_train)
    loss= criterion(y_prd, y_train)
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
#Training Result
for i in range(epochs):
    if i%10 == 0:
        print(i,losses[i])

import matplotlib.pyplot as plt
plt.plot(range(epochs), losses)
plt.xlabel('epochs')
plt.ylabel('losses')
plt.show()
"""
#Model validation
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val= Model.forward(data)
        print(criterion(y_val, y_test))
        print(f'{y_val} \t {y_test[i]}')
print(y_test.unique())
