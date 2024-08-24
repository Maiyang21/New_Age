#libraries involved
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

#loading the data from source
FRA = pd.read_csv("C:/Users/PC/Documents/DATABASE/financial_risk_assessment.csv")
FRA = FRA.drop(['Country', 'State', 'City','Gender','Marital Status','Loan Purpose'], axis=1)



# Data Cleaning and Preprocessing
def Data_PP(data: pd.DataFrame, choice: str):
    global y_train, x_train, y_test, x_test, res
    RA = data

    # encoding of sparse features
    RA['Age'] = RA['Age'].astype(int)
    bin = [16, 24, 30, 40, 60, 70]
    val = [1, 2, 3, 4, 5]
    RA['Age'] = pd.cut(RA['Age'], bins=bin, labels=val)

    # encoding ordinal category
    if choice == 'train' or choice == 'test':
        OE1 = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
        RA['Risk Rating'] = OE1.fit_transform(RA['Risk Rating'].values.reshape(-1, 1).astype(str))
        OE2 = OrdinalEncoder(categories=[['High School', "Bachelor's", "Master's", 'PhD']])
        RA['Education Level'] = OE2.fit_transform(RA['Education Level'].values.reshape(-1, 1).astype(str))
        OE3 = OrdinalEncoder(categories=[['Unemployed', 'Employed', 'Self-employed']])
        RA['Employment Status'] = OE3.fit_transform(RA['Employment Status'].values.reshape(-1, 1).astype(str))
        OE4 = OrdinalEncoder(categories=[['Poor', 'Fair', 'Good', 'Excellent']])
        RA['Payment History'] = OE4.fit_transform(RA['Payment History'].values.reshape(-1, 1).astype(str))



    elif choice == 'predict':
        RA = RA.drop(['Risk Rating'], axis=1)
        OE2 = OrdinalEncoder(categories=[['High School', "Bachelor's", "Master's", 'PhD']])
        RA['Education Level'] = OE2.fit_transform(RA['Education Level'].values.reshape(-1, 1).astype(str))
        OE3 = OrdinalEncoder(categories=[['Unemployed', 'Employed', 'Self-employed']])
        RA['Employment Status'] = OE3.fit_transform(RA['Employment Status'].values.reshape(-1, 1).astype(str))
        OE4 = OrdinalEncoder(categories=[['Poor', 'Fair', 'Good', 'Excellent']])
        RA['Payment History'] = OE4.fit_transform(RA['Payment History'].values.reshape(-1, 1).astype(str))


    # cleaning
    for i in RA.columns:
        RA = RA.dropna()
        RA[i] = RA[i].astype(float)



    # Data Selection
    if choice == 'train' or choice == 'test':
        x = RA.drop(['Risk Rating'], axis=1)
        y = RA['Risk Rating']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=50)
        # Tenderizing data
        x_train = torch.FloatTensor(x_train.values)
        x_test = torch.FloatTensor(x_test.values)
        y_train = torch.LongTensor(y_train.values)
        y_test = torch.LongTensor(y_test.values)
        if choice == 'train':
            res = y_train, x_train
        if choice == 'test':
            res = y_test, x_test

    elif choice == 'predict':
        x = torch.FloatTensor(RA.values)
        res = x

    return res


# Neural Network Architecture
class model(nn.Module):
    # neuron connections
    def __init__(self, input_l=13, h1=10, h2=8, h3=5, output_l=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_l, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output_l)

    # neuron functions
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.softmax(self.out(x), dim=-1)
        # output result
        return x


# Network train setup
torch.manual_seed(30)
Model = model()
epochs = 100
losses = []
optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Network Training
y_train, x_train = Data_PP(FRA, "train")
Model.forward(x_train)
for i in range(epochs):
    y_prd = Model.forward(x_train)
    loss = criterion(y_prd, y_train)
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

#Model validation
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = Model.forward(data)
        y_prd= torch.LongTensor(y_val.argmax().detach())
        #print(criterion(y_prd, y_test))
        print(f'{y_prd} \t {y_test[i]}')
print(y_test.unique())
print(confusion_matrix(y_test, y_prd))
"""

# Serializing model
pickle.dump(Model, open('Model.pkl', 'wb'))

# Serializing data format
pickle.dump(Data_PP, open('dataprep.pkl', 'wb'))
