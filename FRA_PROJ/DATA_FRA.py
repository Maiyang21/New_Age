# libraries/engines involved
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.decomposition import PCA
import torch
import pickle

" Defining data preprocessing module and Feature Engineering"


def FeatureEngine(space: pd.DataFrame, i=None):
    fs = StandardScaler().fit_transform(space)
    n_dim= len(list(space.columns))
    pcf = PCA(n_dim, random_state=50)
    pcf.fit_transform(fs)
    # features sorting based on significance on data
    feat_prob = list(pcf.explained_variance_)
    feat_dim = []
    [feat_dim.append(i) for i in space.columns]
    feat_dict = dict(zip(feat_prob, feat_dim))
    feat_srt_dict = dict(sorted(feat_dict.items(), key=lambda kv: kv[1], reverse=bool(1)))
    feat_sel = list(feat_srt_dict)[:10]
    # removing less significant features
    if i not in feat_sel:
        space.drop([i], axis=1)
    # returning data scheme
    return space, list(space.columns)


def Data_PP(data: pd.DataFrame, choice: str):
    global res

    RA = data.drop(['Country', 'State', 'City'], axis=1)

    # encoding ordinal data
    if choice == 'train' or choice == 'test':
        OE = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
        RA['Risk Rating'] = OE.fit_transform(RA['Risk Rating'].values.reshape(-1, 1).astype(str))
        TE = TargetEncoder()
        RA['Education Level'] = TE.fit_transform(RA['Education Level'].values.reshape(-1, 1).astype(str),
                                                 RA['Risk Rating'])
        RA['Employment Status'] = TE.fit_transform(RA['Employment Status'].values.reshape(-1, 1).astype(str),
                                                   RA['Risk Rating'])
        RA['Payment History'] = TE.fit_transform(RA['Payment History'].values.reshape(-1, 1).astype(str),
                                                 RA['Risk Rating'])
        # encoding nominal category
        RA = pd.get_dummies(RA, columns=['Gender', 'Marital Status', 'Loan Purpose'],
                            prefix=['Gender', 'Marital Status', 'Loan Purpose'])

        # encoding categorical intervals
        RA['Age'] = RA['Age'].astype(int)
        bin = [16, 24, 30, 40, 60, 70]
        val = [1, 2, 3, 4, 5]
        RA['Age'] = pd.cut(RA['Age'], bins=bin, labels=val)

        # Data Selection

        x = RA.drop(['Risk Rating'], axis=1)
        y = RA['Risk Rating']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=50)
        # selecting features
        x_train, x1_col = FeatureEngine(x_train)
        x_test, x2_col = FeatureEngine(x_test)
        # Tensor data
        x_train = torch.FloatTensor(x_train.values)
        x_test = torch.FloatTensor(x_test.values)
        y_train = torch.LongTensor(y_train.values)
        y_test = torch.LongTensor(y_test.values)
        # output choice branching from user
        if choice == 'train':
            res = y_train, x_train, x1_col
        if choice == 'test':
            res = y_test, x_test

    elif choice == 'predict':
        z = pd.get_dummies(data, columns=data.columns, prefix=data.columns) # since predict_api takes in reduced data
        z = torch.FloatTensor(z.values)
        res = z

    # tuple output for net model
    return res


# serializing preprocessing modules
pickle.dump(Data_PP, open('FRA_PP.pkl', 'wb'))
pickle.dump(FeatureEngine, open('Feat_Eng.pkl', 'wb'))
