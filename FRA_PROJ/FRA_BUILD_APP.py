# libraries involved
import flask
from flask import Flask, app, render_template, request, jsonify, Blueprint
import requests, json, urllib
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
from scratches import MODEL_FRA, DATA_FRA, TRAIN_MODEL_FRA, VALIDATE_MODEL_FRA

"""Web App build setup"""

# initializing the web app server
App = Flask(__name__, template_folder='home.html')

# exposing template to environment
site = Blueprint('site', __name__, template_folder='template')

# deserializing project modules
model = pickle.load(open('FRA_model.pkl', 'rb'))
data_fra = pickle.load(open('FRA_PP.pkl', 'rb'))
feat_eng = pickle.load(open('Feat_Eng.pkl', 'rb'))
model_t = pickle.load(open('FRA_train.pkl', 'rb'))
model_v = pickle.load(open('FRA_test.pkl', 'rb'))
model_cm = pickle.load(open('FRA_cm.pkl', 'rb'))
model_cs = pickle.load(open('FRA_cs.pkl', 'rb'))
model_ovm = pickle.load(open('FRA_om.pkl', 'rb'))


# defining home directory
@App.route('/', methods=['Post'])
def home():
    return render_template('home.html')


# declaring request type from client to server
url = ('https://web.postman.co/workspace/My-Workspace~6b276ab1-9487-4b69-8499-ecbb9d89ca6a/request/37693541-12bc1852'
       '-10e6-45e3-8a0c-606c10b25676?tab=params')
payload = {'some': 'FRA_data'}
headers = {'content-type': 'application/json'}
requests.post(url, data=json.dumps(payload), headers=headers)


# api test for training model
@App.route('/Train_API', methods=['Post'])
def Train_API():
    global col # making the reduced features available for prediction feature reduction
    data = request.json['FRA_data']
    val = np.array(list(data)).reshape(20, -1)
    t_data = pd.DataFrame(val)
    p_data = pd.DataFrame(t_data[1].values.reshape(1, -1), columns=t_data[0])
    out, col = model_t(p_data, 100, F.tanh, 3, torch.optim.rmsprop)
    return jsonify(str(out))


# api test for validating model
@App.route('/Validate_API', methods=['Post'])
def Validate_API():
    data = request.json['FRA_data']
    val = np.array(list(data)).reshape(20, -1)
    t_data = pd.DataFrame(val)
    p_data = pd.DataFrame(t_data[1].values.reshape(1, -1), columns=t_data[0])
    y_v, y_t = model_v(p_data)
    out_cm = model_cm(y_v, y_t)
    out_cs = model_cs(y_v, y_t)
    out_om = model_ovm(y_v, y_t)
    return jsonify(f'{out_cm} /t {out_cs} /t {out_om}')


# api test for model prediction
@App.route('/Predict_API', methods=['Post'])
def Predict_API():
    data = request.json['FRA_data']
    val = np.array(list(data)).reshape(20, -1)
    t_data = pd.DataFrame(val)
    p_data = pd.DataFrame(t_data[1].values.reshape(1, -1), columns=t_data[0])
    for i in p_data.columns:
        if i not in col: # carrying out prediction feature reduction
            p_data.drop([i], axis=1)
    val = data_fra(p_data, 'predict')
    out = model.forward(val)
    res = {"Risk Rating": str(out.argmax().detach())}
    return jsonify(res)


if __name__ == '__main__':
    App.run(debug=bool(1))
