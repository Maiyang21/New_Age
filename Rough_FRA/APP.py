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
from scratches.rough_FRA_ANN import Model, Data_PP

# initiating web app server
App = Flask(__name__, template_folder='template')

# exposing template to environment
site = Blueprint('site', __name__, template_folder='template')

# deserializing model objects
data_prp = pickle.load(open('dataprep.pkl', 'rb'))
n_model = pickle.load(open('Model.pkl', 'rb'))


# defining webapp home page directory
@App.route("/")
def home():
    return render_template('FRA_home.html')


# declaring request type from client to server
url = ('https://web.postman.co/workspace/My-Workspace~6b276ab1-9487-4b69-8499-ecbb9d89ca6a/request/37693541-12bc1852'
       '-10e6-45e3-8a0c-606c10b25676?tab=params')
payload = {'some': 'FRA_data'}
headers = {'content-type': 'application/json'}
requests.post(url, data=json.dumps(payload), headers=headers)


# defining the prediction APi
@App.route('/Predict_API', methods=['Post'])
def Predict_API():
    data = request.json['data']
    r_data = data
    val = np.array(list(r_data)).reshape(14, -1)
    t_data = pd.DataFrame(val)
    p_data = pd.DataFrame(t_data[1].values.reshape(1, -1), columns=t_data[0])
    n_data = data_prp(p_data, 'predict')
    out = n_model.forward(n_data)
    res = {"Risk Rating": str(out.argmax().detach())}
    return jsonify(res)


# defining the prediction page function
@App.route('/Predict', methods=['Get'])
def Predict():
    data = request.form.items()
    val = np.array(list(data)).reshape(14, -1)
    t_data = pd.DataFrame(val)
    p_data = pd.DataFrame(t_data[1].values.reshape(1, -1), columns=t_data[0])
    n_data = data_prp(p_data, 'predict')
    out = n_model.forward(n_data)
    return render_template('FRA_home.html', result= out.argmax().detach())


if __name__ == '__main__':
    App.run(debug=bool(1))
