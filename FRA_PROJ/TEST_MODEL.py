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
from FRA_PROJ import DATA_FRA, MODEL_FRA


"""Testing model after training """
def test_model(ds_test: pd.DataFrame, model: nn.Module, n_class: int, n_loss: torch.nn.modules.loss._WeightedLoss):
    target= F.one_hot(ds_test['target'], num_classes=n_class).float() #encoding target based on classes present
    ds_test= ds_test.drop(['target'], axis=1)
    with torch.no_grad():# turn off back_prop
        for i, data in enumerate(ds_test):
            y_eval= model.forward(data)
            print(f'loss:{n_loss(y_prd, target)}')
            y_prd= y_eval.argmax().detach().numpy() #persuading model to be decisive
            print(f'prediction:{y_prd} \t true:{target[i]}')
