# libraries involved
from typing import Dict, Any
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_recall_fscore_support
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from FRA_PROJ import DATA_FRA, MODEL_FRA
from DATA_FRA import Data_PP, FeatureEngine
from MODEL_FRA import Model
import pickle

"""Model testing and performance analysis."""

# deserializing modules needed for testing
pickle.load(open('FRA_model.pkl', 'rb'))
pickle.load(open('FRA_PP.pkl', 'rb'))
pickle.load(open('Feat_Eng.pkl', 'rb'))


# Testing model for performance analysis
def test(ds: pd.DataFrame):
    # get testing data scheme
    y_true, x_val = Data_PP(ds, 'test')
    y_eval = Model.forward(x_val, F.relu)  # using relu activation for validating model for abrupt approach
    # converting results to type array
    y_eval= y_eval.detch().numpy()
    y_true= y_true.detach().numpy()
    # output test predicted val and true val
    return y_eval, y_true


# Get overall performance metrics.
def get_overall_metrics(y_true: np.ndarray, y_eval: np.ndarray) -> dict:
    metrics = precision_recall_fscore_support(y_true, y_eval, average="weighted")
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1_score": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics


# Get class Specific metrics
def class_metrics(y_true: np.ndarray, y_eval: np.ndarray,
                  class_to_index: dict) -> dict:
    class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_eval, average=None)
    for i, _class in enumerate(class_to_index):
        class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1_score": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    sorted_class_metrics = OrderedDict(sorted(class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True))
    return sorted_class_metrics


# Get accuracy and matrix classification of data
def classification_score(y_true: np.ndarray, y_eval: np.ndarray) -> dict[str, float | int | Any]:
    acc = accuracy_score(y_true, y_eval)
    mat = confusion_matrix(y_true, y_eval)
    res = {'accuracy': acc, 'relationship': mat, }

    return res


# serializing validation modules
pickle.dump(test, open('FRA_test.pkl', 'wb'))
pickle.dump(get_overall_metrics,open('FRA_om.pkl','wb'))
pickle.dump(class_metrics,open('FRA_cm.pkl','wb'))
pickle.dump(classification_score,open('FRA_cs.pkl','wb'))
