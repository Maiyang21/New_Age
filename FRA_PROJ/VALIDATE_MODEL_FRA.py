#libraries involved
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


def get_overall_metrics(y_true: np.ndarray, y_eval: np.ndarray) -> dict:
    """Get overall performance metrics."""
    metrics = precision_recall_fscore_support(y_true, y_eval, average="weighted")
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1_score": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics


def class_metrics(y_true: np.ndarray, y_eval: np.ndarray,
                  class_to_index: dict) -> dict:
    """Get class Specific metrics"""
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


def classification_score(y_true: np.ndarray, y_eval: np.ndarray) -> float:
    """Get accuracy and matrix classification of data"""
    acc = accuracy_score(y_true, y_eval)
    mat = confusion_matrix(y_true, y_eval)
    res = {'accuracy': acc, 'relationship': mat, }

    return res
