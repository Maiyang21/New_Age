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


"""functional model training"""
def train(
        ds: pd.DataFrame,
        epoch: int,
        model: nn.Module,
        num_classes: int,
        loss_fn: torch.nn.modules.loss._WeightedLoss,
        optimizer: torch.optim.Optimizer,
) -> float:

    model.train() #initiating training procedure
    loss = 0.0
    for i in range(epoch):
        optimizer.zero_grad()  # reset gradients
        y_eval = model(ds)  # forward pass
        targets = F.one_hot(ds["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
        m_loss = loss_fn(y_eval, targets)  # define loss
        m_loss.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (m_loss.detach().item() - loss) / (i + 1)  # cumulative loss
    torch.save(model.state_dict(), 'Trained_params.pt') #saving adjusted weights and bias
    return loss

