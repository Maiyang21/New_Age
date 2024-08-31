# libraries involved
from typing import Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from FRA_PROJ import DATA_FRA, MODEL_FRA
import pickle
from DATA_FRA import FeatureEngine, Data_PP
from MODEL_FRA import Model

"""functional model training"""
# deserializing modules for preprocessing and training
pickle.load(open('Feat_Eng.pkl', 'rb'))
pickle.load(open('FRA_model.pkl', 'rb'))
pickle.load(open('FRA_PP.pkl', 'rb'))


def train(
        ds: pd.DataFrame,
        epoch: int,
        act_func: torch.nn.functional,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
) -> tuple[float, Any]:
    target, trainer, col = Data_PP(ds, 'train')
    # initiating training procedure
    loss = 0.0
    for i in range(epoch):
        optimizer.zero_grad()  # reset gradients
        y_eval = Model.forward(trainer, act_func)  # forward pass
        target = F.one_hot(target, num_classes=num_classes).float()  # one-hot (for loss_fn)
        m_loss = nn.CrossEntropyLoss(y_eval, target)  # define loss
        m_loss.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (m_loss.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss , col


# serializing the train module
pickle.dump(train, open('FRA_train.pkl', 'wb'))
