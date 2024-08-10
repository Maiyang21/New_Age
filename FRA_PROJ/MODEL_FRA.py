#libraries/engines/frameworks involved
import torch
import torch.nn as nn
import torch.nn.functional as F

"""ANN Model architecture"""
class model(nn.Module):
    #network topology
    def __init__(self, input_l:int, h1:int, h2: int, h3: int, output_l: int) -> None:
        super().__init__()
        self.fc1= nn.Linear(input_l, h1)
        self.fc2= nn.Linear(h1, h2)
        self.fc3= nn.Linear(h2, h3)
        self.out= nn.Linear(h3, output_l)

    #network behaviour
    def forward(self, x, act_f: torch.nn.functional):
        x = act_f(self.fc1(x))
        x = act_f(self.fc2(x))
        x = act_f(self.fc3(x))
        # tanh--->relu--->elu in that order of training
        # to ensure steady growth of model
        x= F.softmax(self.out(x), dim=-1)
        #softmax used to allow bias influence on training model
        # output result
        return x
