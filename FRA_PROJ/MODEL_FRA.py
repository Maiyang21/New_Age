# libraries/engines/frameworks involved
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

"""ANN Model architecture"""


class model(nn.Module):
    # neuron connections
    def __init__(self, input_l=10, h1=8, h2=6, h3=5, output_l=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_l, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output_l)

    # neuron functions and behaviours
    def forward(self, x, act_f: torch.nn.functional):
        x = act_f(self.fc1(x))
        x = act_f(self.fc2(x))
        x = act_f(self.fc3(x))
        x = F.softmax(self.out(x), dim=-1)
        # output result
        return x


# serializing model
torch.manual_seed(50) # locking into a random state
Model = model()
pickle.dump(Model, open('FRA_model.pkl', 'wb'))