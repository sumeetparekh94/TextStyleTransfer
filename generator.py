import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


class GenerativeModel(nn.Module):
    def __init__(self, dimension):
        super(GenerativeModel, self).__init__()
        self.fc_input = nn.Linear(dimension, 1024)
        self.fc_hidden1 = nn.Linear(1024, 1024)
        self.fc_output = nn.Linear(1024, dimension)

    def forward(self, x):
        x = F.relu(self.fc_input(x))
        x = F.relu(self.fc_hidden1(x))
        return torch.tanh(self.fc_output(x))