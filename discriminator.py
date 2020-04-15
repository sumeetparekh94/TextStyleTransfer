import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


class DiscriminaterModel(nn.Module):
    def __init__(self, input_size):
        super(DiscriminaterModel, self).__init__()
        self.fc_input_Discriminator = nn.Linear(input_size, 1)
        self.trainable_params = None
        
    def forward(self, x):
        x = F.dropout(self.fc_input_Discriminator(x), p=0.2)
        return torch.sigmoid(x)
