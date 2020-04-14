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

# import torch 
# from torch import nn
# from torch.autograd import Variable

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)

#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
def get_lstm():
  # layer1
  # input_dim=10, output_dim=20
  # layers = []
  # layers.append(nn.LSTM(1500, 1000, 1))
  # layers.append(nn.LSTM(1000, 500, 1))
  # layers.append(nn.LSTM(500, 1000, 1))
  # layers.append(nn.LSTM(1000, 1500, 1))

  # net = nn.Sequential(*layers)
  # print(net)
  # input = Variable(torch.randn(5, 3, 1500))
  # output1= net.forward(input)
  # print(output1)
  rnn1 = nn.LSTM(10, 20, 1)
  input = Variable(torch.randn(5, 3, 10))
  output1, hn = rnn1(input)

  # layer2
  # input_dim=20 output_dim=30
  rnn2 = nn.LSTM(20, 30, 1)
  output2, hn2 = rnn2(output1) 

# if __name__ == '__main__':
#   get_lstm()