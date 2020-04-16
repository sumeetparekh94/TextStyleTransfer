import torch
import torch.nn as nn
import torch.nn.functional as F

class Generative_model_RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm_layers = list()
        # for i in range(len(hidden_dim)):
        #     if i == 0:
        #         lstm1 = nn.LSTM(embedding_dim, hidden_dim[i])
        #     else:
        #         lstm1 = nn.LSTM(hidden_dim[i - 1], hidden_dim[i])
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
            
        # The linear layer that maps from hidden state space to tag space
        self.generated = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, sentence):
        
        # embeds = self.word_embeddings(sentence)
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores
        embeds = self.word_embeddings(sentence)
        prev_out , out = None
        # for i in range(len(self.lstm_layers)):
        #     if (i == 0):
        #         out,_ = self.lstm_layers[i](embeds.view(len(sentence), 1, -1))
        #     else:
        #         out,_ = self.lstm_layers[i](prev_out)
        #     prev_out = out
        lstm_output = self.lstm(embeds.view(len(sentence), 1, -1))
        output_space = self.generated(out.view(len(sentence), -1))
        output_scores = F.gumbel_softmax(output_space, tau=0.5)
        return output_scores