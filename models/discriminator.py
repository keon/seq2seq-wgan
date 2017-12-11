import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(Discriminator, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embed = nn.Linear(vocab_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, context):
        """
            input: I x B x Vocab
            hidden: I x B x H
            context: I x B x E
        """
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input)  # (I,B,E)
        embedded = self.dropout(embedded)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, None)
        out = self.out(output[-1])  # [b, h] -> [b, 1]
        return out
