import torch
import torch.nn as nn
rnns = {'LSTM': nn.LSTM,
        'GRU': nn.GRU}


class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_rnn, num_layers, num_class, hidden, dropout_rnn=0.15, dropout=0,
                 bidirectional=False, rnn='GRU', aux=0):
        """
        RNN classifier

        Parameters
        ----------
        num_inputs: int
            dimension of input seqeunce
        hidden_rnn: int
            hidden dimension of RNN
        num_layers: int
            number of RNN layers
        num_class: int
            number of classes
        hidden: int
            hidden dimension of the two-layer MLP classifier
        dropout_rnn: float
            RNN dropout rate
        dropout: float
            MLP dropout rate
        bidirectional: bool
            Bidirectional RNN
        rnn: str
            "GRU" or "LSTM"
        aux: int
            Number of auxiliary inputs
        """
        super(type(self), self).__init__()
        self.aux = aux
        network = rnns[rnn]
        self.rnn = rnn
        self.encoder = network(input_size=num_inputs, hidden_size=hidden_rnn,
                               num_layers=num_layers, dropout=dropout_rnn)
        if bidirectional:
            hidden_rnn *= 2
        if aux > 0:
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(hidden_rnn + aux, hidden)
            self.linear2 = nn.Linear(hidden, num_class)
            self.relu = nn.ReLU()
            self.linear = nn.Sequential(self.linear1, self.dropout, self.relu, self.linear2)
        else:
            self.linear = nn.Linear(hidden_rnn, num_class)

    def forward(self, x, aux=None):
        # N, C, L --> L, N, H0
        x = x.permute(2, 0, 1)
        code = self.encoder(x)[1] if self.rnn == 'GRU' else self.encoder(x)[1][0]
        feature = code[-1]
        if self.aux > 0:
            feature = torch.cat((feature, aux), dim=1)
        logprob = self.linear(feature)
        return logprob
