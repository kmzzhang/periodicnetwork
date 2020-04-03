import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.tcn_util import Chomp1d


class Classifier(nn.Module):
    def __init__(self, num_inputs, num_channels, num_class, hidden, dropout=0, kernel_size=2,
                 dropout_classifier=0, aux=0):

        super(type(self), self).__init__()
        self.aux = aux
        self.TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.linear1 = nn.Linear(num_channels[-1] + aux, hidden)
        self.linear2 = nn.Linear(hidden, num_class)
        self.dropout = nn.Dropout(dropout_classifier)
        if aux > 0:
            self.linear = nn.Sequential(self.linear1, self.dropout, nn.ReLU(), self.linear2)
        else:
            self.linear = nn.Linear(num_channels[-1], num_class)

    def forward(self, x, aux=None):
        feature = self.TCN(x)[:, :, -1]
        if self.aux > 0:
            feature = torch.cat((feature, aux), dim=1)
        logprob = self.linear(feature)
        return logprob


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dilation, padding, kernel_size=2, dropout=0.2):
        super(type(self), self).__init__()
        layers = []
        for i in range(2):
            prev = n_inputs if i == 0 else n_outputs
            layers += [weight_norm(nn.Conv1d(prev, n_outputs, kernel_size=kernel_size,
                                             dilation=dilation, padding=padding))]
            layers[-1].weight.data.normal_(0, 0.01)
            layers += [Chomp1d(padding)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(type(self), self).__init__()
        layers = []
        block = TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [block(in_channels, out_channels, dilation=dilation_size, kernel_size=kernel_size,
                             padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
