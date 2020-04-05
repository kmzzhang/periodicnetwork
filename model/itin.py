import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.padding import wrap


class Classifier(nn.Module):
    def __init__(self, num_inputs, kernel_sizes, num_channels, num_class, hidden, dropout=0,
                 dropout_classifier=0, aux=0, padding='cyclic'):
        """
        Cyclic-permutation invariant Temporal Inception Network Classifier

        Parameters
        ----------
        num_inputs: int
            dimension of input seqeunce
        kernel_sizes: list
            list of kernel sizes used in the inception module
        num_channels: list
            list of hidden dimensions for each layer.
            len must be equal to hidden
        num_class: int
            number of classes
        hidden: int
            hidden dimension of the final two layer MLP classifier
        dropout: float
            dropout rate
        dropout_classifier: float
            dropout rate for the final MLP classifier
        aux: int
            number of auxiliary inputs
        padding: str
            "cyclic": symmetry padding for invariance
            "zero": zero padding for ordinary Cartesian network
        """
        super(type(self), self).__init__()
        self.aux = aux
        self.TCN = CyclicTemporalConvNet(num_inputs, num_channels, kernel_sizes, dropout, padding=padding)
        self.linear1 = nn.Conv1d(num_channels[-1] + aux, hidden, 1)
        self.linear2 = nn.Conv1d(hidden, num_class, 1)
        self.dropout = nn.Dropout(dropout_classifier)
        if aux > 0:
            self.linear = nn.Sequential(self.linear1, self.dropout, nn.ReLU(), self.linear2)
        else:
            self.linear = nn.Conv1d(num_channels[-1], num_class, 1)
        self.padding = padding

    def forward(self, x, aux=None):
        feature = self.TCN(x)
        self.cache = feature
        if self.aux > 0:
            feature = torch.cat((feature, aux[:, :, None].expand(-1, -1, feature.shape[2])), dim=1)
        logprob_ = self.linear(feature)
        if self.padding == 'cyclic':
            logprob = logprob_.mean(dim=2)
        if self.padding == 'zero':
            logprob = logprob_[:, :, -1]
        return logprob


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dilation, kernel_sizes, dropout=0.2, padding='cyclic'):
        super(type(self), self).__init__()
        layers = list()
        for i in range(2):
            prev = n_inputs if i == 0 else n_outputs
            layers += [InceptionBlock(prev, n_outputs, kernel_sizes=kernel_sizes,
                                      dilation=dilation, dropout=dropout, mode=padding)]
        self.net = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sizes, dilation, dropout=0.0, mode='cyclic'):
        super(type(self), self).__init__()
        self.net = nn.ModuleList()
        for kernel in kernel_sizes:
            padding = (kernel - 1) * dilation
            net = [wrap(padding, mode),
                   weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size=kernel, dilation=dilation)),
                   nn.ReLU(),
                   nn.Dropout(dropout)]
            net[-3].weight.data.normal_(0, 0.01)
            self.net.append(nn.Sequential(*net))
        self.downsample = weight_norm(nn.Conv1d(out_channel * len(kernel_sizes), out_channel, kernel_size=1))
        self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        outputs = []
        for net in self.net:
            outputs.append(net(x))
        out = torch.cat(outputs, dim=1)
        return self.downsample(out)


class CyclicTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dropout=0.2, padding='cyclic'):
        super(type(self), self).__init__()
        layers = []
        block = TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [block(in_channels, out_channels, dilation=dilation_size,
                             kernel_sizes=kernel_sizes, dropout=dropout, padding=padding)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
