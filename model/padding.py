import torch.nn as nn
import torch


class wrap(nn.Module):
    def __init__(self, wrap_length, mode='cyclic'):
        super(type(self), self).__init__()
        self.wrap_length = wrap_length
        self.mode = mode

    def forward(self, x):
        if self.mode == 'zero':
            return torch.cat([torch.zeros(x.shape[0], x.shape[1], self.wrap_length).type_as(x), x], dim=2)
        else:
            n = int(self.wrap_length / x.shape[2]) + 1
            mod = self.wrap_length % x.shape[2]
            if mod == 0:
                x0 = torch.cat([x] * n, dim=2)
            else:
                x0 = torch.cat([x[:, :, -mod:]] + [x] * n, dim=2)
        return x0

