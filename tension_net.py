import torch.nn as nn
import torch.nn.functional as F


class TensionNet(nn.Module):
    def __init__(self, input_size):
        super(TensionNet, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input):
        output = self.linear(input)
        return output


class TensionNet1(nn.Module):
    def __init__(self, input_size, activation_f=F.relu):
        super(TensionNet1, self).__init__()
        self.activation_f = activation_f
        self.linear1 = nn.Linear(input_size, 16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, input):
        output = self.activation_f(self.linear1(input))
        output = self.linear2(output)
        return output
