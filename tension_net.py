import torch.nn as nn
import torch.nn.functional as F


class TensionNet(nn.Module):
    def __init__(self, input_size):
        super(TensionNet, self).__init__()
        self.input_size =input_size
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input):
        output = self.linear(input)
        return output


class TensionNet1(nn.Module):
    def __init__(self, input_size, activation_f=F.relu, hidden_size=16):
        super(TensionNet1, self).__init__()
        self.input_size = input_size
        self.activation_f = activation_f
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        output = self.activation_f(self.linear1(input))
        output = self.linear2(output)
        return output


class TensionNet2(nn.Module):
    def __init__(self, input_size, activation_f=F.relu, hidden_size=(16, 16)):
        super(TensionNet2, self).__init__()
        self.input_size = input_size
        self.activation_f = activation_f
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)

    def forward(self, input):
        output = self.activation_f(self.linear1(input))
        output = self.activation_f(self.linear2(output))
        output = self.linear3(output)
        return output


class TensionNet3(nn.Module):
    def __init__(self, input_size, activation_f=F.relu, hidden_size=(16, 64, 16)):
        super(TensionNet3, self).__init__()
        self.input_size = input_size
        self.activation_f = activation_f
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.linear4 = nn.Linear(hidden_size[2], 1)

    def forward(self, input):
        output = self.activation_f(self.linear1(input))
        output = self.activation_f(self.linear2(output))
        output = self.activation_f(self.linear3(output))
        output = self.linear4(output)
        return output


class TensionNet4(nn.Module):
    def __init__(self, input_size, activation_f=F.relu,
                 hidden_size=(64, 128, 128, 64)):
        super(TensionNet4, self).__init__()
        self.input_size = input_size
        self.activation_f = activation_f
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.linear4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.linear5 = nn.Linear(hidden_size[3], 1)

    def forward(self, input):
        output = self.activation_f(self.linear1(input))
        output = self.activation_f(self.linear2(output))
        output = self.activation_f(self.linear3(output))
        output = self.activation_f(self.linear4(output))
        output = self.linear5(output)
        return output
