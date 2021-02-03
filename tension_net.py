import torch
import torch.nn as nn


class TensionNet(nn.Module):
    def __init__(self, input_size):
        super(TensionNet, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, input):
        output = self.linear(input)
        return output


