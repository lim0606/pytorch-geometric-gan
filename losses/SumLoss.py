from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class SumLoss(nn.Module):
    def __init__(self, sign=1.0):
        super(SumLoss, self).__init__()
        self.sign = sign

    def forward(self, input, target=0):
        output = torch.mul(input, self.sign)
        if input.dim() == 4:
            output = output.view(input.size(0),
                                 input.size(1) * input.size(2) * input.size(3))
        elif input.dim() == 3:
            output = output.view(input.size(0),
                                 input.size(1) * input.size(2))
        output = output.mean(0)
        return output.view(1)
