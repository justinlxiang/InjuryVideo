import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class AvgPool(nn.Module):
    def __init__(self, inp, classes):
        super(AvgPool, self).__init__()
        self.classes = classes

        self.dropout = nn.Dropout(0.7)
        self.cls = nn.Conv3d(inp, classes, (1,1,1))
        
    def forward(self, inp):
        inp = torch.mean(torch.mean(torch.mean(inp, dim=2, keepdim=True), dim=3, keepdim=True), dim=4, keepdim=True)
        inp = self.dropout(inp)
        return self.cls(inp).squeeze()
