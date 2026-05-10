# File: core/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryWeightConv2d(nn.Conv2d):
    def forward(self, input):
        w_bin = torch.where(self.weight >= 0,
                            torch.tensor(1.0, device=self.weight.device),
                            torch.tensor(-1.0, device=self.weight.device))
        
        # Straight-Through Estimator (STE)
        w_bin = self.weight + (w_bin - self.weight).detach()
        return F.conv2d(input, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryActivation(nn.Module):
    def forward(self, x):
        out = torch.where(x >= 0, 
                          torch.tensor(1.0, device=x.device),
                          torch.tensor(-1.0, device=x.device))
        return x + (out - x).detach()