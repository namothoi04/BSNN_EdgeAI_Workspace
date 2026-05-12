# File: core/models.py
import torch
import torch.nn as nn
from core.layers import BinaryWeightConv2d, BinaryActivation

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BNN(nn.Module):
    def __init__(self, activation_type="binary"):
        super(BNN, self).__init__()
        
        self.conv1 = BinaryWeightConv2d(1, 32, 3)
        # self.batnorm =  nn.BatchNorm2d(32)

        if activation_type == "binary":
            self.act = BinaryActivation()
        elif activation_type == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Chỉ hỗ trợ 'binary' hoặc 'relu'")
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        # x = self.batnorm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
