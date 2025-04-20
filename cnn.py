import torch
import numpy as np
from torch.nn import Module, LazyConv2d, MaxPool2d, functional


# GoogLeNet Inception Module
def VanillaInception(Module):
    # Output channels for each branch 
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(VanillaInception, self).__init__(**kwargs)
        # First branch
        self.b1_1 = LazyConv2d(c1, kernel_size=1)

        # Second branch
        self.b2_1 = LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = LazyConv2d(c2[1], kernel_size=3, padding=1)

        # Third branch
        self.b3_1 = LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = LazyConv2d(c3[1], kernel_size=5, padding=2)
        
        # Fourth Branch
        self.b4_1 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = LazyConv2d(c4, kernel_size=1)
    
    # Forward pass, return concatenated tensor
    def forward(self, x):
        b1 = functional.relu(self.b1_1(x))
        b2 = functional.relu(self.b2_2(functional.relu(self.b2_1(x))))
        b3 = functional.relu(self.b3_2(functional.relu(self.b3_1(x))))
        b4 = functional.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)