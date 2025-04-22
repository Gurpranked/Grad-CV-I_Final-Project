# Author: Gurpreet Singh
# Date: 4/21/2025
# Description: This file contains the baseline model layer and module definitions

import torch
import numpy as np
from torch import nn

# GoogLeNet Inception Module
def VanillaInception(nn.Module):
    # Output channels for each branch 
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(VanillaInception, self).__init__(**kwargs)
        # First branch
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)

        # Second branch
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)

        # Third branch
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        
        # Fourth Branch
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)
    
    # Forward pass, return concatenated tensor
    def forward(self, x):
        b1 = nn.functional.relu(self.b1_1(x))
        b2 = nn.functional.relu(self.b2_2(nn.functional.relu(self.b2_1(x))))
        b3 = nn.functional.relu(self.b3_2(nn.functional.relu(self.b3_1(x))))
        b4 = nn.functional.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)

def kNN(cloud, center, k):
    center = center.expand(cloud.shape)

    # L2 Distance
    dist = cloud.add( - center).pow(2).sum(dim=3).pow(0.5)

    # Get k nearest neighbors
    knn_indices = dist.topk(k, largest=False, sorted=False)[1]

    return cloud.gather(2, knn_indices.unsqueeze(-1).repeat(1,1,1,3))

def InceptionkNN(nn.Module):
    def __init__(self):
        super(InceptionkNN, self).__init__()

        # Stem Portion
        self.stem = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 80, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
        # Inception Module with Custom Aggreation (1)
        self.inception = VanillaInception()

    def forward(self, x):
        return self.stem(self.inception(x))
    