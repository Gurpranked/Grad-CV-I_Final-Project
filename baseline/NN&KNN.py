# Author: Gurpreet Singh
# Date: 4/21/2025
# Description: This file contains the baseline model layer and module definitions

import torch
import numpy as np
from torch import nn

# GoogLeNet Inception Module
class VanillaInception(nn.Module):
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

# Define a kNN module in PyTorch
class kNN(nn.Module):
    def __init__(self, k, **kwargs):
        super(kNN, self).__init__(**kwargs)
        self.k = k

    # Forward pass, find the k nearest neighbors and return their average feature vector
    def forward(self, x):
        dists = torch.cdist(x, x).sort(dim=1)[0]
        idxs = (dists.argsort(dim=1))[:, 1:self.k+1].view(-1)
        x_avg = torch.zeros_like(x).float()
        for i in range(idxs.shape[0]):
            x_avg[i] = torch.mean(x[idxs[i]], dim=0, keepdim=True)
        return x_avg

class InceptionkNN(nn.Module):
    def __init__(self, k: int = 2):
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

        # kNN
        self.knn = kNN(2)

    # Compute forward pass
    # Step -> inception module -> kNN -> Averaged feature vector
    def forward(self, x):
        return self.stem(self.inception(self.knn(x)))
    