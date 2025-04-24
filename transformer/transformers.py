# Author: Gurpreet Singh
# Date: 4/23/2025
# Description: Vision Transformer definition

import torch
import os
from torchvision.models import VisionTransformer
from dotenv import load_dotenv

load_dotenv()
PATCH_SIZE=int(os.getenv('PATCH_SIZE'))
DROPOUT=float(os.getenv('DROPOUT'))
ATTENTION_DROPOUT=float(os.getenv('ATTENTION_DROPOUT'))
HIDDEN_DIM=int(os.getenv('HIDDEN_DIM'))
MLP_DIM=int(os.getenv('MLP_DIM'))
NUM_HEADS=int(os.getenv('NUM_HEADS'))
NUM_LAYERS=int(os.getenv('NUM_LAYERS'))

class ShipsVisionTransformer():
    """
    Vision Transformer for Ship Detection Model.
    """
    def __init__(self):
        """
        Initializes the model with pre-specified parameters from .env file
        """
        super().__init__()
        self.model = VisionTransformer(image_size=80, patch_size=PATCH_SIZE, num_layers=NUM_LAYERS, 
                                       num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, 
                                       mlp_dim=MLP_DIM, dropout=DROPOUT,
                                       attention_dropout=ATTENTION_DROPOUT,
                                       num_classes=2)
    
    # Model Information
    def info(self):
        print(self.model)
    
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)