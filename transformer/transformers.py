# Author: Gurpreet Singh
# Date: 4/23/2025
# Description: Vision Transformer definition

from torchvision.models import vit_b_32
import torch
class ViT():
    def __init__(self, num_classes: int, input_dim: tuple):
        super().__init__()
        self.model = vit_b_32()
        self.num_classes = num_classes
        self.input_dim = input_dim