import torch
import argparse
from torchvision.models import VisionTransformer
import os

def demo():
    parser = argparse.ArgumentParser(description='Demo script')
    parser.add_argument('--model', type=str, default="transformer")
    
    IMAGES_PATH = os.getenv('IMAGES_PATH')
    ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
    BATCH_SIZE= int(os.getenv('BATCH_SIZE'))
    EPOCHS = int(os.getenv('EPOCHS'))
    LR = float(os.getenv('LR'))
    PATCH_SIZE=int(os.getenv('PATCH_SIZE'))
    DROPOUT=float(os.getenv('DROPOUT'))
    ATTENTION_DROPOUT=float(os.getenv('ATTENTION_DROPOUT'))
    HIDDEN_DIM=int(os.getenv('HIDDEN_DIM'))
    MLP_DIM=int(os.getenv('MLP_DIM'))
    NUM_HEADS=int(os.getenv('NUM_HEADS'))
    NUM_LAYERS=int(os.getenv('NUM_LAYERS'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VisionTransformer(image_size=80, patch_size=PATCH_SIZE, num_layers=NUM_LAYERS, 
                                       num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, 
                                       mlp_dim=MLP_DIM, dropout=DROPOUT,
                                       attention_dropout=ATTENTION_DROPOUT,
                                       num_classes=2).to(device)
    model.load_state_dict(torch.load("results/transformer/saved_model.pt", weights_only=True))

    _, _, test_loader = get_dataloaders()
    img = next(iter(test_loader))

if __name__ == '__main__':
    # Your code here
    demo()