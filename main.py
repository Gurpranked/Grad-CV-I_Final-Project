# Author: Gurpreet Singh
# Date: 5/5/2025
# Description: Driver Code for training and testing your choice of model

# NOTE: kNN & CNN and SVM were not implemented within this driver code, check `cnn.ipynb` for those implementations and results

import os
import torch
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from preprocess.data_process import get_dataloaders
from utilities import train_step, val_step, test_step, save_metrics
from timeit import default_timer as timer
from torchvision.models import VisionTransformer


def main():
    # Arugment parsers
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--model', type=str, required=True, help="Must be the following: 'transformer'")
    args = parsers.parse_args()

    # Validate arguments
    if args.model not in ['transformer']:
        parsers.print_help()
        exit(1)
    
    print("Friendly Reminder: All Hyperparameters are configured in the .env file. Please make sure to set all of them before running this script.")

    # Load environment variables
    load_dotenv()
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
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Set manual random seed
    torch.manual_seed(42)

    # Initialize metrics
    train_metrics = pd.DataFrame(columns=["train_loss", "train_acc"])
    val_metrics = pd.DataFrame(columns=["val_loss", "val_acc"])

    min_val_loss = np.inf

    # Load the model
    # TODO: Specify class names for each model after they are implemented
    model = VisionTransformer(image_size=80, patch_size=PATCH_SIZE, num_layers=NUM_LAYERS, 
                                       num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, 
                                       mlp_dim=MLP_DIM, dropout=DROPOUT,
                                       attention_dropout=ATTENTION_DROPOUT,
                                       num_classes=2) if args.model == "transformer" else None
    
    # Send model to device
    model = model.to(device)

    # Set MODEL_TYPE for use in saving metrics and saving best model
    os.environ["MODEL_TYPE"] = args.model

    # Dynamically set the loss function based on the model type 
    loss_fn = lambda output, labels: torch.nn.functional.binary_cross_entropy_with_logits(
        output, 
        torch.nn.functional.one_hot(labels, num_classes=2).float()) if args.model == "transformer" else None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
    
    train_start_time = timer()

    print("-----------Training starting----------")
    for epoch in tqdm(range(EPOCHS)):
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train the model
        train_loss, train_acc = train_step(model, loss_fn, optimizer, device, train_loader)
        train_metrics.loc[epoch, 'train_loss'] = train_loss
        train_metrics.loc[epoch, 'train_acc'] = train_acc

        # Validate the model
        val_loss, val_acc, min_val_loss = val_step(model, loss_fn, device, min_val_loss, val_loader)
        val_metrics.loc[epoch, 'val_loss'] = val_loss
        val_metrics.loc[epoch, 'val_acc'] = val_acc
    print("-----------Training finished----------")

    train_end_time = timer()

    total_train_time = train_end_time - train_start_time

    print(f"Training completed in {total_train_time:.2f} seconds")

    # Testing
    test_start_time = timer()

    print("-----------Testing starting----------")
    
    test_metrics = test_step(model, loss_fn, device, test_loader)

    print("-----------Testing finished----------")

    test_end_time = timer()

    total_test_time = test_end_time - test_start_time

    print(f"Testing completed in {total_test_time:.2f} seconds")

    MODEL_TYPE = os.getenv("MODEL_TYPE")
    print("Aggregating result metrics in results/" + MODEL_TYPE + ". Please wait...")
    
    save_metrics(train_metrics, val_metrics, test_metrics)

    print("Results saved!")


if __name__ == '__main__':
    # Your code here
    main()