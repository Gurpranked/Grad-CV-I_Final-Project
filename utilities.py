# Author: Gurpreet Singh
# Date: 4/22/2025
# Description: Various utility functions

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the accuracy of the model on a given BATCH of data
    @args:
        y_true: The true labels of the batch
        y_pred: The predicted labels of the batch
    @returns:
       The accuracy of the model on the given batch of data
    """
    return (torch.eq(y_true, y_pred).sum().item()/len(y_pred))

def train_step(model: torch.nn.Module, loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, device: torch.device, 
               train_loader: torch.utils.data.DataLoader,
               use_tqdm=False) -> tuple[float, float]:
    """
    Trains the model for one epoch on the given data loader.
    @args:
        model: The model to be trained.
        loss_fn: The loss function to use for training.
        optimizer: The optimizer to use for training.
        device: The device on which to run the training (e.g. "cpu" or "cuda").
        dataloader: The data loader to use for training.
    @returns:
        A tuple containing:
            - The batch averaged training loss
            - The batch averaged training accuracy
    """
    model.train()
    train_loss = 0
    train_acc = 0
    
    # GT and Preds converted to float for loss function computations
    for batch, (images, labels) in tqdm(enumerate(train_loader), disable=not use_tqdm,
                                        desc="\tTraining: ", total=len(train_loader), unit="batches"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        # Logits for transformer implementation
        preds = model(images)

        # Compute the loss
        loss = loss_fn(preds, labels)
        
        # Accumulate the loss
        train_loss += loss.item()
        # Accumulate the accuracy
        train_acc += accuracy_fn(labels, preds.argmax(1))

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Step the optimizer
        optimizer.step()

    # Return Averaged training loss and accuracy
    return (train_loss / len(train_loader), train_acc / len(train_loader))

def val_step(model: torch.nn.Module, loss_fn: torch.nn.Module, 
             device: torch.device, 
             min_val_loss: float,
             val_loader: torch.utils.data.DataLoader,
             use_tqdm=False) -> tuple[float, float, float]:
    """
    Validate the model for one epoch on the given data loader.
    The model automatically saves if there is an improvement 
        in model's performance when compared to previous runs. 
    @args:
        model: The model to be validated
        loss_fn: THe loss function to be used for validation
        device: The device on which the model is running
        val_loader: The data loader containing the validation data
    @returns:
        A tuple containing:
        - The batch averaged loss
        - The batch averaged accuracy
    """

    val_loss, val_acc = 0, 0
    model.eval()

    # GT and Preds converted to float for loss function computations
    for batch, (images, labels) in tqdm(enumerate(val_loader), disable=not use_tqdm, 
                                        desc="\t\tValidating: ", unit="batches", total=len(val_loader)):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        # Logits for transformer implementation
        preds = model(images)

        # Accumulate loss
        val_loss += loss_fn(preds, labels).item()

        # Accumulate accuracy
        val_acc += accuracy_fn(labels, preds.argmax(1))

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    if val_loss < min_val_loss:
        print(f"\t\tValidation Loss Decreased ({min_val_loss:.6f}) --->{val_loss:.6f} \t Saving model")
        min_val_loss = val_loss
        MODEL_TYPE = os.getenv("MODEL_TYPE")
        # If model saving path doesn't exist, make it
        if (not os.path.exists('results/' + MODEL_TYPE)):
            os.mkdir('results/' + MODEL_TYPE + "/")
        
        # Serialize and save the model
        torch.save(model.state_dict(), "results/" + MODEL_TYPE +  "/saved_model.pt")
    
    return (val_loss, val_acc, min_val_loss)


def test_step(model: torch.nn.Module, loss_fn: torch.nn.Module, 
              device: torch.device, 
              test_loader: torch.utils.data.DataLoader,
              use_tqdm=True) -> pd.DataFrame:
    """
    Test the model for one epoch on the given data loader.
    @args:
        model: The model to be tested
        loss_fn: The loss function to be used for testing
        device: The device on which the model is running
        test_loader: The data loader containing the test data
    @returns:
        A Pandas Dataframe with two columns:
            - `test_loss` - Per batch test loss
            - `test_acc` - Per batch test accuracy
    """

    test_loss, test_acc = 0, 0
    idx = 0

    test_metrics = pd.DataFrame(columns=["test_loss", "test_acc"])

    model.eval()
    with torch.inference_mode():
        # GT and Preds converted to float for loss function computations
        for batch, (images, labels) in tqdm(enumerate(test_loader), disable=not use_tqdm,
                                            desc="Testing: ", total=len(test_loader), unit="batches"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            # Logits for transformer implementation
            preds = model(images)
            
            # Accumulate loss
            test_loss = loss_fn(preds, labels).item()

            # Accumulate accuracy
            test_acc = accuracy_fn(labels, preds.argmax(dim=1))

            test_metrics.loc[idx, "test_loss"] = test_loss
            test_metrics.loc[idx, "test_acc"] = test_acc
            
            idx += 1

    return test_metrics

def save_metrics(train_metrics: pd.DataFrame, val_metrics: pd.DataFrame, test_metrics: pd.DataFrame):
    """
    Saves the stored metrics from the training, validation, and test metric DataFrame objects.
    Metrics stored in 'results/MODEL_TYPE' with MODEL_TYPE as the configured model that was just trained.
    MODEL_TYPE is intended to be configured before training as an environment variable.

    Args:
        train_metrics (pd.DataFrame): Training metrics DataFrame object.
        val_metrics (pd.DataFrame): Validation metrics DataFrame object.
        test_metrics (pd.DataFrame): Test metrics DataFrame object.
    
    Returns:
        None

    """
    MODEL_TYPE = str(os.getenv("MODEL_TYPE"))

    # Save metrics to csv
    train_metrics.to_csv("results/" + MODEL_TYPE + "/train_metrics.csv", index=True)
    val_metrics.to_csv("results/" + MODEL_TYPE + "/val_metrics.csv", index=True)
    test_metrics.to_csv("results/"+ MODEL_TYPE + "/test_metrics.csv", index=True)

    # Save metric plots

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_title("Training and Validation Loss")
    axs[0].plot(train_metrics['train_loss'], label='Training Loss')
    axs[0].plot(val_metrics['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].plot(train_metrics['train_acc'], label='Training Accuracy')
    axs[1].plot(val_metrics['val_acc'], label='Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("results/" + MODEL_TYPE + "/train_val__loss_accuracy_plot.png")
    plt.close()

    fig, axs = plt.subplots(2, figsize=(8, 6))
    axs[0].set_title("Testing Loss")
    axs[0].plot(test_metrics['test_loss'], label='Test Loss')
    axs[0].set_xlabel('Batches')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].set_title("Testing Accuracy")
    axs[1].plot(test_metrics['test_acc'], label='Testing Accuracy')
    axs[1].set_xlabel('Batches')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("results/" + MODEL_TYPE + "/test_loss_accuracy_plot.png")
    plt.close()

