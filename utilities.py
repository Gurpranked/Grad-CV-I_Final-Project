# Author: Gurpreet Singh
# Date: 4/22/2025
# Description: Various utility functions

import torch
import pandas as pd
from tqdm import tqdm

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
               train_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
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
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        pred_labels = model(images).argmax(dim=1)

        # Compute the loss
        loss = loss_fn(pred_labels, labels)
        
        # Accumulate the loss
        train_loss += loss.item()
        # Accumulate the accuracy
        train_acc += accuracy_fn(labels, pred_labels)

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
             val_loader: torch.utils.data.DataLoader) -> tuple[float, float, float]:
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
    for batch, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device).float()

        # Forward pass
        pred_labels = model(images)

        # Accumulate loss
        val_loss += loss_fn(pred_labels, labels).item()

        # Accumulate accuracy
        val_acc += accuracy_fn(labels, pred_labels.argmax(dim=1))

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    if val_loss > min_val_loss:
        print(f"Validation Loss Decreased ({min_val_loss:.6f}) --->{val_loss:.6f} \t Saving model")
        min_val_loss = val_loss
        torch.save(model.state_dict(), "saved_model.pth")
    
    return (val_loss, val_acc, min_val_loss)


def test_step(model: torch.nn.Module, loss_fn: torch.nn.Module, 
              device: torch.device, 
              test_loader: torch.utils.data.DataLoader) -> pd.DataFrame:
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

    test_metrics = pd.DataFrame(columns=["test_loss", "test_acc"])

    model.eval()
    with torch.inference_mode():
        # GT and Preds converted to float for loss function computations
        for batch, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float()
            
            # Forward pass
            pred_labels = model(images)
            
            # Accumulate loss
            test_loss += loss_fn(pred_labels, labels)

            # Accumulate accuracy
            test_acc += accuracy_fn(labels, pred_labels.argmax(dim=1))

            test_metrics["test_loss"].append(test_loss)
            test_metrics["test_acc"].append(test_acc)

    return test_metrics



