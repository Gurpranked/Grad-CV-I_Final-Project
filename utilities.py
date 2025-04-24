# Author: Gurpreet Sing
# Date: 4/22/2025
# Description: Various utility functions
import torch
from tqdm import tqdm

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the accuracy of the model on a given batch of data
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
    
    for batch, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        # Forward pass
        pred_logits = model(image)

        # Compute the loss
        loss = loss_fn(pred_logits, label.float())
        
        # Obtain classification label probabilites 
        pred_labels = torch.round(torch.sigmoid(pred_logits))

        # Accumulate the loss
        train_loss += loss.item()
        # Accumulate the accuracy
        train_acc += accuracy_fn(label, pred_labels.argmax(dim=1))

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Step the optimizer
        optimizer.step()

    # Return Averaged training loss and accuracy
    return (train_loss / len(train_loader), train_acc / len(train_loader))

def test_step(model: torch.nn.Module, loss_fn: torch.nn.Module, 
              device: torch.device, 
              test_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    """
    Test the model for one epoch on the given data loader.
    @args:
        model: The model to be tested
        loss_fn: The loss function to be used for testing
        device: The device on which the model is running
        test_loader: The data loader containing the test data
    @returns:
        A tuple containing:
        - The batch averaged loss
        - The batch averaged accuracy
    """

    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (image, label)  in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            
            # Forward pass
            pred_label = model(image)
            
            # Accumulate loss
            test_loss += loss_fn(pred_label, label)

            # Accumulate accuracy
            test_acc += accuracy_fn(label, pred_label)

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    return (test_loss, test_acc)

def 


