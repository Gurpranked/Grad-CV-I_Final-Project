import os
import torch
import argsparse
from dotenv import load_dotenv
from tqdm import tqdm
from preprocess.data_process import get_dataloaders
from torch.utils.data import DataLoader

def main():
    # Arugment parsers
    parsers = argsparse.ArgumentParser()
    parsers.add_argument('--model', type=str, description='Which model to train', )
    args = parsers.parse_args()

    # Load environment variables
    load_dotenv()
    IMAGES_PATH = os.getenv('IMAGES_PATH')
    ROOT_DATA_PATH = os.getnenv('ROOT_DATA_PATH')
    BATCH_SIZE= os.getenv('BATCH_SIZE')
    EPOCHS = os.getenv('EPOCHS')
    LR = os.getenv('LR')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(IMAGES_PATH, ROOT_DATA_PATH, BATCH_SIZE)
    
    # Set manual random seed
    torch.manual_seed(42)

    # Load the model
    # TODO: Specify class names for each model after they are implemented
    model = None if args.model == "baseline" else None
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train the model
        train(model, loss_fn, optimizer, device, train_loader)


def train(model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, dataloader: torch.data.DataLoader):
    """
    Trains the model for one epoch on the given data loader.
    """
    model.train()
    train_loss = 0
    
    for batch in tqdm(dataloader):
        images, labels = batch[0].to(device), batch[1].to(device)
        
        pred_logits = model(images)
        loss = loss_fn(pred_logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

if __name__ == '__main__':
    # Your code here
    main()