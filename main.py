import os
import torch
import argsparse
from dotenv import load_dotenv
from tqdm import tqdm
from preprocess.data_process import get_dataloaders
from torch.utils.data import DataLoader
from utilities import train_step

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
    
    for epoch in tqdm(range(EPOCHS)):
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train the model
        train_step(model, loss_fn, optimizer, device, train_loader)

if __name__ == '__main__':
    # Your code here
    main()