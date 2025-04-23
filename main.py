import os
import torch
import argsparse
from dotenv import load_dotenv
from tqdm import tqdm
from preprocess.data_process import get_dataloaders
from utilities import train_step, test_step
from timeit import default_timer as timer

def main():
    # Arugment parsers
    parsers = argsparse.ArgumentParser()
    parsers.add_argument('--model', type=str, description='Which model to train', required=True)
    args = parsers.parse_args()

    print("Friendly Reminder: All Hyperparameters are configured in the .env file. Please make sure to set all of them before running this script.")


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
    
    train_start_time = timer()

    print("-----------Training starting----------")
    for epoch in tqdm(range(EPOCHS)):
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train the model
        train_step(model, loss_fn, optimizer, device, train_loader)
        test_step(model, loss_fn, device, test_loader)
    print("-----------Training finished----------")

    train_end_time = timer()

    total_train_time = train_end_time - train_start_time

    print(f"Training completed in {total_train_time:.2f} seconds")


if __name__ == '__main__':
    # Your code here
    main()