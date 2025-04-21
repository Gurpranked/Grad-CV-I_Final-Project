import os
from dotenv import load_dotenv
from preprocess.data_process import get_dataloaders


def main():
    load_dotenv()
    IMAGES_PATH = os.getenv('IMAGES_PATH')
    ROOT_DATA_PATH = os.getnenv('ROOT_DATA_PATH')
    BATCH_SIZE= os.getenv('BATCH_SIZE')
    train_loader, val_loader, test_loader = get_dataloaders(IMAGES_PATH, ROOT_DATA_PATH, BATCH_SIZE)
    
    

if __name__ == '__main__':
    # Your code here
    main()