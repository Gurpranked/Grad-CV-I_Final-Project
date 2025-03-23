import os, stat
from kaggle.api.kaggle_api_extended import KaggleApi

# Requires Kaggle account with configured credentials via API key
def download_dataset(destination="/home/public/datasets/ships-in-satellite/"):
    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define the dataset and destination path
    dataset = "rhammell/ships-in-satellite-imagery"  # Replace with the actual dataset identifier

    # Ensure the destination folder exists
    os.makedirs(destination, exist_ok=True)
    
    # Only the owner has access to the directory
    os.chmod(destination, stat.S_IRWXU)

    # Download the dataset
    api.dataset_download_files(dataset, path=destination, unzip=True)

    print(f"Dataset successfully downloaded to {destination}")

if __name__ == "__main__":
    download_dataset()