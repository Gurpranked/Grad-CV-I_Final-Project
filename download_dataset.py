import os, stat
import argsparse
from kaggle.api.kaggle_api_extended import KaggleApi

# Requires Kaggle account with configured credentials via API key
def download_dataset(destination: str):
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
    parser = argsparse.ArgumentParser()
    parser.add_argument('--dest', type=str, description='Where to download dataset', required=True)
    args = parser.parse_args()
    print("Downloading Dataset to " + args.dest + ". NOTE: This may take a while. Pleaset")
    download_dataset(args.dest)
    print("Dataset Downloaded to " + args.dest + f". NOTE: Only the owner {os.getuid()} has access to the directory.")