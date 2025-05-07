# Author: Gurpreet Singh
# Date: 4/21/2025
# Description: This file contains the data processing and data loading functions

import torch
import os
import random
import PIL
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from numpy import dot

# Hyperparameters loaded from environment variables (.env file)
load_dotenv()  
IMAGES_PATH = os.getenv('IMAGES_PATH')
ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

def create_dir(path):
    """
    Create a directory if it does not exist
    """
    if not os.path.exists(ROOT_DATA_PATH + path):
        os.makedirs(ROOT_DATA_PATH + path)

def augment_image(image: torch.tensor, index: int):
    """
    Apply one of the 4 augmentation methods to an image, based on a provided index
    """
    transforms = [
        v2.RandomRotation(random.uniform(0, 360)), 
        v2.RandomHorizontalFlip(p=1.0), 
        v2.RandomVerticalFlip(p=1.0), 
        v2.GaussianBlur(kernel_size=(3,3)),
        v2.ColorJitter()
    ]

    return transforms[index%5](image)

def pad_images_with_augmentations(original_images: list[torch.tensor], desired_size: int):
    """
    Supplement the image label pairs with augmentations to match the desired quantity
    """
    
    if len(original_images) >= desired_size:
        print("No padding needed. Desired size is already met.")
        return original_images

    num_augmentations = desired_size - len(original_images)
    augmented_images = []
    num_images = len(original_images)

    while num_augmentations > 0:
        num_augmentations -= 1
        # Can use num_augmentations as index since it's being changed at each iteration anyways
        augmented_images.append(
            augment_image(
                original_images[num_augmentations%num_images], 
                num_augmentations))

    return original_images + augmented_images

def load_image(path):
    """
    Load an image from the given path
    Normalizes and converts to Tensor [0, 255]
    """
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    return transform(Image.open(path))


class ShipsDataset(Dataset):
    """
    A dataset for the ships images
    """
    def __init__(self, image_label_pairs: list[tuple[torch.tensor, float]], transform=None):
        super().__init__()
        self.image_label_pairs = image_label_pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, index: int) -> tuple[torch.tensor, float]:
        sample = self.image_label_pairs[index][0]

        if self.transform:
            sample = self.transform(self.image_label_pairs[index][0])

        return (sample, self.image_label_pairs[index][1])


def split_set(data_list: list, split_ratio_train: float, split_ratio_val: float, shuffle=False) -> tuple[list, list, list]:
    """
    Splits the data list into two parts based on a split ratio
    Also optionally shuffles the data list before splitting
    """
    if shuffle:
       random.shuffle(data_list)
    
    split_index_train = int(len(data_list) * split_ratio_train)
    split_index_val = int(len(data_list) * (split_ratio_val + split_ratio_train))

    # Train, Validation, Test
    return data_list[:split_index_train], data_list[split_index_train:split_index_val], data_list[split_index_val:]



def get_dataloaders(random_seed = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates the dataloaders for train, validation and test sets 
    Split size is hard-coded
    Train:
        - 6K Samples
            - 3K Ships
            - 3K Nonships
    Validation:
        - 200 Samples
            - 100 Ships
            - 100 Nonships
    Test:
        - 500 Samples
            - 250 Ships
            - 250 Nonships
    Returns:
        Tuple:
            - Training Dataloader
            - Validation Dataloader
            - Testing Dataloader
    """

    if os.path.exists('train_set.pt'):
        train_set = torch.load('train_set.pt', weights_only=False)
        print(f"Train Set Size: {len(train_set)}")
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    if os.path.exists('val_set.pt'):
        val_set = torch.load('val_set.pt', weights_only=False)
        print(f"Val Set Size: {len(val_set)}")
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    if os.path.exists('test_set.pt'):
        test_set = torch.load('test_set.pt', weights_only=False)
        print(f"Test Set Size: {len(test_set)}")
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    else:    
        random.seed(random_seed)

        images_paths = os.listdir(IMAGES_PATH)
        # labels = [int(path.split("__")[0] for path in images_paths)]
        
        ships = []
        nonships = []

        for path in images_paths:
            category = int(path.split("__")[0])
            if category == 0:
                nonships.append(load_image(IMAGES_PATH + path))
            else:
                ships.append(load_image(IMAGES_PATH + path))

        test_size = 250
        val_size = 100

        # Randomly sample 250 images from each category for test set
        test_ships = ships[:test_size]
        test_nonships = nonships[:test_size]

        # Randomly sample 100 images from each category for val set
        val_ships = ships[test_size:test_size + val_size]
        val_nonships = nonships[test_size:test_size + val_size]

        # Pad the remaining images to 3000 for each class
        ships_padded = pad_images_with_augmentations(ships[test_size + val_size:], 3000)
        nonships_padded = pad_images_with_augmentations(nonships[test_size + val_size:], 3000)

        # Add corresponding labels to all data
        test_ships_with_labels = [(image, 1) for image in test_ships]
        test_nonships_with_labels = [(image, 0) for image in test_nonships]
        val_ships_with_labels = [(image, 1) for image in val_ships]
        val_nonships_with_labels = [(image, 0) for image in val_nonships]
        train_ships_with_labels = [(image, 1) for image in ships_padded]
        train_nonships_with_labels = [(image, 0) for image in nonships_padded]

        # Combine and shuffle data for each set
        test_set = test_ships_with_labels + test_nonships_with_labels
        val_set = val_ships_with_labels + val_nonships_with_labels
        train_set = train_ships_with_labels + train_nonships_with_labels
        print(f"Train Set Size: {len(train_set)}")
        print(f"Test Set Size: {len(test_set)}")
        print(f"Val Set Size: {len(val_set)}")

        # Create Dataset
        train_set = ShipsDataset(train_set, transform=None)
        val_set = ShipsDataset(val_set, transform=None)
        test_set = ShipsDataset(test_set, transform=None)
        torch.save(train_set, 'train_set.pt')
        torch.save(val_set, 'val_set.pt')
        torch.save(test_set, 'test_set.pt')
        
        # Create the data loaders
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Return the data loaders       
    return train_loader, val_loader, test_loader


