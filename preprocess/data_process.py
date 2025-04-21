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

load_dotenv()  
IMAGES_PATH = os.getenv('IMAGES_PATH')
ROOT_DATA_PATH = os.getnenv('ROOT_DATA_PATH')

    
def create_dir(path):
    """
    Create a directory if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(ROOT_DATA_PATH + path)

def augment_image(image: PIL.Image):
    """
    Randomly apply one or more transformations to the image
    """
    
    transform_pipeline = v2.Compose([
        v2.RandomRotation(random.uniform(0, 360)),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.GaussianBlur(kernel_size=(7, 7)),
    ])

    return transform_pipeline(image)

def pad_images_with_augmentations(original_images: list[PIL.Image], desired_size: int):
    
    """
    Supplement the image label pairs with augmentations to match the desired quantity
    """
    
    num_augmentations = max(1, (desired_size - len(original_images)) // len(original_images))

    padded_images = []
    
    for i in range(num_augmentations):
        for image in original_images:
            padded_images.append(augment_image(image))

    return padded_images


def load_image(path):
    return Image.open(path)


class ShipsDataset(Dataset):
    def __init__(self, image_label_pairs: list[tuple], transform=None):
        super().__init__()
        self.image_label_pairs = image_label_pairs
        if transform:
            self.transform = transform
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, index):
        sample = self.image_label_pairs[index][0]

        if self.transform:
            sample = self.transform(self.image_label_pairs[index][0])
        
        return (sample, self.image_label_pairs[index][1])


def split_set(data_list: , split_ratio: float, shuffle=False):
    if shuffle:
       random.shuffle(data_list)
    
    split_index = int(len(data_list) * split_ratio)

    return data_list[:split_index], data_list[split_index:]



def data_process(ships_samples_target_amount=4000, nonships_samples_target_amount=4000, , train_split=0.8, val_split=0.1, random_seed = 42):
    create_dir("formatted/train/ships")
    create_dir("formatted/train/nonships")
    create_dir("formatted/val/ships")
    create_dir("formatted/val/nonships")
    create_dir("formatted/test/ships")
    create_dir("formatted/test/nonships")
    random.seed(random_seed)

    images_paths = os.list_dir(IMAGES_PATH)
    # labels = [int(path.split("__")[0] for path in images_paths)]
    
    ships_paths = []
    nonships_paths = []

    for path in images_paths:
        category = int(path.split("__")[0])
        if category == 0:
            nonships_paths.append(load_image(IMAGES_PATH + path))
        else:
            ships_paths.append(load_image(IMAGES_PATH + path))

    

    padded_ships = pad_images_with_augmentations(ships_paths, ships_samples_amount)
    padded_nonships = pad_images_with_augmentations(nonships_paths, nonships_samples_amount)

    padded_ships_with_labels = [(image, 1) for image in padded_ships]
    padded_nonships_with_labels = [(image, 0) for image in padded_nonships]

    # Combine the images of both classes
    images_with_labels = padded_ships_with_labels + padded_nonships_with_labels

    # Split the dataset into train, validation, and test sets
    train_set, test_set = split_set(images_with_labels, train_split, shuffle=True)
    train_set, val_set = split_set(train_set, val_split, shuffle=False)
    
    # Create the data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    




