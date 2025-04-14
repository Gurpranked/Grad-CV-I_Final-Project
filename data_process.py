import torch
import os
import random
import PIL
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torchvision.transforms import v2
from PIL import Image


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
    Pad the image label pairs with augmentations to match the desired size
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
    def __init__(self, )

def data_process(ships_samples_amount=4000, nonships_samples_amount=4000):
    create_dir("formatted/train/ships")
    create_dir("formatted/train/nonships")
    create_dir("formatted/val/ships")
    create_dir("formatted/val/nonships")
    create_dir("formatted/test/ships")
    create_dir("formatted/test/nonships")

    images_paths = os.list_dir(IMAGES_PATH)
    labels = [int(path.split("__")[0] for path in images_paths)]
    
    ships_paths = []
    nonships_paths = []

    for path in images_paths:
        category = int(path.split("__")[0])
        if category == 0:
            nonships_paths.append(load_image(IMAGES_PATH + path))
        else:
            ships_paths.append(Image.open(IMAGES_PATH + path))

    

    padded_ships = pad_images_with_augmentations(ships_paths, ships_samples_amount)
    padded_nonships = pad_images_with_augmentations(nonships_paths, nonships_samples_amount)

    padded_ships_with_labels = [(image, 1) for image in padded_ships]
    padded_nonships_with_labels = [(image, 0) for image in padded_nonships]

    images_with_labels = padded_ships_with_labels + padded_nonships_with_labels

