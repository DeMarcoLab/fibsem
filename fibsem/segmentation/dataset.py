#!/usr/bin/env python3

import glob
import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
import dask.array as da
import os
import tifffile as tff

# transformations
transformation = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, num_classes: int, transforms=None):
        self.images = images
        self.masks = masks
        self.num_classes = num_classes
        self.transforms = transforms

    def __getitem__(self, idx):
        image = self.images[idx].compute()

        if self.transforms:
            image = self.transforms(image)

        mask = self.masks[idx].compute()

        # - the problem was ToTensor was destroying the class index for the labels (rounding them to 0-1)
        # need to to transformation manually
        mask = torch.tensor(mask).unsqueeze(0)
<<<<<<< HEAD
        #print(np.unique(mask))

        #mask = torch.round((mask/255) * (self.num_classes - 1)) # -1 because background is index 0
=======
>>>>>>> ca8a3b1ef9078cdddcc68fdc723b7813f137de5b

        return image, mask

    def __len__(self):
        return len(self.images)


def load_dask_dataset(data_dir: str):
    sorted_img_filenames = sorted(glob.glob(os.path.join(data_dir, "**\image.tif*")))  #[-435:]
    sorted_mask_filenames = sorted(glob.glob(os.path.join(data_dir, "**\label.tif*")))  #[-435:]

    img_arr = tff.imread(sorted_img_filenames, aszarr=True)
    mask_arr = tff.imread(sorted_mask_filenames, aszarr=True)

    images = da.from_zarr(img_arr)
    masks = da.from_zarr(mask_arr)

    return images, masks

def preprocess_data(data_path: str, num_classes: int = 3, batch_size: int = 1, val_split: float = 0.2):
    images, masks = load_dask_dataset(data_path)
    print(f"Loading dataset from {data_path} of length {images.shape[0]}")
    #print(np.unique(masks[0].compute())) 

    # load dataset
    seg_dataset = SegmentationDataset(
        images, masks, num_classes, transforms=transformation
    )

    # train/validation splits
    dataset_size = len(seg_dataset)
    dataset_idx = list(range(dataset_size))
    split_idx = int(np.floor(val_split * dataset_size))
    train_idx = dataset_idx[split_idx:]
    val_idx = dataset_idx[:split_idx]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_data_loader = DataLoader(
        seg_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
    )  # shuffle=True,
    print(f"Train dataset has {len(train_data_loader)} batches of size {batch_size}")

    val_data_loader = DataLoader(
        seg_dataset, batch_size=1, sampler=val_sampler
    )  # shuffle=True,
    print(f"Validation dataset has {len(val_data_loader)} batches of size {batch_size}")

    return train_data_loader, val_data_loader


# ref: https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

# Helper functions
# WIP
def convert_dataset_to_tif(dataset_path, f_extension):
    old_images = glob.glob(os.path.join(dataset_path, f"*.{f_extension}"))
    for img in old_images:
        new_image = Image.open(img)
        f_name = pathlib.Path(img).stem()
        new_image.save(os.path.join(dataset_path), f"{f_name}.tif") 


#convert_dataset_to_tif("C:\\Users\lucil\OneDrive\Bureau\DeMarco_Lab\data\train\000000001", 'png') 