#!/usr/bin/env python3

import glob

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
import dask.array as da
import os

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
        mask = Image.fromarray(mask).resize(
            (1536 // 4, 1024 // 4), resample=Image.NEAREST
        )
        mask = torch.tensor(np.asarray(mask)).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.images)


def load_dask_dataset(data_path: str):
    images = da.from_zarr(os.path.join(data_path, "images.zarr"))
    masks = da.from_zarr(os.path.join(data_path, "masks.zarr"))
    return images, masks

def preprocess_data(data_path: str, num_classes: int = 3, batch_size: int = 1, val_split: float = 0.2):
    images, masks = load_dask_dataset(data_path)
    print(f"Loading dataset from {data_path} of length {images.shape[0]}")

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
        seg_dataset, batch_size=batch_size, sampler=train_sampler
    )  # shuffle=True,
    print(f"Train dataset has {len(train_data_loader)} batches of size {batch_size}")

    val_data_loader = DataLoader(
        seg_dataset, batch_size=batch_size, sampler=val_sampler
    )  # shuffle=True,
    print(f"Validation dataset has {len(val_data_loader)} batches of size {batch_size}")

    return train_data_loader, val_data_loader


# ref: https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a