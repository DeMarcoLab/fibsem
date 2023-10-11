#!/usr/bin/env python3

import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
import dask.array as da
import os
import tifffile as tff
import random
from PIL import Image

# transformations
ROT_ANGLE = 15
PROB = 0.1

transformations_input = transforms.Compose(
    [

        transforms.RandomRotation(ROT_ANGLE),
        transforms.RandomHorizontalFlip(p=PROB),
        transforms.RandomVerticalFlip(p=PROB),
        transforms.RandomAutocontrast(p=PROB),
        transforms.RandomEqualize(p=PROB),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
)

transformations_target = transforms.Compose(
    [
        transforms.RandomRotation(ROT_ANGLE),
        transforms.RandomHorizontalFlip(p=PROB),
        transforms.RandomVerticalFlip(p=PROB),

    ]
)


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, num_classes: int, transforms_input=None, transforms_target=None):
        self.images = images
        self.masks = masks
        self.num_classes = num_classes
        self.transforms_input = transforms_input
        self.transforms_target = transforms_target

        assert len(self.images) == len(self.masks), "Images and masks are not the same length"
        assert self.transforms_target is not None if self.transforms_input is not None else True, "transforms_target must be provided if transforms_input is provided"

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):


        image = np.asarray(self.images[idx])
        mask = np.asarray(self.masks[idx])

        # if mask > num_class, set to zero
        mask[mask >= self.num_classes] = 0
        # - the problem was ToTensor was destroying the class index for the labels (rounding them to 0-1)
        # need to to transformation manually
        # mask = torch.tensor(mask).unsqueeze(0)
        
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        
        if self.transforms_input:
            seed = random.randint(0, 1000)
            self._set_seed(seed)
            image = self.transforms_input(image)
            self._set_seed(seed)
            mask = self.transforms_target(mask)

            # convert image to float32 scaled between 0-1
            image = image.float() / 255.0
                        
        return image, mask

    def __len__(self):
        return len(self.images)

from pathlib import Path

def load_dask_dataset_v2(data_paths: list[Path], label_paths: list[Path]):

    sorted_img_filenames, sorted_mask_filenames = [], []
    for data_path, label_path in zip(data_paths, label_paths):
        sorted_img_filenames += sorted(glob.glob(os.path.join(data_path, "*.tif*")))
        sorted_mask_filenames += sorted(glob.glob(os.path.join(label_path, "*.tif*")))

    # TODO: change to dask-image
    img_arr = tff.imread(sorted_img_filenames, aszarr=True)
    mask_arr = tff.imread(sorted_mask_filenames, aszarr=True)

    images = da.from_zarr(img_arr)
    masks = da.from_zarr(mask_arr)

    images = images.rechunk(chunks=(1, images.shape[1], images.shape[2]))
    masks = masks.rechunk(chunks=(1, images.shape[1], images.shape[2]))

    return images, masks


def preprocess_data(data_paths: list[Path], label_paths: list[Path], num_classes: int = 3, 
                    batch_size: int = 1, val_split: float = 0.15, 
                    _validate_dataset:bool = True):
    
    # if _validate_dataset:
        # validate_dataset(data_path, label_path)

    if not isinstance(data_paths, list):
        data_paths = [data_paths]
    if not isinstance(label_paths, list):
        label_paths = [label_paths]

    images, masks = load_dask_dataset_v2(data_paths, label_paths)


    print(f"Loading dataset from {len(data_paths)} paths: {images.shape[0]}")
    for path in data_paths:
        print(f"Loaded from {path}")

    # load dataset
    seg_dataset = SegmentationDataset(
        images, masks, num_classes, 
        transforms_input=transformations_input, 
        transforms_target=transformations_target
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

    # TODO: try larger val batch size? TODO: turn off transformations for validation set
    val_data_loader = DataLoader(
        seg_dataset, batch_size=1, sampler=val_sampler
    )  # shuffle=True,
    print(f"Validation dataset has {len(val_data_loader)} batches of size 1")

    return train_data_loader, val_data_loader


# ref: https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

# Helper functions
def validate_dataset(data_path: str, label_path: str):
    print("validating dataset...")
    # get data
    filenames = sorted(glob.glob(os.path.join(data_path, "*.tif*")))
    labels = sorted(glob.glob(os.path.join(label_path,  "*.tif*")))

    # check length
    assert len(filenames) == len(labels), "Images and labels are not the same length"
    print(f"{len(filenames)} images, {len(labels)} labels")

    base_shape = tff.imread(filenames[0]).shape
    for i, (fname, lfname) in enumerate(list(zip(filenames, labels))):
        img, label = tff.imread(fname), tff.imread(lfname)

        if (img.shape[0:2] != label.shape[0:2]) or (img.shape[0:2] != base_shape[0:2]):

            raise ValueError(
                "invalid data, image shape is different to label shape",
                i,
                os.path.basename(fname),
                os.path.basename(lfname),
                img.shape,
                label.shape,
                "\n",
                "You can run convert_img_size() in utils.py to convert all images and labels in the dataset to the desired size.",
            )

        if (img.ndim > 2) or (label.ndim > 2):

            raise ValueError(
                "Image has too many dimensions, must be in 2D grayscale format.",
                i,
                os.path.basename(fname),
                os.path.basename(lfname),
                img.shape,
                label.shape,
                "\n",
                "You can run convert_to_grayscale() in utils.py to convert all images and labels in the dataset to the desired format.",
            )

        if (img.shape[0] % 32 != 0) or (img.shape[1] % 32 != 0):
            raise ValueError(
                "Wrong padding, dimensions must be divisible by 32.",
                i,
                os.path.basename(fname),
                os.path.basename(lfname),
                img.shape,
                label.shape,
                "\n",
                "You can run pad_data in utils.py to convert the dataset to correct format.",
            )

    print("finished validating dataset.")
