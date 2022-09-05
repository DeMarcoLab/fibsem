#!/usr/bin/env python3

import glob

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

# transformations
transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((1024 // 4, 1536 // 4)),
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
        image = self.images[idx]

        if self.transforms:
            image = self.transforms(image)

        mask = self.masks[idx]

        # - the problem was ToTensor was destroying the class index for the labels (rounding them to 0-1)
        # need to to transformation manually
        mask = Image.fromarray(mask).resize(
            (1536 // 4, 1024 // 4), resample=PIL.Image.NEAREST
        )
        mask = torch.tensor(np.asarray(mask)).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.images)


# change this to pre-processing- and cache
def load_images_and_masks_in_path(images_path, masks_path):
    images = []
    masks = []
    sorted_img_filenames = sorted(glob.glob(images_path + ".png"))  #[-435:]
    sorted_mask_filenames = sorted(glob.glob(masks_path + ".png"))  #[-435:]

    for img_fname, mask_fname in tqdm(
        list(zip(sorted_img_filenames, sorted_mask_filenames))
    ):

        image = np.asarray(Image.open(img_fname))
        mask = np.asarray(Image.open(mask_fname))

        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)


def preprocess_data(data_path, num_classes=3, batch_size=1, val_size=0.2):

    img_path = f"{data_path}/train/**/img"
    label_path = f"{data_path}/train/**/label"
    print(f"Loading dataset from {img_path}")

    train_images, train_masks = load_images_and_masks_in_path(img_path, label_path)

    # load dataset
    seg_dataset = SegmentationDataset(
        train_images, train_masks, num_classes, transforms=transformation
    )

    # train/validation splits
    dataset_size = len(seg_dataset)
    dataset_idx = list(range(dataset_size))
    split_idx = int(np.floor(val_size * dataset_size))
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