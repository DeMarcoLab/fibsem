import glob
import napari
import numpy as np
from skimage import data
import zarr
import dask.array as da
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
import tifffile as tff
import os
from PIL import Image
from tqdm import tqdm

img_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data\train\**\img"
label_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data\train\**\label"
save_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\fibsem segmentation"

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
            (1536 // 4, 1024 // 4), resample=Image.NEAREST
        )
        mask = torch.tensor(np.asarray(mask)).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.images)

def import_images(path: str) -> zarr.Array:
    vol = tff.imread(os.path.join(path, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    imgs = zarr.open(vol)
    return imgs

def label_images(save_dir: str, img_dir: str, zarr_set: zarr.Array) -> None:

    filenames = sorted(glob.glob(os.path.join(img_dir, "*.tif*")))
    # while i <= zarr_set.size:
    for img, fname in zip(zarr_set, filenames):
        print(fname)
        viewer = napari.view_image(img)
        #manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # screenshot = ImageGrab.grabclipboard()
        # if screenshot is None:
        #     print("You forgot to copy image to clipboard.")

        # Saves an img with the keypoints superimposed.
        # viewer.layers.save(os.path.join(save_dir, os.path.basename(fname)))
        os.makedirs(os.path.join(save_dir, os.path.basename(fname).split(".")[0]))
        viewer.layers["img"].save(os.path.join(save_dir, os.path.basename(fname), "image"))
        # i = i +1  b

def save_zarr_dataset(img_path: str, label_path: str, save_path: str, img_size = (1024,1024)) -> None:
    images = []
    masks = []
    sorted_img_filenames = sorted(glob.glob(img_path + ".png"))  #[-435:]
    sorted_mask_filenames = sorted(glob.glob(label_path + ".png"))  #[-435:]

    for img_fname, mask_fname in tqdm(
        list(zip(sorted_img_filenames, sorted_mask_filenames))
    ):
        image = np.asarray(Image.open(img_fname).resize(img_size))
        mask = np.asarray(Image.open(mask_fname).resize(img_size))

        images.append(image)
        masks.append(mask)

    zarr.save(os.path.join(save_path, "images.zarr"), np.array(images))
    zarr.save(os.path.join(save_path, "masks.zarr"), np.array(masks))

def load_dask_dataset(data_path: str):
    images = da.from_zarr(os.path.join(data_path, "images.zarr"))
    masks = da.from_zarr(os.path.join(data_path, "masks.zarr"))
    return images, masks

def preprocess_data(data_path: str, num_classes: int = 3, batch_size: int = 1, val_split: float = 0.2):
    images, masks = load_dask_dataset(data_path)
    print(f"Loading dataset from {data_path} of size {images.size}")

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


# imgs = import_images(img_dir)
# label_images(save_dir, imgs)
save_zarr_dataset(img_path, label_path, save_path)
tr, vd = preprocess_data(save_path)