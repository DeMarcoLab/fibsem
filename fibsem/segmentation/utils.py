


import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from huggingface_hub import HfApi, hf_hub_download
from fibsem import config as cfg

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from fibsem.segmentation.config import CLASS_COLORS, CLASS_COLORS_RGB, CLASS_LABELS


# helper functions
def decode_output(output):
    """decodes the output of segmentation model to RGB mask"""
    output = F.softmax(output, dim=1)
    mask = torch.argmax(output, dim=1).detach().cpu().numpy()
    return mask


def decode_segmap(image, nc=5):

    """
    Decode segmentation class mask into an RGB image mask
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    # 0=background, 1=lamella, 2= needle
    label_colors = np.zeros((8, 3))
    label_colors[1, :] = (255, 0, 0)
    label_colors[2, :] = (0, 255, 0)
    label_colors[3, :] = (0, 255, 255)
    label_colors[4, :] = (255, 255, 0)    
    label_colors[5, :] = (255, 0, 255)
    label_colors[6, :] = (0, 0, 255)
    label_colors[7, :] = (128, 0, 0) # TODO: convert this to use CLASS_COLORS

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # TODO: make this more efficient
    # apply the class label colours to each pixel
    for class_idx in range(1, nc+1):
        idx = image == class_idx
        # class_idx = class_idx % len(label_colors)
        r[idx] = label_colors[class_idx, 0]
        g[idx] = label_colors[class_idx, 1]
        b[idx] = label_colors[class_idx, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=-1).squeeze()
    return rgb_mask
   
def decode_segmap_v2(image, colormap: list[tuple] = None) -> np.ndarray:
    """
    Decode segmentation class mask into an RGB image mask
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    if colormap is None:
        from fibsem.segmentation.config import CLASS_COLORS_RGB
        colormap = CLASS_COLORS_RGB

    # convert class masks to rgb values
    rgb_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(image)

    for class_idx in unique_labels:
        rgb_mask[image == class_idx] = colormap[class_idx]

    return rgb_mask

def show_img_and_mask(imgs, gts, mask, title="Image, Ground Truth and Mask"):
    """Show a plot of the image, ground truth mask, and predicted mask"""
    n_imgs = len(imgs)
    imgs = imgs.cpu().detach()

    fig, ax = plt.subplots(n_imgs, 3, figsize=(8, 6))
    fig.suptitle(title)

    for i in range(len(imgs)):

        img = imgs[i].permute(1, 2, 0).squeeze()
        gt = decode_segmap(gts[i].permute(1, 2, 0).squeeze())  # convert to rgb mask

        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(gt)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(mask)
        ax[2].set_title("Predicted Mask")

    plt.show()


def show_memory_usage():
    """Show total, reserved and allocated gpu memory"""
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    if r == 0:
        r = 0.00001
    print("GPU memory", t, r, a, f, f"{f/r:.3}")


def show_values(ten):
    """Show tensor statistics for debugging"""
    unq = np.unique(ten.detach().cpu().numpy())
    print(ten.shape, ten.min(), ten.max(), ten.mean(), ten.std(), unq)


def validate_config(config:dict):
    if "data_paths" not in config:
        raise ValueError("data_path is missing. Should point to path containing labelled images.")
    else:
        for path in config["data_paths"]:
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (data_path)")
    if "label_paths" not in config:
        raise ValueError("label_path is missing. Should point to path containing labelled images.")
    else:
        for path in config["label_paths"]:
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (label_path)")
    if "save_path" not in config:
        raise ValueError("save_path is missing. Should point to save the model.")
    else:
        path = config["save_path"]
        os.makedirs(path, exist_ok=True)
    if "wandb" not in config:
        raise ValueError("wandb is missing. Used to enable/disable wandb logging in training loop. Should be a boolean value.")
    else:
        val = config["wandb"]
        if type(val) != bool:
            raise TypeError(f"{val} is not a boolean (True/False). (wandb)")
    if "checkpoint" not in config:
        raise ValueError("checkpoint is missing. Either a path leading to the desired saved model, or None value.")
    else:
        path = config["checkpoint"]
        if path is not None and not os.path.exists(path):
            raise ValueError(f"{path} directory does not exist. (checkpoint)")
    if "encoder" not in config:
        raise ValueError("encoder is missing. Used to specify which model architecture to use. Default is resnet18.")
    else:
        val = config["encoder"]
        if type(val) != str:
            raise TypeError(f"{val} must be a string. (encoder)")
        elif val not in unet_encoders:
            raise ValueError(f"{val} not a valid encoder. Check readme for full list. (encoder)")
    if "epochs" not in config:
        raise ValueError("epochs is missing. Integer value used to determine number of epochs model trains for.")
    else:
        val = config["epochs"]
        if type(val) != int or val <= 0:
            raise TypeError(f"{val} is not a positive integer. (epochs)")  
    if "batch_size" not in config:
        raise ValueError("batch_size is missing. Integer value used to determine batch size of dataset.")
    else:
        val = config["batch_size"]
        if type(val) != int or val <= 0:
            raise TypeError(f"{val} is not a positive integer. (batch_size)")    
    if "num_classes" not in config:
        raise ValueError("num_classes is missing. Integer value used to determine number of classes model classifies.")
    else:
        val = config["num_classes"]
        if type(val) != int or val <= 0:
            raise TypeError(f"{val} is not a positive integer. (num_classes)")  
    if "lr" not in config:
        raise ValueError("lr is missing. Float value indicating the learning rate of the model.")
    else:
        val = config["lr"]
        if type(val) == float:
            if val <= 0:
                raise ValueError(f"{val} must be a positive float value (lr).")
        else:
            raise TypeError(f"{val} is not a float. (lr)")  
    if "wandb_project" not in config:
        raise ValueError("wandb_project is missing. String indicating the wandb project title for login.")
    else:
        val = config["wandb_project"]
        if type(val) != str:
            raise TypeError(f"{val} is not a string. (wandb_project)")
    if "wandb_entity" not in config:
        raise ValueError("wandb_entity is missing. String indicating the wandb login credentials.")
    else:
        val = config["wandb_entity"]
        if type(val) != str:
            raise TypeError(f"{val} is not a string. (wandb_project)")
    return
    
        
# All UNet encoders that work with Imagenet weights
unet_encoders = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x16d",
    "resnext101_32x32d",
    "resnext101_32x48d",
    "dpn68",
    "dpn98",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "mobilenet_v2",
    "efficientnet-b0",
    "xception"
]
        

def plot_segmentations(images: list[np.ndarray], masks: list[np.ndarray], 
    alpha=0.5, legend: bool = True, show: bool = True) -> plt.Figure:
    """Plot the image and mask overlaid with a class legend."""
    
    if not isinstance(images, list):
        images = [images]
    if not isinstance(masks, list):
        masks = [masks]

    if len(images) != len(masks):
        raise ValueError("images and masks must be the same length")
    
    n_cols = len(images)
    fig, ax = plt.subplots(1, len(images), figsize=(10*n_cols/2, 10*n_cols/2))
    for i, (image, mask) in enumerate(zip(images, masks)):

        # convert to rgb mask        
        rgb = decode_segmap_v2(mask)

        if len(images) == 1:
            axes = ax
        else:
            axes = ax[i]
        # plot
        axes.imshow(image.data, cmap='gray')
        axes.imshow(rgb, alpha=0.4)

        # filter legend to only include classes present in mask
        class_ids = np.unique(mask)
        
        colors, labels = [], []
        for idx in class_ids:
            colors.append(CLASS_COLORS[idx])
            labels.append(CLASS_LABELS[idx])

        # Create a patch for each class color
        patches = [mpatches.Patch(color=color, label=label) 
                for color, label in zip(colors, labels)]

        # Add the patches to the legend
        if legend:
            axes.legend(handles=patches, loc="best", prop={'size': 6})
    
    if show:
        plt.show()

    return fig

## Huggingface Utils

def list_available_checkpoints():
    api = HfApi()
    files = api.list_repo_files(cfg.HUGGINFACE_REPO)
    checkpoints = []
    for file in files:
        if file.endswith(".pt") and "archive" not in file:
            checkpoints.append(file)

    return checkpoints

def download_checkpoint(checkpoint: str):
    if os.path.exists(checkpoint):
        checkpoint = checkpoint
    else:
        checkpoint = hf_hub_download(repo_id=cfg.HUGGINFACE_REPO, filename=checkpoint)
    return checkpoint