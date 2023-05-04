from PIL import Image
import glob
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def convert_img_size(path_ini, img_size, path_save=None, inference=False):
    """converts image to defined image size"""
    if path_save == None:
        path_save = path_ini

    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", "*.tif*")))

    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", "*.tif*")))

    pil_size = [img_size[1], img_size[0]]

    for x, (im, label) in tqdm(enumerate(zip(images_sorted, masks_sorted))):
        num_file = str(x).zfill(9)
        path = path_save

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = im.resize(pil_size)
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label = label.resize(pil_size)
            label.save(
                os.path.join(path, "labels", f"{num_file}.tiff")
            )  # or 'test.tif'


def convert_to_tiff(path_ini, img_ext, lab_ext, path_save=None, inference=False):
    """converts images to tiff"""
    if path_save == None:
        path_save = path_ini

    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", f"*.{img_ext}")))
    if not inference:
        masks_sorted = sorted(
            glob.glob(os.path.join(path_ini, "labels", f"*.{lab_ext}"))
        )

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9)
        path = path_save

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label.save(
                os.path.join(path, "labels", f"{num_file}.tiff")
            )  # or 'test.tif'


def convert_to_grayscale(path_ini, path_save=None, inference=False):
    """converts images to grayscale"""
    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", "*.tif*")))

    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", "*.tif*")))

    if path_save == None:
        path_save = path_ini

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9)
        path = path_save

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = Image.fromarray(np.array(im)[:, :, 0])
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label.save(
                os.path.join(path, "labels", f"{num_file}.tiff")
            )  # or 'test.tif'


def round_to_32_pad(num: int) -> tuple[int, int]:
    """Rounds up an integer to the nearest multiple of 32. The difference
    between the number and its nearest multiple of 32 is then split in half
    and provided. This function is used to easily calculate padding values
    when padding images for suitability with PyTorch NN

    e.g. if num == 60
    then closest is 64 and two values returned will be 2 and 2
    if num == 61, then values returned will be 1 and 2

    Args:
        num (int): value of dimension

    Returns:
        tuple[int,int]:
    """

    m1 = -(-num // 32)
    m2 = 32 * m1

    val = m2 - num
    if val % 2 == 0:
        x1, x2 = val / 2, val / 2
    else:
        x1 = round(val / 2)
        x2 = val - x1

    return int(x1), int(x2)


def pad_data(path_ini, path_save=None, inference=False):
    """converts image size to multiple of 32"""
    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", "*.tif*")))

    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", "*.tif*")))

    if path_save == None:
        path_save = path_ini

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9)
        path = path_save

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = np.array(im)

        r1, r2 = round_to_32_pad(im.shape[0])
        c1, c2 = round_to_32_pad(im.shape[1])

        im = Image.fromarray(np.pad(im, pad_width=((r1, r2), (c1, c2))))
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label = np.array(label)

            r1, r2 = round_to_32_pad(label.shape[0])
            c1, c2 = round_to_32_pad(label.shape[1])

            label = Image.fromarray(np.pad(label, pad_width=((r1, r2), (c1, c2))))
            label.save(
                os.path.join(path, "labels", f"{num_file}.tiff")
            )  # or 'test.tif'


def convert_folder_format(directory, path_save):
    """converts folder format to correct format"""
    images_sorted = sorted(glob.glob(os.path.join(directory, "**", "image.tif*")))
    masks_sorted = sorted(glob.glob(os.path.join(directory, "**", "label.tif*")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9)
        path = path_save

        os.mkdir(os.path.join(path, "images"), exist_ok=True)
        os.mkdir(os.path.join(path, "labels"), exist_ok=True)

        im = Image.open(im)
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        label = Image.open(label)
        label.save(os.path.join(path, "labels", f"{num_file}.tiff"))  # or 'test.tif'


# helper functions
def decode_output(output):
    """decodes the output of segmentation model to RGB mask"""
    output = F.softmax(output, dim=1)
    mask = torch.argmax(output, dim=1).detach().cpu().numpy()
    return mask


def decode_segmap(image, nc=3):

    """
    Decode segmentation class mask into an RGB image mask
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    # 0=background, 1=lamella, 2= needle
    label_colors = np.zeros((8, 3))
    label_colors[1, :] = (255, 0, 0)
    label_colors[2, :] = (0, 255, 0)
    label_colors[3, :] = (0, 0, 255)
    label_colors[4, :] = (0, 255, 255)
    label_colors[5, :] = (255, 255, 0)
    label_colors[6, :] = (255, 0, 255)
    label_colors[7, :] = (128, 0, 0)

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # TODO: make this more efficient
    # apply the class label colours to each pixel
    for class_idx in range(1, nc):
        idx = image == class_idx
        class_idx = class_idx % 5
        r[idx] = label_colors[class_idx, 0]
        g[idx] = label_colors[class_idx, 1]
        b[idx] = label_colors[class_idx, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=-1).squeeze()
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


import os

def validate_config(config:dict):
    if "data_path" not in config:
        raise ValueError("data_path is missing. Should point to path containing labelled images.")
    else:
        path = config["data_path"]
        if not os.path.exists(path):
            raise ValueError(f"{path} directory does not exist. (data_path)")
    if "label_path" not in config:
        raise ValueError("label_path is missing. Should point to path containing labelled images.")
    else:
        path = config["label_path"]
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
        