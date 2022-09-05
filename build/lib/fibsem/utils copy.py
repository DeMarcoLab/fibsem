#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# helper functions
def decode_output(output):
    """decodes the output of segmentation model to RGB mask"""
    output = F.softmax(output, dim=1)
    mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    mask = decode_segmap(mask)
    return mask


def decode_segmap(image, nc=3):

    """
    Decode segmentation class mask into an RGB image mask 
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    # 0=background, 1=lamella, 2= needle
    label_colors = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0)])

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # apply the class label colours to each pixel
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=2)
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

    print("GPU memory", t, r, a, f, f"{f/r:.3}")


def show_values(ten):
    """Show tensor statistics for debugging"""
    unq = np.unique(ten.detach().cpu().numpy())
    print(ten.shape, ten.min(), ten.max(), ten.mean(), ten.std(), unq)
