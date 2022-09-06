import argparse
from datetime import datetime
    
# import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from model_utils import *
import torch
import wandb

def inference(images, model, model_path, device, WANDB=False):
    """Helper function for performing inference with the model"""
    # Load the model
    model.load_state_dict(torch.load(model_path)).to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        output_mask = decode_output(outputs)

        if WANDB:
            img_base = images.detach().cpu().squeeze().numpy()
            img_rgb = np.dstack((img_base, img_base, img_base))

            wb_img = wandb.Image(img_rgb, caption="Input Image")
            wb_mask = wandb.Image(output_mask, caption="Output Mask")
            wandb.log({"image": wb_img, "mask": wb_mask})

        return outputs, output_mask

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="the directory containing the training data",
        dest="data",
        action="store",
        default="data",
    )
    parser.add_argument(
        "--model_path",
        help="show debugging visualisation during training",
        dest="debug",
        action="store_true",
    )
    parser.add_argument(
        "--wandb",
        help="report results to wandb during training and validation",
        dest="wandb",
        action="store_true",
    )
    parser.add_argument(
        "--checkpoint",
        help="start model training from checkpoint",
        dest="checkpoint",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        help="number of epochs to train",
        dest="epochs",
        action="store",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    data_path = args.data
    model_checkpoint = args.checkpoint
    epochs = args.epochs
    DEBUG = args.debug
    WANDB = args.wandb

    model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,  # grayscale images
            classes=3,  # background, needle, lamella
        )

    