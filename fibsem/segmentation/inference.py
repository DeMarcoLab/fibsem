#!/usr/bin/env python3

import argparse
    
# import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from fibsem.segmentation import dataset, utils, validate_config
import torch
import os
import wandb
from PIL import Image
import yaml
import zarr
import glob
import tifffile as tff
import time

def inference(images, output_dir, model, model_path, device, WANDB=False):
    """Helper function for performing inference with the model"""
    # Load the model
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    # model.eval()

    # Inference
    with torch.no_grad(): # TODO: move this down to only wrap the model inference
        vol = tff.imread(os.path.join(images, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
        zarr_set = zarr.open(vol)

        filenames = sorted(glob.glob(os.path.join(images, "*.tif*")))

        for img, fname in zip(zarr_set, filenames):
            img = torch.tensor(np.asarray(img)).unsqueeze(0)
            img = img.to(device)
            outputs = model(img[None, :, :, :].float())
            output_mask = utils.decode_output(outputs)
            
            output = Image.fromarray(output_mask) 
            path = os.path.join(output_dir, os.path.basename(fname).split(".")[0])
            
            # if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
            input_img = Image.fromarray(img.detach().cpu().squeeze().numpy())
            input_img.save(os.path.join(path, "input.tif"))
            output.save(os.path.join(path, "output.tif"))  # or 'test.tif'

            if WANDB:
                img_base = img.detach().cpu().squeeze().numpy()
                img_rgb = np.dstack((img_base, img_base, img_base))

                wb_img = wandb.Image(img_rgb, caption="Input Image")
                wb_mask = wandb.Image(output_mask, caption="Output Mask")
                wandb.log({"image": wb_img, "mask": wb_mask})

        return outputs, output_mask

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="the directory containing the config file to use",
        dest="config",
        action="store",
        default=os.path.join("fibsem", "segmentation", "config.yml"),
    )
    args = parser.parse_args()
    config_dir = args.config

    # NOTE: Setup your config.yml file
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)

    print("Validating config file.")
    validate_config.validate_config(config, "inference")

    # directories
    data_path = config["inference"]["data_dir"]
    model_weights = config["inference"]["model_dir"]
    output_dir = config["inference"]["output_dir"]

    # other parameters
    cuda = config["inference"]["cuda"]
    WANDB = config["inference"]["wandb"]


    model = smp.Unet(
            encoder_name=config["inference"]["encoder"],
            encoder_weights="imagenet",
            in_channels=1,  # grayscale images
            classes=config["inference"]["num_classes"],  # background, needle, lamella
        )

    if WANDB:
        # weights and biases setup
        wandb.init(project=config["inference"]["wandb_project"], entity=config["inference"]["wandb_entity"])

    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")

    inference(data_path, output_dir, model, model_weights, device, WANDB)
    