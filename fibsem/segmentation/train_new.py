# %% [markdown]
# Imports

# %%
import glob

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
import argparse
from datetime import datetime
    
# import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from dataset import *
from model_utils import *
from tqdm import tqdm
import wandb
import random
import yaml
from validate_config import *

# %%
def save_model(save_dir, model, epoch):
    """Helper function for saving the model based on current time and epoch"""
    
    # datetime object containing current date and time
    now = datetime.now()
    # format
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1:02d}"
    model_save_file = f"{save_dir}/{dt_string}_model.pt"
    torch.save(model.state_dict(), model_save_file)

    print(f"Model saved to {model_save_file}")

def train(model, device, data_loader, criterion, optimizer, WANDB):
    data_loader = tqdm(data_loader)
    train_loss = 0

    for i, (images, masks) in enumerate(data_loader):
        # set model to training mode
        model.train()

        # move img and mask to device, reshape mask
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        masks = masks.reshape(
            masks.shape[0], masks.shape[2], masks.shape[3]
        )  # remove channel dim
        masks = masks.to(device)

        # forward pass
        outputs = model(images).type(torch.FloatTensor).to(device)
        loss = criterion(outputs, masks)

        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        train_loss += loss.item()

        if WANDB:
            wandb.log({"train_loss": loss.item()})
            data_loader.set_description(f"Train Loss: {loss.item():.04f}")

            idx = random.choice(np.arange(0, batch_size))

            output = model(images[idx][None, :, :, :])
            output_mask = decode_output(output)
            
            img_base = images[idx].detach().cpu().squeeze().numpy()
            img_rgb = np.dstack((img_base, img_base, img_base))
            gt_base = decode_segmap(masks[idx].detach().cpu()[:, :, None])  #.permute(1, 2, 0))

            wb_img = wandb.Image(img_rgb, caption="Input Image")
            wb_gt = wandb.Image(gt_base, caption="Ground Truth")
            wb_mask = wandb.Image(output_mask, caption="Output Mask")
            wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})

    return train_loss

def validate(model, device, data_loader, criterion, WANDB):
    val_loader = tqdm(data_loader)
    val_loss = 0

    for i, (images, masks) in enumerate(val_loader):
        
        model.eval()
        
        # move img and mask to device, reshape mask
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        masks = masks.reshape(
            masks.shape[0], masks.shape[2], masks.shape[3]
        )  # remove channel dim
        masks = masks.to(device)


        # forward pass
        outputs = model(images).type(torch.FloatTensor).to(device)
        loss = criterion(outputs, masks)

        val_loss += loss.item()
        if WANDB:
            wandb.log({"val_loss": loss.item()})
            val_loader.set_description(f"Val Loss: {loss.item():.04f}")

            output = model(images[0][None, :, :, :])
            output_mask = decode_output(output)
            
            img_base = images[0].detach().cpu().squeeze().numpy()
            img_rgb = np.dstack((img_base, img_base, img_base))
            gt_base = decode_segmap(masks[0].detach().cpu()[:, :, None])  #.permute(1, 2, 0))

            wb_img = wandb.Image(img_rgb, caption="Validation Input Image")
            wb_gt = wandb.Image(gt_base, caption="Validation Ground Truth")
            wb_mask = wandb.Image(output_mask, caption="Validation Output Mask")
            wandb.log({"Validation image": wb_img, "Validation mask": wb_mask, "Validation ground_truth": wb_gt})

    return val_loss

def train_model(model, device, optimizer, train_data_loader, val_data_loader, epochs, save_dir, WANDB=True):
    """ Helper function for training the model """
    # initialise loss function and optimizer
    def multi_loss(pred, target) -> float:
        c_loss = torch.nn.CrossEntropyLoss()
        d_loss = smp.losses.DiceLoss(mode="multiclass")
        f_loss = smp.losses.FocalLoss(mode="multiclass")
        return  c_loss(pred, target) + d_loss(pred, target) + f_loss(pred, target)

    criterion = multi_loss
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = len(train_data_loader)
    print(f"{epochs} epochs, {total_steps} total_steps per epoch")

    train_losses = []
    val_losses = []

    # training loop
    for epoch in range(epochs):
        print(f"------- Epoch {epoch+1} of {epochs}  --------")
        
        train_loss = train(model, device, train_data_loader, criterion, optimizer, WANDB)
        val_loss = validate(model, device, val_data_loader, criterion, WANDB)
   
        train_losses.append(train_loss / len(train_data_loader))
        val_losses.append(val_loss / len(val_data_loader))

        # save model checkpoint
        save_model(save_dir, model, epoch)

    return model


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="the directory containing the config file to use",
        dest="config",
        action="store",
        default=os.path.join("fibsem", "segmentation", "config.yml")
    )
    args = parser.parse_args()
    config_dir = args.config

    # NOTE: Setup your config.yml file
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)

    print("Validating config file.")
    validate_config(config, "train")

    # directories
    data_path = config["train"]["data_dir"]
    model_checkpoint = config["train"]["checkpoint"]
    save_dir = config["train"]["save_dir"]

    # hyper-params
    epochs = config["train"]["epochs"]
    num_classes = config["train"]["num_classes"] # Includes background class
    batch_size = config["train"]["batch_size"]

    # other parameters
    cuda = config["train"]["cuda"]
    WANDB = config["train"]["wandb"]
    if WANDB:
        # weights and biases setup
        wandb.init(project=config["train"]["wandb_project"], entity=config["train"]["wandb_entity"])

        wandb.config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_classes": num_classes
    }

    ################################## LOAD DATASET ##################################
    print(
        "\n----------------------- Loading and Preparing Data -----------------------"
    )

    train_data_loader, val_data_loader = preprocess_data(data_path, num_classes=num_classes, batch_size=batch_size)

    print("\n----------------------- Data Preprocessing Completed -----------------------")

    ################################## LOAD MODEL ##################################
    print("\n----------------------- Loading Model -----------------------")
    # from smp
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,  # grayscale images
        classes=num_classes,  # background, needle, lamella
    )

    # Use gpu for training if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
    model.to(device)

    # load model checkpoint
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        print(f"Checkpoint file {model_checkpoint} loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    ################################## TRAINING ##################################
    print("\n----------------------- Begin Training -----------------------\n")

    # train model
    model = train_model(model, device, optimizer, train_data_loader, val_data_loader, epochs = epochs, save_dir=save_dir, WANDB=True)

# config["train"]["learning_rate"]

