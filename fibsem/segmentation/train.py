#!/usr/bin/env python3

import argparse
from datetime import datetime

# import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from validate_config import validate_config
import torch
from dataset import *
from model_utils import *
from tqdm import tqdm
import wandb
import yaml

def save_model(model, epoch, save_dir):
    """Helper function for saving the model based on current time and epoch"""
    
    # datetime object containing current date and time
    now = datetime.now()
    # format
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1:02d}"
    model_save_file = os.path.join(save_dir, f"{dt_string}_model.pt")
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
        print(masks.shape)
        masks = masks.to(device)
        # print(np.unique(masks[0]))

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

        if i % 100 == 0:
            if WANDB:
                model.eval()
                with torch.no_grad():

                    outputs = model(images)
                    output_mask = decode_output(outputs)
                    
                    img_base = images.detach().cpu().squeeze().numpy()
                    img_rgb = np.dstack((img_base, img_base, img_base))
                    gt_base = decode_segmap(masks.detach().cpu().permute(1, 2, 0))

                    wb_img = wandb.Image(img_rgb, caption="Input Image")
                    wb_gt = wandb.Image(gt_base, caption="Ground Truth")
                    wb_mask = wandb.Image(output_mask, caption="Output Mask")
                    wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})


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
        print(np.unique(masks[0].compute())) 
        masks = masks.to(device)

        # forward pass
        outputs = model(images).type(torch.FloatTensor).to(device)
        loss = criterion(outputs, masks)

        val_loss += loss.item()
        if WANDB:
            wandb.log({"val_loss": loss.item()})
            val_loader.set_description(f"Val Loss: {loss.item():.04f}")


def train_model(model, device, optimizer, train_data_loader, val_data_loader, epochs, save_dir, WANDB=True):
    """ Helper function for training the model """
    # initialise loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    total_steps = len(train_data_loader)
    print(f"{epochs} epochs, {total_steps} total_steps per epoch")

    # training loop
    for epoch in tqdm(range(epochs)):
        print(f"------- Epoch {epoch+1} of {epochs}  --------")
        
        train(model, device, train_data_loader, criterion, optimizer, WANDB)
        validate(model, device, val_data_loader, criterion, WANDB)

        # save model checkpoint
        save_model(model, epoch, save_dir)

    return model


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="the directory containing the config file to use",
        dest="config",
        action="store",
        default=os.path.join("fibsem", "segmentation", "lachie_config.yml")
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
        encoder_name=config["train"]["encoder"],
        encoder_weights="imagenet",
        in_channels=1,  # grayscale images
        classes=num_classes,  # background, needle, lamella
    )
    if config["train"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["train"]["learning_rate"])

    # Use gpu for training if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
    model.to(device)

    # load model checkpoint
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        print(f"Checkpoint file {model_checkpoint} loaded.")

    ################################## SANITY CHECK ##################################
    print("\n----------------------- Begin Sanity Check -----------------------\n")

    for i in range(2):
        # testing dataloader
        imgs, masks = next(iter(train_data_loader))
        #print(np.unique(masks)) 
        
        # sanity check - model, imgs, masks
        imgs = imgs.to(device)
        output = model(imgs)
        pred = decode_output(output)
<<<<<<< HEAD
=======
        print(pred.shape)
>>>>>>> ca8a3b1ef9078cdddcc68fdc723b7813f137de5b
        print("imgs, masks, output")
        print(imgs.shape, masks.shape, output.shape)

        if WANDB:
            for i in range(batch_size):
                img_base = imgs[i].detach().cpu().squeeze().numpy()[0]
                img_rgb = np.dstack((img_base, img_base, img_base))
                gt_base = decode_segmap(masks[i].permute(1, 2, 0).squeeze())

                wb_img = wandb.Image(img_rgb, caption="Input Image")
                wb_gt = wandb.Image(gt_base, caption="Ground Truth")
                wb_mask = wandb.Image(pred[i], caption="Output Mask")
                wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})

    ################################## TRAINING ##################################
    print("\n----------------------- Begin Training -----------------------\n")

    # train model
    model = train_model(model, device, optimizer, train_data_loader, val_data_loader, epochs, save_dir, WANDB=WANDB)

    ################################## SAVE MODEL ##################################
    

# ref:
# https://towardsdatascience.com/train-a-lines-segmentation-model-using-pytorch-34d4adab8296
# https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/14
# https://github.com/qubvel/segmentation_models.pytorch
