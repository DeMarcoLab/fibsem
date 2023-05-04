import argparse
import os
import random
from datetime import datetime

import numpy as np
import segmentation_models_pytorch as smp
import torch

try:
    import wandb
except:
    pass

import yaml
from tqdm import tqdm

from fibsem.segmentation import dataset, utils


def save_model(save_dir, model, epoch):
    """Helper function for saving the model based on current time and epoch"""

    # dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1:02d}"
    model_save_file = os.path.join(save_dir, f"model.pt")
    torch.save(model.state_dict(), model_save_file)

    print(f"Model saved to {model_save_file}")


def train(model, device, data_loader, criterion, optimizer, WANDB, ui):
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

        data_loader.set_description(f"Train Loss: {loss.item():.04f}")

        idx = random.choice(np.arange(0, images.shape[0]))

        output = model(images[idx][None, :, :, :])
        output_mask = utils.decode_output(output)

        img_base = images[idx].detach().cpu().squeeze().numpy()
        gt_base = masks[idx].detach().cpu()[:, :, None].permute(2, 0, 1).numpy()
        if WANDB:
            wandb.log({"train_loss": loss.item()})

            wb_img = wandb.Image(np.dstack((img_base, img_base, img_base)), caption="Input Image")
            wb_gt = wandb.Image(utils.decode_segmap(gt_base), caption="Ground Truth")
            wb_mask = wandb.Image(utils.decode_segmap(output_mask), caption="Output Mask")
            wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})

        if ui:
            ui.emit({"stage": "train", "train_loss": loss.item(), "image": img_base, "pred": output_mask, "gt": gt_base})


    return train_loss


def validate(model, device, data_loader, criterion, WANDB, ui):
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

        val_loader.set_description(f"Val Loss: {loss.item():.04f}")

        output = model(images[0][None, :, :, :])
        output_mask = utils.decode_output(output)

        img_base = images[0].detach().cpu().squeeze().numpy()
        gt_base = masks[0].detach().cpu()[:, :, None].permute(2, 0, 1).numpy()

        if WANDB:
            wandb.log({"val_loss": loss.item()})

            wb_img = wandb.Image(np.dstack((img_base, img_base, img_base)), caption="Validation Input Image")
            wb_gt = wandb.Image(utils.decode_segmap(gt_base), caption="Validation Ground Truth")
            wb_mask = wandb.Image(utils.decode_segmap(output_mask), caption="Validation Output Mask")
            wandb.log(
                {
                    "Validation image": wb_img,
                    "Validation mask": wb_mask,
                    "Validation ground_truth": wb_gt,
                }
            )

        if ui:
            ui.emit({"stage": "val", "val_loss": loss.item(), "image": img_base, "pred": output_mask, "gt": gt_base})


    return val_loss

# initialise loss function and optimizer
def multi_loss(pred, target) -> float:
    c_loss = torch.nn.CrossEntropyLoss()
    d_loss = smp.losses.DiceLoss(mode="multiclass")
    f_loss = smp.losses.FocalLoss(mode="multiclass")
    return c_loss(pred, target) + d_loss(pred, target) + f_loss(pred, target)


def train_model(
    model,
    device,
    optimizer,
    train_data_loader,
    val_data_loader,
    epochs,
    save_dir,
    WANDB=True,
    ui = None, # pyqtSignal
):
    """Helper function for training the model"""

    criterion = multi_loss

    total_steps = len(train_data_loader)
    print(f"{epochs} epochs, {total_steps} total_steps per epoch")

    if ui:
        # emit signal to update ui
        ui.emit({"stage": "end", 
                 "epoch": 0, "epochs": epochs, 
                 "train_loss": 0.0, "val_loss": 0.0})

    train_losses = []
    val_losses = []

    # training loop
    for epoch in range(epochs):
        print(f"------- Epoch {epoch+1} of {epochs}  --------")

        train_loss = train(
            model, device, train_data_loader, criterion, optimizer, WANDB, ui
        )
        val_loss = validate(model, device, val_data_loader, criterion, WANDB, ui)

        train_losses.append(train_loss / len(train_data_loader))
        val_losses.append(val_loss / len(val_data_loader))
        
        # only save if val_loss is minimum
        if val_loss / len(val_data_loader) == min(val_losses):
            save_model(save_dir, model, epoch)

        # TODO: add better ui updates
        if ui:
            # emit signal to update ui
            ui.emit({"stage": "end", "epoch": epoch, "epochs": epochs, "train_loss": train_losses[-1], "val_loss": val_losses[-1]})

    return model


def _setup_model(config: dict) -> tuple:
    # from smp
    model = smp.Unet(
        encoder_name=config["encoder"],
        encoder_weights="imagenet",
        in_channels=1,  # grayscale images
        classes=int(config["num_classes"]),  # background, needle, lamella
    )

    # Use gpu for training if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load model checkpoint
    if config["checkpoint"]:
        model.load_state_dict(torch.load(config["checkpoint"], map_location=device))
        print(f"Checkpoint file {config['checkpoint']} loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    return model, optimizer, device

def _setup_dataset(config:dict):

    train_data_loader, val_data_loader = dataset.preprocess_data(
        data_path = config["data_path"], 
        label_path= config["label_path"], 
        num_classes=config["num_classes"], 
        batch_size=config["batch_size"]
    )

    return train_data_loader, val_data_loader


def _setup_wandb(config:dict):
    # other parameters
    WANDB = config["wandb"]
    if WANDB:
        # weights and biases setup
        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
        )

        wandb.config = {
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "num_classes": config["num_classes"],
        }

def main(config: dict):

    utils.validate_config(config)

    _setup_wandb(config)

    ################################## LOAD DATASET ##################################
    train_data_loader, val_data_loader = _setup_dataset(config)

    ################################## LOAD MODEL ##################################

    model, optimizer, device = _setup_model(config)
        
    ################################## TRAINING ##################################

    # train model
    model = train_model(
        model,
        device,
        optimizer,
        train_data_loader,
        val_data_loader,
        epochs=config["epochs"],
        save_dir=config["save_path"],
        WANDB=config["wandb"],
        ui=None
    )

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="the directory containing the config file to use",
        dest="config",
        action="store",
        default=os.path.join(os.path.join(os.path.dirname(__file__), "config.yml")),
    )
    args = parser.parse_args()
    config_dir = args.config

    # NOTE: Setup your config.yml file
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)

    main(config)    