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

# %%
# transformations
# transformation = transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize((1024 // 4, 1536 // 4)),
#         transforms.ToTensor(),
#     ]
# )

# %%
# class SegmentationDataset(Dataset):
#     def __init__(self, images, masks, num_classes: int, transforms=None):
#         self.images = images
#         self.masks = masks
#         self.num_classes = num_classes
#         self.transforms = transforms

#     def __getitem__(self, idx):
#         image = self.images[idx]

#         if self.transforms:
#             image = self.transforms(image)

#         mask = self.masks[idx]

#         # - the problem was ToTensor was destroying the class index for the labels (rounding them to 0-1)
#         # need to to transformation manually
#         mask = Image.fromarray(mask).resize(
#             (1536 // 4, 1024 // 4), resample=PIL.Image.NEAREST
#         )
#         mask = torch.tensor(np.asarray(mask)).unsqueeze(0)

#         return image, mask

#     def __len__(self):
#         return len(self.images)



# %%
# def load_images_and_masks_in_path(images_path, masks_path):
#     images = []
#     masks = []
#     sorted_img_filenames = sorted(glob.glob(images_path + ".tiff"))  #[-435:]
#     sorted_mask_filenames = sorted(glob.glob(masks_path + ".tiff"))  #[-435:]

#     for img_fname, mask_fname in tqdm(
#         list(zip(sorted_img_filenames, sorted_mask_filenames))
#     ):

#         image = np.asarray(Image.open(img_fname))
#         mask = np.asarray(Image.open(mask_fname))

#         images.append(image)
#         masks.append(mask)
#     return np.array(images), np.array(masks)


# def preprocess_data(data_path, num_classes=3, batch_size=25, val_size=0.2):

#     img_path = f"{data_path}/train/**/img"
#     label_path = f"{data_path}/train/**/label"
#     print(f"Loading dataset from {img_path}")

#     train_images, train_masks = load_images_and_masks_in_path(img_path, label_path)

#     # load dataset
#     seg_dataset = SegmentationDataset(
#         train_images, train_masks, num_classes, transforms=transformation
#     )

#     # train/validation splits
#     dataset_size = len(seg_dataset)
#     dataset_idx = list(range(dataset_size))
#     split_idx = int(np.floor(val_size * dataset_size))
#     train_idx = dataset_idx[split_idx:]
#     val_idx = dataset_idx[:split_idx]

#     train_sampler = SubsetRandomSampler(train_idx)
#     val_sampler = SubsetRandomSampler(val_idx)

#     train_data_loader = DataLoader(
#         seg_dataset, batch_size=batch_size, sampler=train_sampler
#     )  # shuffle=True,
#     print(f"Train dataset has {len(train_data_loader)} batches of size {batch_size}")

#     val_data_loader = DataLoader(
#         seg_dataset, batch_size=batch_size, sampler=val_sampler
#     )  # shuffle=True,
#     print(f"Validation dataset has {len(val_data_loader)} batches of size {batch_size}")

#     return train_data_loader, val_data_loader

# %%
# def save_model(model, epoch):
#     """Helper function for saving the model based on current time and epoch"""
    
#     # datetime object containing current date and time
#     now = datetime.now()
#     # format
#     dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1:02d}"
#     model_save_file = f"models/{dt_string}_model.pt"
#     torch.save(model.state_dict(), model_save_file)

#     print(f"Model saved to {model_save_file}")

# def train_model(model, device, train_data_loader, val_data_loader, epochs, DEBUG=False):
#     """ Helper function for training the model """
#     # initialise loss function and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#     total_steps = len(train_data_loader)
#     print(f"{epochs} epochs, {total_steps} total_steps per epoch")

#     # accounting
#     train_losses = []
#     val_losses = []

#     # training loop
#     for epoch in tqdm(range(epochs)):
#         print(f"------- Epoch {epoch+1} of {epochs}  --------")
        
#         train_loss = 0
#         val_loss = 0
        
#         data_loader = tqdm(train_data_loader)

#         for i, (images, masks) in enumerate(data_loader):

#             # set model to training mode
#             model.train()

#             # move img and mask to device, reshape mask
#             images = images.to(device)
#             masks = masks.type(torch.LongTensor)
#             masks = masks.reshape(
#                 masks.shape[0], masks.shape[2], masks.shape[3]
#             )  # remove channel dim
#             masks = masks.to(device)

#             # forward pass
#             outputs = model(images).type(torch.FloatTensor).to(device)
#             loss = criterion(outputs, masks)

#             # backwards pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # evaluation
#             train_loss += loss.item()
#             wandb.log({"train_loss": loss.item()})
#             data_loader.set_description(f"Train Loss: {loss.item():.04f}")

#             if i % 100 == 0:
          
#                 if DEBUG:
#                     model.eval()
#                     with torch.no_grad():

#                         outputs = model(images)
#                         output_mask = decode_output(outputs)
                        
#                         img_base = images.detach().cpu().squeeze().numpy()
#                         img_rgb = np.dstack((img_base, img_base, img_base))
#                         gt_base = decode_segmap(masks.detach().cpu().permute(1, 2, 0))

#                         wb_img = wandb.Image(img_rgb, caption="Input Image")
#                         wb_gt = wandb.Image(gt_base, caption="Ground Truth")
#                         wb_mask = wandb.Image(output_mask, caption="Output Mask")
#                         wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})
                           
        
#         val_loader = tqdm(val_data_loader)
#         for i, (images, masks) in enumerate(val_loader):
            
#             model.eval()
            
#             # move img and mask to device, reshape mask
#             images = images.to(device)
#             masks = masks.type(torch.LongTensor)
#             masks = masks.reshape(
#                 masks.shape[0], masks.shape[2], masks.shape[3]
#             )  # remove channel dim
#             masks = masks.to(device)

#             # forward pass
#             outputs = model(images).type(torch.FloatTensor).to(device)
#             loss = criterion(outputs, masks)

#             val_loss += loss.item()
#             wandb.log({"val_loss": loss.item()})
#             val_loader.set_description(f"Val Loss: {loss.item():.04f}")

#         train_losses.append(train_loss / len(train_data_loader))
#         val_losses.append(val_loss / len(val_data_loader))

#         # save model checkpoint
#         save_model(model, epoch)


#     return model


# %%
def save_model(model, epoch):
    """Helper function for saving the model based on current time and epoch"""
    
    # datetime object containing current date and time
    now = datetime.now()
    # format
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1:02d}"
    model_save_file = f"models/{dt_string}_model.pt"
    torch.save(model.state_dict(), model_save_file)

    print(f"Model saved to {model_save_file}")

def train(model, device, data_loader, criterion, optimizer, DEBUG, WANDB):
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

        if i % 100 == 0:
        
            if DEBUG and WANDB:
                model.eval()
                with torch.no_grad():
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

    return val_loss

def train_model(model, device, train_data_loader, val_data_loader, epochs, DEBUG=True, WANDB=True):
    """ Helper function for training the model """
    # initialise loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_steps = len(train_data_loader)
    print(f"{epochs} epochs, {total_steps} total_steps per epoch")

    # accounting13.73 GiB reserved in 
    train_losses = []
    val_losses = []

    # training loop
    for epoch in tqdm(range(epochs)):
        print(f"------- Epoch {epoch+1} of {epochs}  --------")
        
        train_loss = train(model, device, train_data_loader, criterion, optimizer, DEBUG, WANDB)
        val_loss = validate(model, device, val_data_loader, criterion, WANDB)
   
        train_losses.append(train_loss / len(train_data_loader))
        val_losses.append(val_loss / len(val_data_loader))

        # save model checkpoint
        # save_model(model, epoch)

    return model

# %%

# weights and biases setup
wandb.init(project="fibsem_pipeline", entity="demarcolab")

# hyperparams
num_classes = 3
batch_size = 8

wandb.config = {
    "epochs": 8,
    "batch_size": batch_size,
    "num_classes": num_classes
}

################################## LOAD DATASET ##################################
print(
    "\n----------------------- Loading and Preparing Data -----------------------"
)

data_path = "G:\\DeMarco\\train"

# train_data_loader, val_data_loader = preprocess_data(data_path, num_classes=num_classes, batch_size=batch_size)
from dataset import preprocess_data
train_data_loader, val_data_loader = preprocess_data(data_path, num_classes=num_classes, batch_size=batch_size)


print("\n----------------------- Data Preprocessing Completed -----------------------")

################################## LOAD MODEL ##################################
print("\n----------------------- Loading Model -----------------------")
# from smp
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,  # grayscale images
    classes=3,  # background, needle, lamella
)

# Use gpu for training if available else use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# load model checkpoint
# if model_checkpoint:
#     model.load_state_dict(torch.load(model_checkpoint, map_location=device))
#     print(f"Checkpoint file {model_checkpoint} loaded.")

################################## SANITY CHECK ##################################
print("\n----------------------- Begin Sanity Check -----------------------\n")

# for i in range(2):
#     # testing dataloader
#     imgs, masks = next(iter(train_data_loader))

#     # sanity check - model, imgs, masks
#     imgs = imgs.to(device)
#     output = model(imgs)
#     pred = decode_output(output)

#     print("imgs, masks, output")
#     print(imgs.shape, masks.shape, output.shape)


#     img_base = imgs.detach().cpu().squeeze().numpy()[0]
#     img_rgb = np.dstack((img_base, img_base, img_base))
#     gt_base = decode_segmap(masks[0].permute(1, 2, 0).squeeze())

#     # wb_img = wandb.Image(img_rgb, caption="Input Image")
#     # wb_gt = wandb.Image(gt_base, caption="Ground Truth")
#     # wb_mask = wandb.Image(pred, caption="Output Mask")
#     # wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})

################################## TRAINING ##################################
print("\n----------------------- Begin Training -----------------------\n")

# train model
model = train_model(model, device, train_data_loader, val_data_loader, epochs = 8, DEBUG=True, WANDB=True)

################################## SAVE MODEL ##################################

# %%


# %%


# %%


# %%


# %%


# %%



