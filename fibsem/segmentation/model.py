from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from fibsem.segmentation.utils import decode_segmap

from pathlib import Path
from fibsem import config as cfg
import os
from huggingface_hub import hf_hub_download


class SegmentationModel:
    def __init__(
        self,
        checkpoint: str = None,
        encoder: str = "resnet18",
        mode: str = "eval",
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        self.load_model(checkpoint=checkpoint, encoder=encoder)

    def load_model(self, checkpoint: Optional[str], encoder: str = "resnet18") -> None:
        """Load the model, and optionally load a checkpoint"""
        self.model = self.load_encoder(encoder=encoder)
        self.load_weights(checkpoint=checkpoint)
        # self.model.eval() # TODO: this causes a bug? why
        if self.mode == "train":
            self.model.train()

    def load_encoder(self, encoder: str = "resnet18"):
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,  # grayscale images
            classes=self.num_classes,
        )
        model.to(self.device)
        return model

    def load_weights(self, checkpoint: Optional[str]):
        if checkpoint:
            

            # TODO, STORE CONFIG INFO IN HF (e.g. num_classes, encoder)
            # check if checkpoint is an actual path, otherwise load from HF
            if os.path.exists(checkpoint):
                checkpoint = checkpoint
            else:
                REPO_ID = "patrickcleeve/openfibsem-baseline"
                checkpoint = hf_hub_download(repo_id=REPO_ID, filename=checkpoint)

            self.checkpoint = checkpoint
            checkpoint_state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint_state)

    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process the image for inference"""

        # TODO: this is a hack, fix it
        img_t = torch.Tensor(img).float().to(self.device)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)  # add batch dim and channel dim
        elif img_t.ndim == 3:
            if img_t.shape[0] > 1:  # means the first dim is batch dim
                img_t = img_t.unsqueeze(1)  # add channel dim
            else:
                img_t = img_t.unsqueeze(0)  # add batch dim

        return img_t

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""
        with torch.no_grad():
            img_t = self.pre_process(img)

            outputs = self.model(img_t)
            outputs = F.softmax(outputs, dim=1)
            masks = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        
        # decode to rgb
        if rgb:
            masks = self.postprocess(masks, nc=self.num_classes)

        return masks

    def postprocess(self, masks, nc):
        # TODO: vectorise this properly
        output_masks = []
        for i in range(len(masks)):
            output_masks.append(decode_segmap(masks[i], nc=nc))

        if len(output_masks) == 1:
            output_masks = output_masks[0]
        return np.array(output_masks)


def load_model(
    checkpoint: Path, encoder: str = "resnet18", nc: int = 3
) -> SegmentationModel:

    # load model
    model = SegmentationModel(checkpoint=checkpoint, encoder=encoder, num_classes=nc)

    return model


if __name__ == "__main__":

    model = SegmentationModel(checkpoint="checkpoint_train.pth.tar", mode="train")
