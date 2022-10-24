from typing import Optional, List

import numpy as np
import torch

import utils
import segmentation_models_pytorch as smp


class SegmentationModel:
    def __init__(
        self, checkpoint: str = None, mode: str = "eval", num_classes: int = 3
    ) -> None:
        super().__init__()

        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model(checkpoint=checkpoint, num_classes=num_classes)
        # TODO: inference transforms?

    def load_model(self, checkpoint: Optional[str], num_classes: int) -> None:
        """Load the model, and optionally load a checkpoint"""
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,  # grayscale images
            classes=num_classes,  # num_layers meaning number of classes? Or depth of the model?
        )
        self.model.to(self.device)

        if checkpoint:
            checkpoint_state = torch.load(checkpoint, map_location=self.device)

            self.model.load_state_dict(checkpoint_state["state_dict"])
            self.model.eval()

            if self.mode == "train":
                # TODO: pass state to optimizer
                self.model.train()

    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process the image for inference"""

        img_t = torch.Tensor(img).float().to(self.device)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)  # add batch dim and channel dim
        elif img_t.ndim == 3:
            if img_t.shape[0] > 1: # means the first dim is batch dim
                img_t = img_t.unsqueeze(1) # add channel dim
            else:
                img_t = img_t.unsqueeze(0) # add batch dim

        return img_t

    def inference(self, img: np.ndarray) -> np.ndarray:
        """Run model inference on the input image"""
        with torch.no_grad():
            img_t = self.pre_process(img)
            masks = []

            for i in range(img_t.shape[0]):
                # inference
                output = self.model(img_t[i][None, :, :, :]).detach()
                output_mask = utils.decode_output(output)
                masks.append(output_mask)

            return masks


if __name__ == "__main__":

    model = SegmentationModel(checkpoint="checkpoint_train.pth.tar", mode="train")