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


# NOTE: these models contain numeric training bugs that were fixed in later versions. For reproducibility, we need to re-implement the bug for these models...
# please contact (pat) if these models arent working as expected. 
__DEPRECIATED_CHECKPOINTS__ = [
    "autolamella-02-34.pt",
    "autolamella-03-34.pt",
    "autolamella-04-34.pt",
    "autolamella-05-34.pt",
    "autolamella-latest.pt",
    "openfibsem-01-18.pt",
    "openfibsem-02-18.pt",
    "openfibsem-03-18.pt",
    "openfibsem-baseline-34.pt"
    "openfibsem-baseline-latest.pt",
    "autoliftout-serial-01-34.pt",
]


class SegmentationModel:
    def __init__(
        self,
        checkpoint: str = None,
        encoder: str = "resnet34",
        mode: str = "eval",
        num_classes: int = 3,
        _fix_numeric_scaling: bool = True,
        use_v2: bool = True,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self._fix_numeric_scaling = _fix_numeric_scaling

        if checkpoint in __DEPRECIATED_CHECKPOINTS__:
            self._fix_numeric_scaling = False

        if "latest" in checkpoint:
            use_v2 = True
        if use_v2:
            print(f"using latest checkpoint: {checkpoint}")
            self.model = self.load_model_v2(checkpoint=checkpoint)
        else:
            self.load_model(checkpoint=checkpoint, encoder=encoder)


    def load_model(self, checkpoint: Optional[str], encoder: str = "resnet18") -> None:
        """Load the model, and optionally load a checkpoint"""
        self.model = self.load_encoder(encoder=encoder)
        self.load_weights(checkpoint=checkpoint)
        if self._fix_numeric_scaling:
            self.model.eval() # this causes a bug? why -> input needs to be scaled between 0-1
        if self.mode == "train":
            self.model.train()

    def load_model_v2(self, checkpoint: str):

        if os.path.exists(checkpoint):
            checkpoint = checkpoint
        else:
            REPO_ID = "patrickcleeve/openfibsem-baseline"
            checkpoint = hf_hub_download(repo_id=REPO_ID, filename=checkpoint)
        
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        self.model = smp.Unet(
            encoder_name=checkpoint_dict["encoder"],
            encoder_weights="imagenet",
            in_channels=1,  # grayscale images
            classes=checkpoint_dict["nc"],
        )
        self.model.to(self.device)

        self.model.load_state_dict(checkpoint_dict["checkpoint"])

        if self._fix_numeric_scaling:
            self.model.eval() # this causes a bug? why -> input needs to be scaled between 0-1
        if self.mode == "train":
            self.model.train()

        # metadata
        self.checkpoint = checkpoint
        self.num_classes = checkpoint_dict["nc"]
        self.encoder = checkpoint_dict["encoder"]

        return self.model

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
        # print(img.min(), img.max(), img.dtype)
        img_t = torch.Tensor(img).float().to(self.device)
        # print values range
        # print(img_t.min(), img_t.max(), img_t.dtype)
        if self._fix_numeric_scaling:
            img_t /=  255.0 # scale float to 0 - 1
        # print(img_t.min(), img_t.max(), img_t.dtype)
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


def get_backend(checkpoint: str) -> str:

    if "nnunet" in checkpoint:
        return "nnunet"
    elif "onnx" in checkpoint:
        return "onnx"
    else:
        return "smp"

def load_model(
    checkpoint: Path, encoder: str = "resnet34", nc: int = 3, _fix_numeric_scaling: bool = True, backend = None
) -> SegmentationModel:
    """Load a model checkpoint
    backend: str, optional The backend to use. If None, will try to infer from the checkpoint name"""
    
    if backend is None:
        backend = get_backend(checkpoint=checkpoint)

    # load model
    if backend == "nnunet":
        from fibsem.segmentation.nnunet_model import SegmentationModelNNUnet
        model = SegmentationModelNNUnet(checkpoint=checkpoint)
    elif backend == "onnx":
        from fibsem.segmentation.onnx_model import SegmentationModelONNX
        model = SegmentationModelONNX(checkpoint=checkpoint)
    else:
        model = SegmentationModel(checkpoint=checkpoint, encoder=encoder, num_classes=nc, _fix_numeric_scaling=_fix_numeric_scaling)

    return model


if __name__ == "__main__":

    model = SegmentationModel(checkpoint="checkpoint_train.pth.tar", mode="train")
