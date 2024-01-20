from typing import Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from fibsem.segmentation.utils import decode_segmap, download_checkpoint

from pathlib import Path
import os



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

        self.checkpoint: str = checkpoint
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self._fix_numeric_scaling = _fix_numeric_scaling

        if checkpoint in __DEPRECIATED_CHECKPOINTS__:
            self._fix_numeric_scaling = False

        if use_v2:
            self.model = self.load_model_v2(checkpoint=checkpoint)
        else:
            self.load_model(checkpoint=checkpoint, encoder=encoder)

    # TODO: deprecate this fully
    def load_model(self, checkpoint: Optional[str], encoder: str = "resnet18") -> None:
        """Load the model, and optionally load a checkpoint"""

        # show depreciation warning
        print(f"WARNING: {checkpoint} is a depreciated checkpoint. Please use the latest checkpoint instead.")

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,  # grayscale images
            classes=self.num_classes,
        )
        self.model.to(self.device)        
        
        
        self.load_weights(checkpoint=checkpoint)
        if checkpoint:
            checkpoint = download_checkpoint(checkpoint)

            self.checkpoint = checkpoint
            checkpoint_state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint_state)
        if self._fix_numeric_scaling:
            self.model.eval() # this causes a bug? why -> input needs to be scaled between 0-1
        if self.mode == "train":
            self.model.train()

    def load_model_v2(self, checkpoint: str):

        # download checkpoint if needed
        checkpoint = download_checkpoint(checkpoint)
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
        self.checkpoint = os.path.basename(checkpoint)
        self.num_classes = checkpoint_dict["nc"]
        self.encoder = checkpoint_dict["encoder"]

        return self.model


    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process the image for inference"""
        img_t = torch.Tensor(img).float().to(self.device)

        if self._fix_numeric_scaling:
            img_t /=  255.0 # scale float to 0 - 1
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)  # add batch dim and channel dim
        elif img_t.ndim == 3:
            if img_t.shape[0] > 1:  # means the first dim is batch dim
                img_t = img_t.unsqueeze(1)  # add channel dim
            else:
                img_t = img_t.unsqueeze(0)  # add batch dim

        assert img_t.ndim == 4, f"Expected 4 dims, got {img_t.ndim}"

        logging.debug({"msg": "pre_process", "shape": img_t.shape, "dtype": img_t.dtype, "min": img_t.min(), "max": img_t.max()})

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

        # TODO: return masks, scores, logits
        return masks

    def postprocess(self, masks, nc):
        # TODO: vectorise this properly
        # TODO: use decode_segmap_v2
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
