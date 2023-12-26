
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F


from fibsem.segmentation.utils import decode_segmap

from pathlib import Path
from fibsem import config as cfg
import os
from huggingface_hub import hf_hub_download


from fibsem.segmentation import _nnunet as nnunet

from abc import ABC
def SegmentationModelBase(ABC):

    def __init__(self, checkpoint: str):
        pass 


    def load_model(self, checkpoint: str) -> SegmentationModelBase:
        """Load the model, and optionally load a checkpoint"""
        raise NotImplementedError
    
    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""
        raise NotImplementedError
    
class SegmentationModelNNUnet:
    def __init__(
        self,
        checkpoint: str = None,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.colormap = nnunet.get_colormap()
       
        self.checkpoint = checkpoint
        if checkpoint is not None:
            self.load_model(checkpoint=checkpoint)

    def load_model(self, checkpoint: str) -> None:
        """Load the model, and optionally load a checkpoint"""
        
        self.model = nnunet.load_model(path=checkpoint)

        return self.model

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""
        masks, scores = nnunet.inference(self.model, img)
        
        # convert to rgb image
        if rgb:
            masks = decode_segmap_v2(masks[0, self.colormap]) # 2d only
        else:
            if masks.ndim>=3:
                masks = masks[0] # return 2d
        return masks

    def postprocess(self, masks):
        """Convert the model output to a rgb class map"""
        return decode_segmap_v2(masks)

        


def load_model_v2(
    checkpoint: str = None,
    base: str = "smp",
) -> SegmentationModelBase:
    """Load the model, and optionally load a checkpoint"""

    if base == "smp":
        # load model
        from fibsem.segmentation.model import SegmentationModel
        model = SegmentationModel(checkpoint=checkpoint)
    elif base == "nnunet":
        model = SegmentationModelNNUnet(checkpoint=checkpoint)

    return model


def decode_segmap_v2(image, colormap: list[tuple] = None) -> np.ndarray:
    """
    Decode segmentation class mask into an RGB image mask
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    if colormap is None:
        from fibsem.segmentation.config import CLASS_COLORS_RGB
        colormap = CLASS_COLORS_RGB

    # convert class masks to rgb values
    rgb_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(image)

    for class_idx in unique_labels:
        idx = image == class_idx
        rgb = colormap[class_idx]
        rgb_mask[idx] = rgb

    return rgb_mask