
from typing import Optional, List

import numpy as np
import torch

from fibsem.segmentation.utils import decode_segmap_v2

from pathlib import Path
from fibsem.segmentation import config as scfg
import os
from huggingface_hub import hf_hub_download


from fibsem.segmentation import _nnunet as nnunet

# TODO: actually implement this
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
        self.colormap = scfg.get_colormap()
       
        self.checkpoint = checkpoint
        if checkpoint is not None:
            self.load_model(checkpoint=checkpoint)

    def load_model(self, checkpoint: str) -> None:
        """Load the model, and optionally load a checkpoint"""
        
        self.model = nnunet.load_model(path=checkpoint)

        return self.model

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Pre-process the image for model inference"""

        # convert to 4D
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        elif img.ndim == 3:
            img = img[np.newaxis, :, :, :]
        elif img.ndim == 4:
            img = img[:, :, :, :]
        else:
            raise ValueError(f"Invalid image shape: {img.shape}")
        
        # TODO: also do dtype conversions
        if not isinstance(img.dtype, np.float32):
            img = img.astype(np.float32)

        return img

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""
        
        img = self.pre_process(img)
        
        masks, scores = nnunet.inference(self.model, img)
        
        # convert to rgb image
        if rgb:
            masks = decode_segmap_v2(masks[0], self.colormap) # 2d only
        else:
            if masks.ndim>=3:
                masks = masks[0] # return 2d
        return masks

    def postprocess(self, masks):
        """Convert the model output to a rgb class map"""
        if masks.ndim == 3:
            masks = masks[0]
        return decode_segmap_v2(masks, self.colormap)

     