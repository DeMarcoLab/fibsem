import numpy as np
import torch

from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
from fibsem.segmentation import config as scfg
from fibsem.segmentation.utils import decode_segmap_v2

class SegmentationModelHuggingFace:
    """HuggingFace model for semantic segmentation"""
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
        self.num_classes = 6 # TODO: tmp required, remove

    def load_model(self, checkpoint: str) -> None:
        """Load the model, and optionally load a checkpoint"""
        
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)

        return self.model

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Pre-process the image for model inference"""

        # assume image is 2D grayscale 
        image = np.asarray(Image.fromarray(np.asarray(img)).convert("RGB"))
        inputs = self.processor(images=image, return_tensors="pt")

        return inputs

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""
        
        inputs = self.pre_process(img)
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.shape, # (height, width)
            mode='bilinear',
            align_corners=False
        )

        # Second, apply argmax on the class dimension
        masks = upsampled_logits.argmax(dim=1).detach().cpu().numpy()

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

     