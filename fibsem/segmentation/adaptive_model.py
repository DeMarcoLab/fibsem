import logging
import numpy as np
import torch
from adaptive_polish.dl_segmentation import sem_lamella_segmentor as sgm
from fibsem.segmentation.utils import decode_segmap_v2, download_checkpoint


class AdaptiveSegmentationModel:
    def __init__(
        self,
        checkpoint: str = None,
        mode: str = "eval",
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.checkpoint: str = checkpoint
        self.mode = mode
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model_v2(checkpoint=checkpoint)
        self.device = self.model._device

    def load_model_v2(self, checkpoint: str):
        model = sgm.cSEMLamellaSegmentor(model_path=checkpoint)
        logging.debug(f"Loaded {self.__class__}, model from {checkpoint}")
        return model

    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process the image for inference"""
        return img

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""

        # NOTE: currently all pre-processing is done in adaptive_polish.sem_lamella_segmentor.cSEMLamellaSegmentor
        # TODO: migrate pre-processing to this class
        masks = self.model.get_prediction(img)
        # decode to rgb
        if rgb:
            masks = self.postprocess(masks)
        return masks

    def postprocess(self, masks: np.ndarray) -> np.ndarray:
        return decode_segmap_v2(masks)
