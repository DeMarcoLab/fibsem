import fibsem
import numpy as np
import tifffile as tff
import os
from dataclasses import dataclass
import json
from fibsem.structures import BeamType, GammaSettings, ImageSettings
from autoscript_sdb_microscope_client.structures import AdornedImage

@dataclass
class Metadata:
    image_settings: ImageSettings

    def __to_dict__(self):
        settings_dict = self.image_settings.__to_dict__()
        return settings_dict

class fibsemImage():
    def __init__(
        self,
        data: np.ndarray = None,
        metadata: Metadata = None
    ):  
        if data is not None:
            self.check_data(data)
            self.data = data
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    def load_from_TIFF(self, tiff_path):
        tiff_image = tff.TiffFile(tiff_path)
        self.data = tiff_image.asarray()
        try:
            metadata = json.loads(tiff_image.pages[0].tags["ImageDescription"].value) 
            self.metadata = Metadata(
                image_settings=ImageSettings.__from_dict__(metadata)                
            )
        except:
            self.metadata = None

    def save_to_TIFF(self, save_path: str):
        if self.metadata is not None:
                metadata_dict = self.metadata.__to_dict__()
                tff.imwrite(
                    save_path, 
                    self.data,
                    metadata=metadata_dict,
                )
        else:
                tff.imwrite(
                    save_path,
                    self.data,
                    metadata=None,
                )
        
    def convert_adorned_to_fibsemImage(self, adorned: AdornedImage, metadata: ImageSettings = None):
        self.data = adorned.data
        self.check_data(self.data)
        self.metadata  = metadata

    def check_data(self, data):
        assert data.ndim == 2 or data.ndim == 3
        assert data.dtype == np.uint8
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:,:,0]
            