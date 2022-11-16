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

class fibsemImage():
    def __init__(
        self,
        data: np.ndarray = None,
        metadata: Metadata = None
    ):  
        if data is not None:
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
                metadata_dict = self.metadata.image_settings.__to_dict__()
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
        
    def convert_adorned_to_fibsemImage(self, adorned: AdornedImage):
        self.data = adorned.data
        self.metadata
        pass

    def __construct_image_data(self, data: np.ndarray):
        '''
        Determines the image dimensions such as height, width, and bit depth and saves them as class variables.

        Inputs:
        :param data: Image data in the form of a numpy array.
        '''
        if type(data) != np.ndarray:
            raise TypeError("The input argument 'data' must be a numpy array.")
        try:
            if data.ndim == 2 and data.dtype == np.uint8:
                self.height, self.width, self.bit_depth = data.shape[0], data.shape[1], 8
            elif data.ndim == 2 and data.dtype == np.uint16:
                self.height, self.width, self.bit_depth = data.shape[0], data.shape[1], 16
            elif data.ndim == 3 and data.dtype == np.uint8 and data.shape[2] == 3:
                self.height, self.width, self.bit_depth = data.shape[0], data.shape[1], 24
            else:
                raise ValueError("The image dimensions of the input data could not be determined.")
        except:
            raise ValueError("Could not construct fibsemImage based on the input argument.")

        # Default image data form is numpy array.
        self.data = data

    def __metadata_from_dict(self, metadata):
        '''
        Fills in missing metadata with defaults if needed 

        Inputs:
        :param metadata: Image metadata in the form of a dictionary 
        '''
        #load into class 
        self.metadata = Metadata(
            image_settings = ImageSettings.__from_dict__(metadata)
        )





