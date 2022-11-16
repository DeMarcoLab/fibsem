import fibsem
import numpy as np
import tifffile as tff
import os
from dataclasses import dataclass
from fibsem.structures import BeamType, GammaSettings, ImageSettings


class fibsemImage():
    def __init__(
        self,
        data: np.ndarray = None,
        metadata: dict = None
    ):  
        if data is not None:
            self.data = data
            #self.__construct_image_data(data)
        if metadata is not None:
            self.__metadata_from_dict(metadata)
        else:
            self.metadata = None

    def load_from_TIFF(self, tiff_path):
        tiff_image = tff.TiffFile(tiff_path)
        data = tiff_image.asarray()
        self.data = data
        # self.__construct_image_data(data)
        self.metadata = tiff_image.imagej_metadata
        # self.__metadata_from_dict(metadata)    

    def save_to_TIFF(self, save_path: str):
        if self.metadata is not None:
            if save_path is not None:
                tff.imwrite(
                    os.path.join(save_path), # check that
                    self.data,
                    metadata=self.metadata,
                )
            else:
                raise TypeError("No save path provided.")
        else:
            if save_path is not None:
                tff.imwrite(
                    os.path.join(save_path), # check that
                    self.data,
                    metadata=None,
                )
            else:
                raise TypeError("No save path provided.")
        


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

@dataclass
class Metadata:
    image_settings: ImageSettings



