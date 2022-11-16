import pytest
import tifffile as tff
import fibsem.fibsemImage as fb
import numpy as np
import os
from fibsem.structures import BeamType, GammaSettings, ImageSettings


def test_image():
    array1 = np.random.rand(32,32)
    img = fb.fibsemImage(array1)
    img.save_to_TIFF('test.tif')
    img.load_from_TIFF('test.tif')
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 32
    assert img.data.shape[1] == 32
    
def test_metadata():
    array1 = np.zeros((256,128)) 
    gamma_dict = {
        "enabled": True,
        "min_gamma": 0.5,
        "max_gamma": 1.8,
        "scale_factor": 0.01,
        "threshold": 46
    }
    Image_settings = {
        "resolution": "32x32",
        "dwell_time": 1.e-6,
        "hfw": 150.e-6,
        "autocontrast": True,
        "beam_type": "ELECTRON", 
        "gamma": gamma_dict,
        "save": False,
        "save_path": "path",
        "label": "label"
    }

    metadata = fb.Metadata(
            image_settings = ImageSettings.__from_dict__(Image_settings)
        )
    img = fb.fibsemImage(array1, metadata)
    img.save_to_TIFF('test.tif')
    img.load_from_TIFF('test.tif')
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 256
    assert img.data.shape[1] == 128
    assert img.metadata == metadata
    
    pass


