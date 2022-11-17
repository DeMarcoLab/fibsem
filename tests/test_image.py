import pytest
import tifffile as tff
import fibsem.fibsemImage as fb
import numpy as np
import os
from fibsem.structures import BeamType, GammaSettings, ImageSettings
from autoscript_sdb_microscope_client.structures import AdornedImage


def test_image():
    array1 = np.uint8(255*np.random.rand(32,32))
    img = fb.fibsemImage(array1)
    img.save_to_TIFF('test.tif')
    img.load_from_TIFF('test.tif')
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 32
    assert img.data.shape[1] == 32
    assert img.data.all() >= 0
    assert img.data.all() <= 255
    assert img.data.dtype == np.uint8
    
def test_metadata():
    array1 = np.uint8(np.zeros((256,128)))
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
    
def test_adorned_image_conversion():
    array1 = np.uint8(255*np.random.rand(32,32))
    img1 = fb.fibsemImage(array1)
    adorned = AdornedImage(array1)
    img2 = fb.fibsemImage()
    img2.convert_adorned_to_fibsemImage(adorned)
    assert np.array_equal(img1.data, img2.data)

def test_adorned_metadata_conversion():
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
    image_settings = ImageSettings.__from_dict__(Image_settings)
    metadata = fb.Metadata(
            image_settings
        )

    array1 = np.uint8(255*np.random.rand(32,32))
    img1 = fb.fibsemImage(array1, metadata=metadata)
    adorned = AdornedImage(array1)
    img2 = fb.fibsemImage()
    img2.convert_adorned_to_fibsemImage(adorned, metadata=metadata)
    assert img1.metadata == img2.metadata


