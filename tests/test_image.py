import pytest
import tifffile as tff
import fibsem.fibsemImage as fb
import numpy as np
import os

def test_image():
    array1 = np.random.rand(32,32)
    img = fb.fibsemImage(array1)
    img.save_to_TIFF('test.tif')
    img.load_from_TIFF('test.tif')
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 32
    assert img.data.shape[1] == 32
    
def test_metadata():
     array1 = np.random.rand(32,32)
    img = fb.fibsemImage(array1)
    img.save_to_TIFF('test.tif')
    img.load_from_TIFF('test.tif')
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 32
    assert img.data.shape[1] == 32
    
    pass


