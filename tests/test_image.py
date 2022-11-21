import pytest
import tifffile as tff
import fibsem.fibsemImage as fb
import numpy as np
import json
from matplotlib import pyplot as plt
from fibsem.structures import GammaSettings, ImageSettings, BeamType
from fibsem.config import METADATA_VERSION

THERMO_ENABLED = True
if THERMO_ENABLED:
    from autoscript_sdb_microscope_client.structures import AdornedImage


@pytest.fixture
def gamma_settings() -> GammaSettings:

    gamma_settings = GammaSettings(
        enabled=True,
        min_gamma=0.5,
        max_gamma=1.8,
        scale_factor=0.01,
        threshold=46,
    )

    return gamma_settings


@pytest.fixture
def img_settings(gamma_settings: GammaSettings) -> ImageSettings:

    image_settings = ImageSettings(
        resolution="32x32",
        dwell_time=1.0e-6,
        hfw=150.0e-6,
        autocontrast=True,
        beam_type=BeamType.ELECTRON,
        gamma=gamma_settings,
        save=False,
        save_path="path",
        label="label",
    )

    return image_settings


def test_saving_image():
    """Test saving FibsemImage data to file"""

    array1 = np.uint8(255 * np.random.rand(32, 32))
    img = fb.FibsemImage(array1)
    img.save("test.tif")
    with tff.TiffFile("test.tif") as tiff_image:
        data = tiff_image.asarray()
    assert np.array_equal(array1, data)
    assert img.data.shape == array1.shape
    assert img.data.shape[1] == 32
    assert img.data.dtype == np.uint8


def test_loading_image():
    """Test loading FibsemImage data from file"""
    array1 = np.uint8(255 * np.random.rand(32, 32))
    img = fb.FibsemImage(array1)
    img.save("test.tif")
    img.load("test.tif")
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 32
    assert img.data.shape[1] == 32
    assert img.data.dtype == np.uint8


def test_saving_metadata(img_settings):
    """Test saving FibsemImage metadata to file.

    Args:
        img_settings (fixture): fixture returning ImageSettings object
    """
    array1 = np.zeros((256, 128), dtype=np.uint8)
    metadata = fb.FibsemImageMetadata(image_settings=img_settings)
    img = fb.FibsemImage(array1, metadata)
    img.save("test.tif")
    with tff.TiffFile("test.tif") as tiff_image:
        data = tiff_image.asarray()
        metadata = json.loads(tiff_image.pages[0].tags["ImageDescription"].value)
        metadata = fb.FibsemImageMetadata(
            image_settings=ImageSettings.__from_dict__(metadata)
        )

    assert np.array_equal(array1, data)
    assert img.data.shape[0] == array1.shape[0]
    assert img.data.shape[1] == array1.shape[1]
    assert img.metadata == metadata


def test_loading_metadata(img_settings):
    """Test loading FibsemImage metadata from file.
    Args:
        img_settings (fixture): fixture returning ImageSettings object
    """
    array1 = np.uint8(np.zeros((256, 128)))
    metadata = fb.FibsemImageMetadata(image_settings=img_settings)
    img = fb.FibsemImage(array1, metadata)
    img.save("test.tif")
    img.load("test.tif")
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == array1.shape[0]
    assert img.data.shape[1] == array1.shape[1]
    assert img.metadata == metadata
    assert img.metadata.version == METADATA_VERSION


def test_getting_data_from_adorned_image():
    """Test getting data from an adorned image (microscope output format)"""
    array1 = np.uint8(255 * np.random.rand(32, 32))
    img1 = fb.FibsemImage(array1)
    adorned = AdornedImage(array1)
    img2 = fb.FibsemImage.fromAdornedImage(adorned)
    assert np.array_equal(img1.data, img2.data)


def test_converting_metadata_from_adorned_image(img_settings):
    """Test getting data from an adorned image (microscope output format).
    Args:
        img_settings (fixture): fixture returning ImageSettings object
    """
    metadata = img_settings
    array1 = np.uint8(255 * np.random.rand(32, 32))
    img1 = fb.FibsemImage(array1, metadata=metadata)
    adorned = AdornedImage(array1)
    img2 = fb.FibsemImage.fromAdornedImage(adorned, metadata=metadata)
    assert img1.metadata == img2.metadata


def test_data_checks():
    """Test that FibsemImage data checks raise errors when appropriate"""
    array1 = np.uint16(255 * np.random.rand(32, 32, 32))
    with pytest.raises(Exception) as e_info:
        img = fb.FibsemImage(array1)
