import pytest
import tifffile as tff
import numpy as np
import os
import json
from datetime import datetime
from matplotlib import pyplot as plt
import fibsem.structures as fs
from fibsem.config import METADATA_VERSION

THERMO_ENABLED = True
if THERMO_ENABLED:
    from autoscript_sdb_microscope_client.structures import AdornedImage


@pytest.fixture
def gamma_settings() -> fs.GammaSettings:

    gamma_settings = fs.GammaSettings(
        enabled=True,
        min_gamma=0.5,
        max_gamma=1.8,
        scale_factor=0.01,
        threshold=46,
    )

    return gamma_settings


@pytest.fixture
def microscope_state() -> fs.MicroscopeState:

    microscope_state = fs.MicroscopeState(
        timestamp=datetime.timestamp(datetime.now()),
        absolute_position=fs.StagePosition(),
        eb_settings=fs.BeamSettings(beam_type=fs.BeamType.ELECTRON),
        ib_settings=fs.BeamSettings(beam_type=fs.BeamType.ION),
    )
    return microscope_state


@pytest.fixture
def rectangle() -> fs.FibsemRectangle:
    """Fixture for a rectangle"""
    rectangle = None #FibsemRectangle(left=0.0, top=0.0, height=1.0, width=1.0)
    return rectangle


@pytest.fixture
def image_settings(
    gamma_settings: fs.GammaSettings, rectangle: fs.FibsemRectangle
) -> fs.ImageSettings:

    # image_settings = ImageSettings(
    #     resolution=(32,32),
    #     dwell_time=1.0e-6,
    #     hfw=150.0e-6,
    #     autocontrast=True,
    #     beam_type=BeamType.ELECTRON,
    #     gamma=gamma_settings,
    #     save=False,
    #     save_path="path",
    #     label="label",
    #     reduced_area=rectangle,
    # )
    image_settings = fs.ImageSettings()
    return image_settings


@pytest.fixture
def metadata_fixture(
    image_settings, microscope_state: fs.MicroscopeState
) -> fs.FibsemImageMetadata:

    image_settings = image_settings
    version: str = METADATA_VERSION
    pixel_size: fs.Point = fs.Point(0.0, 0.0)
    microscope_state: microscope_state
    metadata = fs.FibsemImageMetadata(
        image_settings=image_settings,
        version=version,
        pixel_size=pixel_size,
        microscope_state=microscope_state,
    )

    return metadata


def test_saving_image():
    """Test saving FibsemImage data to file"""

    array1 = np.uint8(255 * np.random.rand(32, 32))
    img = fs.FibsemImage(array1)
    img.save("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images\\third.tif")
    with tff.TiffFile("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images\\third.tif") as tiff_image:
        data = tiff_image.asarray()
    assert np.array_equal(array1, data)
    assert img.data.shape == array1.shape
    assert img.data.shape[1] == 32
    assert img.data.dtype == np.uint8


def test_loading_image():
    """Test loading FibsemImage data from file"""
    array1 = np.uint8(255 * np.random.rand(32, 32))
    img = fs.FibsemImage(array1)
    img.save("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images\\second.tif")
    img.load("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images\\second.tif")
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == 32
    assert img.data.shape[1] == 32
    assert img.data.dtype == np.uint8


def test_saving_metadata(metadata_fixture):
    """Test saving FibsemImage metadata to file.

    Args:
        img_settings (fixture): fixture returning ImageSettings object
    """
    array1 = np.zeros((256, 128), dtype=np.uint8)
    metadata = fs.FibsemImageMetadata(
        metadata_fixture.image_settings,
        metadata_fixture.pixel_size,
        metadata_fixture.microscope_state,
        metadata_fixture.version,
    )
    img = fs.FibsemImage(array1, metadata)
    img.save("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images\\first")
    with tff.TiffFile("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images\\first.tif") as tiff_image:
        data = tiff_image.asarray()
        metadata = json.loads(tiff_image.pages[0].tags["ImageDescription"].value)
        metadata = fs.FibsemImageMetadata.__from_dict__(metadata)

    assert np.array_equal(array1, data)
    assert img.data.shape[0] == array1.shape[0]
    assert img.data.shape[1] == array1.shape[1]
    assert img.metadata == metadata


def test_loading_metadata(metadata_fixture):
    """Test loading FibsemImage metadata from file.
    Args:
        img_settings (fixture): fixture returning ImageSettings object
    """
    array1 = np.uint8(np.zeros((256, 128)))
    metadata = fs.FibsemImageMetadata(
        metadata_fixture.image_settings,
        metadata_fixture.pixel_size,
        metadata_fixture.microscope_state,
        metadata_fixture.version,
    )
    img = fs.FibsemImage(array1, metadata)
    save_path = os.path.join("C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\fibsem\\tests\\tests_images", f"{metadata.image_settings.label}.tif")
    img.save(save_path)
    img.load(save_path)
    print(img.metadata.pixel_size)
    assert np.array_equal(array1, img.data)
    assert img.data.shape[0] == array1.shape[0]
    assert img.data.shape[1] == array1.shape[1]
    assert img.metadata == metadata
    assert img.metadata.version == METADATA_VERSION


def test_getting_data_from_adorned_image(image_settings):
    """Test getting data from an adorned image (microscope output format)"""
    with tff.TiffFile("fibsem\\2022-11-17.01-35-21PM_ib.tif") as tiff_image:
        data = tiff_image.asarray()
    adorned = AdornedImage.load("fibsem\\2022-11-17.01-35-21PM_ib.tif")
    img1 = fs.FibsemImage(data, metadata_fixture)
    img2 = fs.FibsemImage.fromAdornedImage(adorned, image_settings)
    assert np.array_equal(img1.data, img2.data)


def test_converting_metadata_from_adorned_image(metadata_fixture, image_settings):
    """Test getting data from an adorned image (microscope output format).
    Args:
        image_settings (fixture): fixture returning ImageSettings object
        metadata_fixture (fixture): fixture returning FibsemImageMetadata object
    """
    with tff.TiffFile("fibsem\\2022-11-17.01-35-21PM_ib.tif") as tiff_image:
        data = tiff_image.asarray()
    adorned = AdornedImage.load("fibsem\\2022-11-17.01-35-21PM_ib.tif")
    img1 = fs.FibsemImage(data, metadata_fixture)
    img2 = fs.FibsemImage.fromAdornedImage(adorned, image_settings)
    assert img1.metadata.image_settings == img2.metadata.image_settings


def test_data_checks():
    """Test that FibsemImage data checks raise errors when appropriate"""
    array1 = np.uint16(255 * np.random.rand(32, 32, 32))
    with pytest.raises(Exception) as e_info:
        img = fs.FibsemImage(array1)
