import pytest


from fibsem import microscope, utils, acquire
import fibsem.alignment as fa
from fibsem.structures import BeamType, ImageSettings, GammaSettings, FibsemRectangle, FibsemImage, check_data_format
import matplotlib.pyplot as plt
from fibsem import calibration
import os
import logging
from fibsem.utils import current_timestamp, configure_logging
from pathlib import Path
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    GrabFrameSettings,
    Rectangle,
    RunAutoCbSettings,
)

@pytest.fixture
def settings():

    settings = utils.load_settings_from_config()

    return settings

@pytest.fixture
def general_setup(settings):

    session_path: Path = None
    config_path: Path = None
    protocol_path: Path = None
    setup_logging: bool = True

    session = f'{settings.protocol["name"]}_{current_timestamp()}'
    if protocol_path is None:
        protocol_path = os.getcwd()

    assert protocol_path == os.getcwd()

    # configure paths
    if session_path is None:
        session_path = os.path.join(os.path.dirname(protocol_path), session)

    os.makedirs(session_path, exist_ok=True)

    assert os.path.exists(session_path)

    # configure logging
    if setup_logging:
        configure_logging(session_path)

    # image_settings
    settings.image.save_path = session_path
    
    return settings


@pytest.fixture
def microscope_thermo():
    microscope = microscope.ThermoMicroscope()
    return microscope

@pytest.fixture
def connection(microscope=microscope_thermo,settings=general_setup):

    try:
        microscope.connect_to_microscope(settings.system.ip_address)
    except:
        logging.info('Could not connect to microscope')
    return microscope

@pytest.fixture
def gamma_settings():

    gamma_settings = GammaSettings(
    enabled=True,
    min_gamma=0.5,
    max_gamma=1.8,
    scale_factor=0.01,
    threshold=46,
    )

    return gamma_settings

@pytest.fixture
def image_settings():

    image_settings =  ImageSettings(
    resolution=(1536,1024),
    dwell_time=1.0e-6,
    hfw=150.0e-6,
    autocontrast=True,
    beam_type=BeamType.ELECTRON,
    gamma=gamma_settings,
    save=True,
    save_path="fibsem\\test_images",
    label=utils.current_timestamp(),
    reduced_area=None,
    )
    
    return image_settings

@pytest.fixture
def frame_settings(image_settings):

    frame_settings = GrabFrameSettings(
    resolution=image_settings.resolution,
    dwell_time=image_settings.dwell_time,
    reduced_area=image_settings.reduced_area,
    )

    return frame_settings

@pytest.fixture
def label(image_settings):

    if image_settings.beam_type is BeamType.ELECTRON:
        label = f"{image_settings.label}_eb"

    elif image_settings.beam_type is BeamType.ION:
        label = f"{image_settings.label}_ib"

    return label


def test_acquire_image_electron():

    microscope, settings = utils.setup_session()
    image_settings = settings.image
    frame_settings = GrabFrameSettings(
        resolution=image_settings.resolution,
        dwell_time=image_settings.dwell_time,
        reduced_area=None,
    )
    
    new_image = microscope.acquire_image(frame_settings, image_settings)
    
    assert new_image is not None
    assert isinstance(new_image,FibsemImage)
    assert check_data_format(new_image.data)
    assert new_image.metadata.version == 'v1'
    

    if image_settings.gamma.enabled:
        image = acquire.auto_gamma(new_image, image_settings.gamma)


    # save image
    if image_settings.save:
        filename = os.path.join(image_settings.save_path, label)
        image.save(save_path=filename)


        print(filename)

        assert os.path.exists(f"{filename}.tif")
        print("Image saved")


def test_acquire_image_ion(image_settings,frame_settings, microscope=connection):

    image_settings.beam_type = BeamType.ION

    try:
        new_image = microscope.acquire_image(frame_settings, image_settings)
    except:
        logging.info("Image could not be taken")

    assert new_image is not None
    assert isinstance(new_image,FibsemImage)
    assert check_data_format(new_image.data)
    assert new_image.metadata.version == 'v1'
    

    if image_settings.gamma.enabled:
        image = acquire.auto_gamma(new_image, image_settings.gamma)


    # save image
    if image_settings.save:
        filename = os.path.join(image_settings.save_path, label)
        image.save(save_path=filename)


        print(filename)

        assert os.path.exists(f"{filename}.tif")
        print("Image saved")

@pytest.fixture
def last_label(image_settings):

    label = f"{image_settings.label}_last"

    return label

def test_last_image(image_settings,last_label, microscope=connection):

    last_image = microscope.last_image(image_settings.beam_type)

    if image_settings.save:
        filename = os.path.join(image_settings.save_path, last_label)
        last_image.save(save_path=filename)
        assert os.path.exists(f"{filename}.tif")

    assert isinstance(last_image,FibsemImage)
    assert check_data_format(last_image.data)
    assert last_image.metadata.version == 'v1'



def test_beam_shift(microscope=connection):

    microscope.reset_beam_shifts()

    assert microscope.connection.beams.electron_beam.beam_shift.value.x == 0
    assert microscope.connection.beams.electron_beam.beam_shift.value.y ==  0

    assert microscope.connection.beams.ion_beam.beam_shift.value.x == 0
    assert microscope.connection.beams.ion_beam.beam_shift.value.y ==  0



# def test_auto_contrast(microscope):

#     microscope.connection.detector.contrast = 1

#     microscope.auto_contrast(beam_type=image_settings.beam_type)

#     assert microscope.connection.detector.contrast != 1

    







