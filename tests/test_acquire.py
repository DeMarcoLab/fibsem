import numpy as np
import pytest

from fibsem import acquire, utils
from fibsem.structures import (
    FibsemImage,
    FibsemRectangle,
)

def test_reduced_area_acquisition():
    """Test the reduced area acquisition functionality of the acquire module."""
    # setup a demo microscope session
    microscope, settings = utils.setup_session(manufacturer="Demo")

    resolution = settings.image.resolution

    # acquire a full frame image
    image = acquire.acquire_image(microscope, settings.image)
    assert isinstance(image, FibsemImage)
    assert image.data.shape == (resolution[1], resolution[0])

    # acquire a reduced area image
    settings.image.reduced_area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    image = acquire.acquire_image(microscope, settings.image)
    assert isinstance(image, FibsemImage)
    assert image.data.shape == (resolution[1] // 2, resolution[0] // 2)