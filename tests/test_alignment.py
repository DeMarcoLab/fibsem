import pytest
from fibsem import utils, acquire, alignment
import random
import numpy as np

def test_align_from_crosscorrelation():
    
    microscope, settings = utils.setup_session(debug=False)

    # create random images
    ref_image = acquire.acquire_image(microscope, settings.image)
    new_image = acquire.acquire_image(microscope, settings.image)

    ref_image.data[:] = 0
    new_image.data[:] = 0

    # crop a random square out
    w = h = 150
    x = random.randint(0, ref_image.data.shape[1] - 2*w)
    y = random.randint(0, ref_image.data.shape[0] - 2*h)
    ref_image.data[y:y+h, x:x+w]  = 255

    # new image should be offset by 250 pixels in x and y
    offset = random.randint(-50, 50)
    new_image.data[y+offset:y+h+offset, x+offset:x+w+offset] = 255

    dx, dy, xcorr = alignment.shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )    

    # write a test case with pytest for this case
    # test that the shift is within 1 pixel of the offset

    pixel_size = ref_image.metadata.pixel_size.x
    assert np.isclose(dx, offset*pixel_size, atol=pixel_size), f"dx: {dx}, offset: {offset*pixel_size}"
    assert np.isclose(dy, offset*pixel_size, atol=pixel_size), f"dy: {dy}, offset: {offset*pixel_size}"