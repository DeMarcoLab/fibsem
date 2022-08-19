# TODO

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import StagePosition, MoveSettings, AdornedImage, Rectangle
from fibsem.structures import ImageSettings, BeamType
from fibsem import calibration, acquire, movement


def correct_stage_eucentric_alignment(microscope: SdbMicroscopeClient, image_settings: ImageSettings, tilt_degrees: float = 25) -> None:

    # iteratively?

    # take images
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    
    # tilt stretch to match feature sizes 
    ib_image = calibration.cosine_stretch(ib_image, tilt_degrees)

    # cross correlate
    lp_px = int(max(ib_image.data.shape) / 12)
    hp_px = int(max(ib_image.data.shape) / 256)
    sigma = 6

    dx, dy, xcorr = calibration.shift_from_crosscorrelation(
        eb_image, ib_image, lowpass=lp_px, highpass=hp_px, sigma=sigma, 
        use_rect_mask=True, ref_mask=None
    )

    # TODO: error check?
    shift_within_tolerance = calibration.check_shift_within_tolerance(
        dx=dx, dy=dy, ref_image=eb_image, limit=0.5
    )

    # move vertically to correct eucentric position
    # TODO: check dy direction?
    movement.move_stage_eucentric_correction(microscope, dy)


def coarse_eucentric_alignment(microscope: SdbMicroscopeClient, hfw: float = 30e-6, eucentric_height: float = 3.91e-3) -> None:


    # focus and link stage
    calibration.auto_link_stage(microscope, hfw=hfw)

    # move to eucentric height
    stage = microscope.specimen.stage
    move_settings = MoveSettings(link_z_y=True)
    z_move = StagePosition(z=eucentric_height, coordinate_system="Specimen")
    stage.absolute_move(z_move, move_settings)



def beam_shift_alignment(
    microscope: SdbMicroscopeClient,
    image_settings: ImageSettings,
    ref_image: AdornedImage,
    reduced_area: Rectangle,
):
    """Align the images by adjusting the beam shift, instead of moving the stage
            (increased precision, lower range)

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client
        image_settings (acquire.ImageSettings): settings for taking image
        ref_image (AdornedImage): reference image to align to
        reduced_area (Rectangle): The reduced area to image with.
    """

    # # align using cross correlation
    new_image = acquire.new_image(
        microscope, settings=image_settings, reduced_area=reduced_area
    )
    dx, dy, _ = calibration.shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    # adjust beamshift
    microscope.beams.ion_beam.beam_shift.value += (-dx, dy)