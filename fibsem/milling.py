import logging
from fibsem.structures import (
    BeamType,
    FibsemPatternType,
    FibsemImage,
    FibsemRectangle,
    FibsemPattern,
    FibsemMillingSettings,
    Point,
    ImageSettings,
    MicroscopeSettings
)
from typing import Union
from fibsem.microscope import FibsemMicroscope


########################### SETUP


def setup_milling(
    microscope: FibsemMicroscope,
    mill_settings: FibsemMillingSettings = None,
):
    """Setup Microscope for Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        patterning_mode (str, optional): Ion beam milling patterning mode. Defaults to "Serial".
        hfw (float, optional): horizontal field width for milling. Defaults to 100e-6.
    """
    microscope.setup_milling(mill_settings = mill_settings)

def run_milling_drift_corrected(
    microscope: FibsemMicroscope, 
    milling_current: float,  
    image_settings: ImageSettings, 
    ref_image: FibsemImage, 
    reduced_area: FibsemRectangle = None,
) -> None:
    """Run Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_current (float, optional): ion beam milling current. Defaults to None.
        asynch (bool, optional): flag to run milling asynchronously. Defaults to False.
    """
    microscope.run_milling_drift_corrected(milling_current, image_settings, ref_image, reduced_area)

def run_milling(
    microscope: FibsemMicroscope,
    milling_current: float,
    milling_voltage: float,
    asynch: bool = False,
) -> None:
    """Run Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_current (float, optional): ion beam milling current. Defaults to None.
        asynch (bool, optional): flag to run milling asynchronously. Defaults to False.
    """
    microscope.run_milling(milling_current, milling_voltage, asynch)


def estimate_milling_time(microscope: FibsemMicroscope, microscope_patterns) -> float:
    """Get the milling status.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance

    """

    total_time = microscope.estimate_milling_time(microscope_patterns)
        
    return total_time

def finish_milling(
    microscope: FibsemMicroscope, imaging_current: float = 20e-12, imaging_voltage: float = 30e3
) -> None:
    """Clear milling patterns, and restore to the imaging current.

    Args:
        microscope (FIbsemMicroscope): Fibsem microscope instance
        imaging_current (float, optional): Imaging Current. Defaults to 20e-12.

    """
    # restore imaging current
    logging.info(f"Changing to Imaging Current: {imaging_current:.2e}")
    microscope.finish_milling(imaging_current=imaging_current, imaging_voltage=imaging_voltage)
    logging.info("Finished Ion Beam Milling.")

def draw_patterns(microscope: FibsemMicroscope, patterns: list[FibsemPattern]) -> None:
    """Draw a milling pattern from settings
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
    """
    microscope_patterns = []
    for pattern in patterns:
        microscope_patterns.append(draw_pattern(microscope, pattern))
    return microscope_patterns

        
def draw_pattern(microscope: FibsemMicroscope, pattern: FibsemPattern):
    """Draw a milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPattern): pattern settings
        mill_settings (FibsemMillingSettings): milling settings
    """
    if pattern.pattern is FibsemPatternType.Rectangle:
        microscope_pattern = microscope.draw_rectangle(pattern)

    elif pattern.pattern is FibsemPatternType.Line:
        microscope_pattern = microscope.draw_line(pattern)

    elif pattern.pattern is FibsemPatternType.Circle:
        microscope_pattern = microscope.draw_circle(pattern)

    elif pattern.pattern is FibsemPatternType.Bitmap:
        microscope_pattern = microscope.draw_bitmap_pattern(pattern, pattern.path)

    elif pattern.pattern is FibsemPatternType.Annulus:
        microscope_pattern = microscope.draw_annulus(pattern)
        
    return microscope_pattern


def draw_rectangle(
    microscope: FibsemMicroscope, pattern_settings: FibsemPattern
):
    """Draw a rectangular milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPattern): pattern settings
        mill_settings (FibsemMillingSettings): milling settings
    """
    pattern = microscope.draw_rectangle(pattern_settings)
    return pattern


def draw_line(microscope: FibsemMicroscope, pattern_settings: FibsemPattern):
    """Draw a line milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        mill_settings (MillingSettings): milling pattern settings
    """
    pattern = microscope.draw_line(pattern_settings)
    return pattern

def draw_circle(microscope: FibsemMicroscope, pattern_settings: FibsemPattern):
    """Draw a circular milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        mill_settings (MillingSettings): milling pattern settings
    """
    pattern = microscope.draw_circle(pattern_settings)
    return pattern

def convert_to_bitmap_format(path):
    from PIL import Image
    import os 
    img=Image.open(path)
    a=img.convert("RGB", palette=Image.ADAPTIVE, colors=8)
    new_path = os.path.join(os.path.dirname(path), "24bit_img.tif")
    a.save(new_path)
    return new_path

def draw_bitmap(microscope: FibsemMicroscope, pattern_settings: FibsemPattern, path: str):
    """Draw a butmap milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem
        mill_settings (MillingSettings): milling pattern settings
    """
    path = convert_to_bitmap_format(path)
    pattern = microscope.draw_bitmap_pattern(pattern_settings, path)
    return pattern

def milling_protocol(
    microscope: FibsemMicroscope,
    mill_settings: FibsemMillingSettings,
    patterns: list = [FibsemPattern],
    drift_correction: bool = False,
    image_settings: ImageSettings = None,
    ref_image: FibsemImage = None,
    reduced_area: FibsemRectangle = None,
    asynch: bool = False,
):
    # setup milling
    setup_milling(microscope, mill_settings)

    # draw patterns
    for pattern in patterns:
        draw_pattern(microscope, pattern)

    # run milling
    if drift_correction:
        run_milling_drift_corrected(microscope, mill_settings.milling_current, image_settings, ref_image, reduced_area)
    else:
        run_milling(microscope, mill_settings.milling_current, mill_settings.milling_voltage, asynch)

    # finish milling
    finish_milling(microscope)

from fibsem.patterning import FibsemMillingStage

def mill_stages(microscope: FibsemMicroscope, settings: MicroscopeSettings, stages: list[FibsemMillingStage], asynch: bool=False):
    for stage in stages:
        mill_stage(microscope=microscope, settings=settings, stage=stage, asynch=asynch)

def mill_stage(microscope: FibsemMicroscope, settings: MicroscopeSettings, stage: FibsemMillingStage, asynch: bool=False):

    # set up milling
    setup_milling(microscope, stage.milling)

    # draw patterns
    for pattern in stage.pattern.patterns:
        draw_pattern(microscope, pattern)

    run_milling(microscope, stage.milling.milling_current, asynch)

    # finish milling
    finish_milling(microscope)


############################# UTILS #############################
