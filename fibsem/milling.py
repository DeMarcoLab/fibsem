import logging
from typing import List

from fibsem.microscope import FibsemMicroscope
from fibsem.patterning import FibsemMillingStage
from fibsem.structures import (
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    ImageSettings,
)

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

def draw_patterns(microscope: FibsemMicroscope, patterns: List[FibsemPatternSettings]) -> None:
    """Draw a milling pattern from settings
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
    """
    microscope_patterns = []
    for pattern in patterns:
        microscope_patterns.append(draw_pattern(microscope, pattern))
    return microscope_patterns

        
def draw_pattern(microscope: FibsemMicroscope, pattern: FibsemPatternSettings):
    """Draw a milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPatternSettings): pattern settings
        mill_settings (FibsemMillingSettings): milling settings
    """
    if isinstance(pattern, FibsemRectangleSettings):
        microscope_pattern = microscope.draw_rectangle(pattern)

    elif isinstance(pattern, FibsemLineSettings):
        microscope_pattern = microscope.draw_line(pattern)

    elif isinstance(pattern, FibsemCircleSettings):
        if pattern.thickness != 0:
            microscope_pattern = microscope.draw_annulus(pattern)
        else:
            microscope_pattern = microscope.draw_circle(pattern)

    elif isinstance(pattern, FibsemBitmapSettings):
        microscope_pattern = microscope.draw_bitmap_pattern(pattern, pattern.path)
        
    return microscope_pattern


def convert_to_bitmap_format(path):
    import os

    from PIL import Image
    img=Image.open(path)
    a=img.convert("RGB", palette=Image.ADAPTIVE, colors=8)
    new_path = os.path.join(os.path.dirname(path), "24bit_img.tif")
    a.save(new_path)
    return new_path


def mill_stages(microscope: FibsemMicroscope, stages: List[FibsemMillingStage], asynch: bool=False):
    for stage in stages:
        mill_stage(microscope=microscope, stage=stage, asynch=asynch)

    # finish milling (restore imaging conditions)
    finish_milling(microscope,
                   imaging_current=microscope.system.ion.beam.beam_current,
                   imaging_voltage=microscope.system.ion.beam.voltage)

def mill_stage(microscope: FibsemMicroscope, stage: FibsemMillingStage, asynch: bool=False):

    # set up milling
    setup_milling(microscope, stage.milling)

    # draw patterns
    for pattern in stage.pattern.patterns:
        draw_pattern(microscope, pattern)

    run_milling(microscope=microscope, 
        milling_current=stage.milling.milling_current, 
        milling_voltage=stage.milling.milling_voltage, 
        asynch=asynch)



def mill_stages_ui(microscope: FibsemMicroscope, stages: List[FibsemMillingStage], asynch: bool=False, parent_ui = None):
    """Run a list of milling stages, with a progress bar and notifications."""
    import napari.utils.notifications

    if isinstance(stages, FibsemMillingStage):
        stages = [stages]

  
    for idx, stage in enumerate(stages):
    
        if parent_ui:
            parent_ui.milling_notification.emit(f"Preparing: {stage.name}")
    
        try:
            mill_stage_ui(microscope, stage, parent_ui, idx, len(stages))
            # TODO implement a special case for overtilt milling

            # drift correction

        except Exception as e:
            napari.utils.notifications.show_error(f"Error running milling stage: {stage.name}")
            logging.error(e)
        finally:
            finish_milling(microscope, 
                                        imaging_current=microscope.system.ion.beam.beam_current, 
                                        imaging_voltage=microscope.system.ion.beam.voltage)
        if parent_ui:
            parent_ui._progress_bar_quit.emit()
            parent_ui.milling_notification.emit(f"Milling stage complete: {stage.name}")
    
    if parent_ui:
        parent_ui.milling_notification.emit(f"Milling complete. {len(stages)} stages completed.")


def mill_stage_ui(microscope: FibsemMicroscope, stage: FibsemMillingStage, parent_ui = None, idx: int = 0, n_stages: int = 1):

    setup_milling(microscope, mill_settings=stage.milling)

    patterns = draw_patterns(microscope, stage.pattern.patterns)
    estimated_time = estimate_milling_time(microscope, patterns)
    logging.info(f"Estimated time for {stage.name}: {estimated_time:.2f} seconds")

    if parent_ui:
        progress_bar_dict = {"estimated_time": estimated_time, "idx": idx, "total": n_stages}
        parent_ui._progress_bar_start.emit(progress_bar_dict)
        parent_ui.milling_notification.emit(f"Running {stage.name}...")

    run_milling(microscope, stage.milling.milling_current, stage.milling.milling_voltage)

    return 



def mill_stage_with_overtilt(microscope: FibsemMicroscope, stage: FibsemMillingStage, ref_image: FibsemImage, asynch=False ):
    """Mill a trench pattern with overtilt, 
    based on https://www.sciencedirect.com/science/article/abs/pii/S1047847716301514 and autolamella v1"""
    # set up milling
    setup_milling(microscope, stage.milling)

    import numpy as np

    from fibsem import alignment, patterning
    from fibsem.structures import FibsemStagePosition

    overtilt_in_degrees = stage.pattern.protocol.get("overtilt", 1)

    # assert pattern is TrenchPattern
    if not isinstance(stage.pattern, patterning.TrenchPattern):
        raise ValueError("Pattern must be TrenchPattern for overtilt milling")

    for i, pattern in enumerate(stage.pattern.patterns):
        
        # save initial position
        initial_position = microscope.get_stage_position()
        
        # overtilt
        if i == 0:
            t = -np.deg2rad(overtilt_in_degrees)
        else:
            t = +np.deg2rad(overtilt_in_degrees)
        microscope.move_stage_relative(FibsemStagePosition(t=t))

        # beam alignment
        image_settings = ImageSettings.fromFibsemImage(ref_image)
        image_settings.filename = f"alignment_target_{stage.name}"
        
        alignment.multi_step_alignment_v2(microscope=microscope, 
                                        ref_image=ref_image, 
                                        beam_type=BeamType.ION, 
                                        alignment_current=None,
                                        steps=3)

        # setup again to ensure we are milling at the correct current, cleared patterns
        setup_milling(microscope, stage.milling)

        # draw pattern
        draw_pattern(microscope, pattern)

        # run milling
        run_milling(microscope, stage.milling.milling_current, asynch)

        # return to initial position
        microscope.move_stage_absolute(initial_position)

    # finish milling
    finish_milling(microscope)



############################# UTILS #############################
