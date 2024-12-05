import logging
from typing import List

from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemRectangleSettings,
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

def mill_stages(
    microscope: FibsemMicroscope,
    stages: List[FibsemMillingStage],
    asynch: bool = False,
    parent_ui=None,
):
    """Run a list of milling stages, with a progress bar and notifications."""

    if isinstance(stages, FibsemMillingStage):
        stages = [stages]

    try:
        for idx, stage in enumerate(stages):
            if parent_ui:
                parent_ui.milling_notification.emit(f"Preparing: {stage.name}")

            if parent_ui:
                if parent_ui._STOP_MILLING:
                    raise Exception("Milling stopped by user.")

            try:
                stage.strategy.run(
                    microscope=microscope,
                    stage=stage,
                    asynch=False,
                    parent_ui=parent_ui,
                    current_stage_index=idx,
                    total_stages=len(stages),
                )
                # TODO: drift correction

                if parent_ui:
                    parent_ui._progress_bar_quit.emit()
                    parent_ui.milling_notification.emit(f"Milling stage complete: {stage.name}")
            except Exception as e:
                logging.error(f"Error running milling stage: {stage.name}")

        if parent_ui:
            parent_ui.milling_notification.emit(
                f"Milling complete. {len(stages)} stages completed."
            )
    
    except Exception as e:
        if parent_ui:
            import napari.utils.notifications
            napari.utils.notifications.show_error(f"Error while milling {e}")
        logging.error(e)
    finally:
        finish_milling(
            microscope=microscope,
            imaging_current=microscope.system.ion.beam.beam_current,
            imaging_voltage=microscope.system.ion.beam.voltage,
        )



############################# UTILS #############################