import logging
import time
from pathlib import Path
from typing import List, Tuple

from fibsem import acquire, config as fcfg
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPatternSettings,
    FibsemRectangleSettings,
    ImageSettings,
    BeamType,
)
from fibsem.utils import current_timestamp_v2

########################### SETUP


def setup_milling(
    microscope: FibsemMicroscope,
    milling_stage: FibsemMillingStage,
):
    """Setup Microscope for FIB Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_stage (FibsemMillingStage): Milling Stage
    """

    # acquire reference image for drift correction
    if milling_stage.alignment.enabled:
        reference_image = get_stage_reference_image(
            microscope=microscope, milling_stage=milling_stage
        )

    # set up milling settings
    microscope.setup_milling(mill_settings=milling_stage.milling)

    # align at the milling current to correct for shift
    if milling_stage.alignment.enabled:
        from fibsem import alignment
        logging.info(f"FIB Aligning at Milling Current: {milling_stage.milling.milling_current:.2e}")
        alignment.multi_step_alignment_v2(
            microscope=microscope,
            ref_image=reference_image,
            beam_type=milling_stage.milling.milling_channel,
            steps=3,
            use_autocontrast=True,
        )  # high current -> damaging


# TODO: migrate run milling to take milling_stage argument, rather than current, voltage
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

def finish_milling(
    microscope: FibsemMicroscope, imaging_current: float = 20e-12, imaging_voltage: float = 30e3
) -> None:
    """Clear milling patterns, and restore to the imaging current.

    Args:
        microscope (FIbsemMicroscope): Fibsem microscope instance
        imaging_current (float, optional): Imaging Current. Defaults to 20e-12.
        imaging_voltage: Imaging Voltage. Defaults to 30e3.
    """
    # restore imaging current
    logging.info(f"Changing to Imaging Current: {imaging_current:.2e}")
    microscope.finish_milling(imaging_current=imaging_current, imaging_voltage=imaging_voltage)
    logging.info("Finished Ion Beam Milling.")

def draw_patterns(microscope: FibsemMicroscope, patterns: List[FibsemPatternSettings]) -> None:
    """Draw milling patterns on the microscope from the list of settings
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        patterns (List[FibsemPatternSettings]): List of milling patterns
    """
    for pattern in patterns:
        draw_pattern(microscope, pattern)

        
def draw_pattern(microscope: FibsemMicroscope, pattern: FibsemPatternSettings):
    """Draw a milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPatternSettings): pattern settings
    """
    if isinstance(pattern, FibsemRectangleSettings):
        microscope.draw_rectangle(pattern)

    elif isinstance(pattern, FibsemLineSettings):
        microscope.draw_line(pattern)

    elif isinstance(pattern, FibsemCircleSettings):
        microscope.draw_circle(pattern)

    elif isinstance(pattern, FibsemBitmapSettings):
        microscope.draw_bitmap_pattern(pattern, pattern.path)

def convert_to_bitmap_format(path):
    import os

    from PIL import Image
    img=Image.open(path)
    a=img.convert("RGB", palette=Image.ADAPTIVE, colors=8)
    new_path = os.path.join(os.path.dirname(path), "24bit_img.tif")
    a.save(new_path)
    return new_path


def mill_stage(microscope: FibsemMicroscope, stage: FibsemMillingStage, asynch: bool=False):
    logging.warning("mill_stage will be deprecated in the next version. Use mill_stages instead.")
    # set up milling
    setup_milling(microscope, milling_stage=stage)

    # draw patterns
    for pattern in stage.pattern.define():
        draw_pattern(microscope, pattern)

    run_milling(microscope=microscope, 
        milling_current=stage.milling.milling_current, 
        milling_voltage=stage.milling.milling_voltage, 
        asynch=asynch)

def mill_stages(
    microscope: FibsemMicroscope,
    stages: List[FibsemMillingStage],
    parent_ui=None,
):
    """Run a list of milling stages, with a progress bar and notifications."""

    if isinstance(stages, FibsemMillingStage):
        stages = [stages]

    # TMP: store initial imaging path
    imaging_path = microscope.get_imaging_settings(beam_type=BeamType.ION).path

    try:
        if hasattr(microscope, "milling_progress_signal"):
            if parent_ui: # TODO: tmp ladder to handle progress indirectly
                def _handle_progress(ddict: dict) -> None:
                    parent_ui.milling_progress_signal.emit(ddict)
            else:
                def _handle_progress(ddict: dict) -> None:
                    logging.info(ddict)
            microscope.milling_progress_signal.connect(_handle_progress)

        reference_image = get_stage_reference_image(
            microscope=microscope, milling_stage=stages[0]
        )

        initial_beam_shift = microscope.get("shift", beam_type=stages[0].milling.milling_channel)

        # TODO: reset beam shift after aligning at milling current

        for idx, stage in enumerate(stages):
            start_time = time.time()
            if parent_ui:
                if parent_ui.STOP_MILLING:
                    raise Exception("Milling stopped by user.")

                msgd =  {"msg": f"Preparing: {stage.name}",
                        "progress": {"state": "start", 
                                    "start_time": start_time,
                                    "current_stage": idx, 
                                    "total_stages": len(stages),
                                    }}
                parent_ui.milling_progress_signal.emit(msgd)

            try:
                stage.reference_image = reference_image
                stage.strategy.run(
                    microscope=microscope,
                    stage=stage,
                    asynch=False,
                    parent_ui=parent_ui,
                )

                # performance logging
                msgd = {"msg": "mill_stages", "idx": idx, "stage": stage.to_dict(), "start_time": start_time, "end_time": time.time()}
                logging.debug(f"{msgd}")

                # optionally acquire images after milling
                if stage.milling.acquire_images:
                    acquire_images_after_milling(microscope=microscope, milling_stage=stage, 
                                                 start_time=start_time, 
                                                 path=imaging_path)

                if parent_ui:
                    parent_ui.milling_progress_signal.emit({"msg": f"Finished: {stage.name}"})
            except Exception as e:
                logging.error(f"Error running milling stage: {stage.name}, {e}")

        if parent_ui:
            parent_ui.milling_progress_signal.emit({"msg": f"Finished {len(stages)} Milling Stages. Restoring Imaging Conditions..."})

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
        # restore initial beam shift
        if initial_beam_shift:
            microscope.set(key="shift", value=initial_beam_shift, beam_type=BeamType.ION)
        if hasattr(microscope, "milling_progress_signal"):
            microscope.milling_progress_signal.disconnect(_handle_progress)

############################# UTILS #############################

def acquire_images_after_milling(
    microscope: FibsemMicroscope,
    milling_stage: FibsemMillingStage,
    start_time: float,
    path: str,
) -> Tuple[FibsemImage, FibsemImage]:
    """Acquire images after milling for reference.
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_stage (FibsemMillingStage): Milling Stage
        start_time (float): Start time of milling (used for filename / tracking)
        path (str): Path to save images
    """

    # restore imaging conditions
    finish_milling(
        microscope=microscope,
        imaging_current=microscope.system.ion.beam.beam_current,
        imaging_voltage=microscope.system.ion.beam.voltage,
    )

    # set imaging parameters (filename, path, etc.)
    if milling_stage.imaging.path is None:
        milling_stage.imaging.path = path
    milling_stage.imaging.filename = f"ref_milling_{milling_stage.name.replace(' ', '-')}_finished_{str(start_time).replace('.', '_')}"
    
    # from pprint import pprint
    # pprint(milling_stage.imaging.to_dict())

    # acquire images
    from fibsem import acquire
    images = acquire.take_reference_images(microscope, milling_stage.imaging)

    # query: set the images to the UI?
    # query: add an id to the milling stage to track the images?
    # QUERY: what is a better way to set the path?

    return images


def get_stage_reference_image(
    microscope: FibsemMicroscope, milling_stage: FibsemMillingStage
) -> FibsemImage:
    ref_image = milling_stage.reference_image
    if isinstance(ref_image, FibsemImage):
        return ref_image
    elif ref_image is None:
        path = milling_stage.imaging.path
        if path is None:
            path = Path(fcfg.DATA_CC_PATH)
        image_settings = ImageSettings(
            hfw=milling_stage.milling.hfw,
            dwell_time=1e-6,
            resolution=[1536, 1024],
            beam_type=milling_stage.milling.milling_channel,
            reduced_area=milling_stage.alignment.rect,
            save=True,
            path=path,
            filename=f"ref_{milling_stage.name}_initial_alignment_{current_timestamp_v2()}",
        )
        return acquire.acquire_image(microscope, image_settings)
    raise TypeError(f"Invalid ref_image type '{type(ref_image)}'")


# QUERY: should List[FibsemMillingStage] be a class? that has it's own settings?
# E.G. should acquire images be set at that level, rather than at the stage level?
