import logging
from copy import deepcopy
from pathlib import Path

import napari
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import acquire, conversions, validation, patterning
from fibsem.detection import detection
from fibsem.detection.detection import DetectedFeatures, Feature
from fibsem.segmentation.model import load_model
from fibsem.structures import MicroscopeSettings, Point
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.FibsemDetectionUI import FibsemDetectionUI
from fibsem.ui.FibsemMovementUI import FibsemMovementUI
from fibsem.ui.user_window import GUIUserWindow
from fibsem.ui.FibsemMillingUI import FibsemMillingUI
from PyQt5.QtWidgets import QMessageBox


def ask_user_interaction(msg="Default Ask User Message", image: np.ndarray = None):
    """Create user interaction window and get return response"""
    ask_user_window = GUIUserWindow(msg=msg, image=image)
    ask_user_window.show()

    response = bool(ask_user_window.exec_())
    return response


def ask_user_movement(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    msg: str = None,
    pattern: patterning.MillingPattern = None,
    parent=None,
):

    viewer = napari.Viewer()

    movement_ui = FibsemMovementUI(
        microscope=microscope, 
        settings=settings, 
        msg=msg, 
        pattern=pattern,
        parent=parent, viewer=viewer
    )

    viewer.window.add_dock_widget(movement_ui, area="right", add_vertical_stretch=False)
    movement_ui.exec_()
    viewer.close()

    # napari.run()


def detect_features_v2(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    features: tuple[Feature],
    validate: bool = True,
    mask_radius: int = 256,
) -> DetectedFeatures:

    # take new image
    image = acquire.new_image(microscope, settings.image)

    # load model
    checkpoint = settings.protocol["ml"]["weights"]
    encoder = settings.protocol["ml"]["encoder"]
    num_classes = int(settings.protocol["ml"]["num_classes"])
    cuda = settings.protocol["ml"]["cuda"]
    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

    # detect features
    pixelsize = image.metadata.binary_result.pixel_size.x
    det = detection.locate_shift_between_features_v2(
        deepcopy(image.data), 
        model, 
        features=features, 
        pixelsize=pixelsize,
        mask_radius=mask_radius
    )

    # user validate features...
    if validate:
        # user validates detection result
        detection_ui = FibsemDetectionUI(
            microscope=microscope, settings=settings, detected_features=det,
        )
        detection_ui.show()
        detection_ui.exec_()

        det = detection_ui.detected_features

    # calculate features in microscope image coords
    det.features[0].feature_m = conversions.image_to_microscope_image_coordinates(
        det.features[0].feature_px, image.data, pixelsize
    )
    det.features[1].feature_m = conversions.image_to_microscope_image_coordinates(
        det.features[1].feature_px, image.data, pixelsize
    )

    return det


def run_validation_ui(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, log_path: Path
):
    """Run validation checks to confirm microscope state before run."""

    response = fibsem_ui.message_box_ui(
        title="Microscope State Validation",
        text="Do you want to validate the microscope state?",
    )

    if response is False:
        logging.info(f"PRE_RUN_VALIDATION cancelled by user.")
        return

    logging.info(f"INIT | PRE_RUN_VALIDATION | STARTED")

    # run validation
    validation.validate_initial_microscope_state(microscope, settings)

    # validate user configuration
    validation._validate_configuration_values(microscope, settings.protocol)

    # reminders
    reminder_str = """Please check that the following steps have been completed:
    \n - Sample is inserted
    \n - Confirm Operating Temperature
    \n - Needle Calibration
    \n - Ion Column Calibration
    \n - Crossover Calibration
    \n - Plasma Gas Valve Open
    \n - Initial Grid and Landing Positions
    """

    response = fibsem_ui.message_box_ui(
        title="Initialisation Checklist", text=reminder_str, buttons=QMessageBox.Ok,
    )

    # Loop backwards through the log, until we find the start of validation
    with open(log_path) as f:
        lines = f.read().splitlines()
        validation_warnings = []
        for line in lines[::-1]:
            if "PRE_RUN_VALIDATION" in line:
                break
            if "WARNING" in line:
                logging.info(line)
                validation_warnings.append(line)
        logging.info(
            f"{len(validation_warnings)} warnings were identified during intial setup."
        )

    if validation_warnings:
        warning_str = f"The following {len(validation_warnings)} warnings were identified during initialisation."

        for warning in validation_warnings[::-1]:
            warning_str += f"\n{warning.split('—')[-1]}"

        fibsem_ui.message_box_ui(
            title="Initialisation Warning", text=warning_str, buttons=QMessageBox.Ok,
        )

    logging.info(f"INIT | PRE_RUN_VALIDATION | FINISHED")


def milling_ui(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    milling_pattern: patterning.MillingPattern,
    point: Point = None,
    change_pattern: bool = False,
    auto_continue: bool = False,
):

    viewer = napari.Viewer()
    milling_ui = FibsemMillingUI(
        viewer=viewer,
        microscope=microscope,
        settings=settings,
        milling_pattern=milling_pattern,
        point=point,
        change_pattern=change_pattern,
        auto_continue=auto_continue,
    )

    viewer.window.add_dock_widget(milling_ui, area="right", add_vertical_stretch=False)

    if auto_continue:
        milling_ui.run_milling()
    else:
        milling_ui.exec_()

    # napari.run(max_loop_level=2)