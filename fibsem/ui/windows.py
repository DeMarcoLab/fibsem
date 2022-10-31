
import logging
from copy import deepcopy
from pathlib import Path

import napari
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import acquire, conversions, validation
from fibsem.detection import detection
from fibsem.detection.detection import (DetectedFeatures,Feature)
from fibsem.segmentation.model import load_model
from fibsem.structures import MicroscopeSettings, Point
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.detection_window import GUIDetectionWindow
from fibsem.ui.FibsemMovementUI import FibsemMovementUI
from fibsem.ui.user_window import GUIUserWindow
from PyQt5.QtWidgets import QMessageBox


def ask_user_interaction(
    msg="Default Ask User Message", image: np.ndarray = None
):
    """Create user interaction window and get return response"""
    ask_user_window = GUIUserWindow(msg=msg, image=image)
    ask_user_window.show()

    response = bool(ask_user_window.exec_())
    return response


def ask_user_movement(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    msg_type="eucentric",
    msg: str = None,
    parent=None,
):

    logging.info(f"Asking user for confirmation for {msg_type} movement")

    viewer = napari.Viewer()

    movement_ui = FibsemMovementUI(
        microscope=microscope,
        settings=settings,
        msg_type=msg_type,
        msg=msg,
        parent=parent,
        viewer=viewer
    )
    
    viewer.window.add_dock_widget(movement_ui, area="right", add_vertical_stretch=False)
    movement_ui.exec_()

    # napari.run()


# def detect_features(
#     microscope: SdbMicroscopeClient,
#     settings: MicroscopeSettings,
#     ref_image: AdornedImage,
#     features: tuple[Feature],
#     validate: bool = True,
# ) -> DetectionResult:
#     """_summary_

#     Args:
#         microscope (SdbMicroscopeClient): _description_
#         settings (dict): _description_
#         image_settings (ImageSettings): _description_
#         ref_image (AdornedImage): _description_
#         features (tuple[Feature]): _description_
#         validate (bool, optional): _description_. Defaults to True.

#     Returns:
#         DetectionResult: _description_
#     """
#     # detect features
#     image = acquire.new_image(microscope, settings.image)

#     # detect features
#     det = detection.locate_shift_between_features(image, features=features)

#     # user validate features...
#     if validate:
#         det = validate_detection(
#             microscope,
#             settings,
#             det,
#         )

#     return det

def detect_features_v2(microscope: SdbMicroscopeClient, settings: MicroscopeSettings, features: tuple[Feature], validate: bool = True) -> DetectedFeatures:

    # take new image
    image = acquire.new_image(microscope, settings.image)

    # load model
    checkpoint = settings.protocol["ml"]["weights"]
    model = load_model(checkpoint)

    # detect features
    pixelsize = image.metadata.binary_result.pixel_size.x
    det = detection.locate_shift_between_features_v2(deepcopy(image.data), model, features=features, pixelsize=pixelsize)

    # user validate features...
    if validate:
        # user validates detection result
        detection_window = GUIDetectionWindow(
            microscope=microscope,
            settings=settings,
            detected_features=det,
        )
        detection_window.show()
        detection_window.exec_()

        det = detection_window.detected_features

    # calculate features in microscope image coords
    det.features[0].feature_m = conversions.image_to_microscope_image_coordinates(det.features[0].feature_px, image.data, pixelsize)
    det.features[1].feature_m = conversions.image_to_microscope_image_coordinates(det.features[1].feature_px, image.data, pixelsize)

    return det



# def validate_detection(
#     microscope: SdbMicroscopeClient,
#     settings: MicroscopeSettings,
#     detected_features: DetectedFeatures,
# ):
#     # user validates detection result
#     detection_window = GUIDetectionWindow(
#         microscope=microscope,
#         settings=settings,
#         detected_features=detected_features,
#     )
#     detection_window.show()
#     detection_window.exec_()

#     return detection_window.detected_features

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
        title="Initialisation Checklist",
        text=reminder_str,
        buttons=QMessageBox.Ok,
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
            title="Initialisation Warning",
            text=warning_str,
            buttons=QMessageBox.Ok,
        )

    logging.info(f"INIT | PRE_RUN_VALIDATION | FINISHED")
