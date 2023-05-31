
import logging
from copy import deepcopy
from pathlib import Path

import napari
import numpy as np
from fibsem import acquire, conversions, validation
from fibsem.detection import detection
from fibsem.detection.detection import (DetectedFeatures,Feature)
from fibsem.segmentation.model import load_model
from fibsem.structures import MicroscopeSettings, Point
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.FibsemDetectionUI import FibsemDetectionUI
from PyQt5.QtWidgets import QMessageBox
from fibsem.microscope import FibsemMicroscope


def detect_features_v2(microscope: FibsemMicroscope, settings: MicroscopeSettings, features: tuple[Feature], validate: bool = True) -> DetectedFeatures:

    # take new image
    image = acquire.new_image(microscope, settings.image)

    # load model
    ml_protocol = settings.protocol.get("ml", {})
    checkpoint = ml_protocol.get("weights", None)    
    encoder = ml_protocol.get("encoder", "ResNet18")
    num_classes = int(ml_protocol.get("num_classes", 3))
    cuda = ml_protocol.get("cuda", False)
    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

    # detect features
    pixelsize = image.metadata.pixel_size.x
    det = detection.detect_features(deepcopy(image.data), model, features=features, pixelsize=pixelsize)

    # user validate features...
    if validate:
        # # user validates detection result
        # detection_ui = FibsemDetectionUI(
        #     microscope=microscope,
        #     settings=settings,
        #     detected_features=det,
        # )
        # detection_ui.show()
        # detection_ui.exec_()

        # det = detection_ui.detected_features
        input("Ensure features are correct, then press enter to continue...")

    # calculate features in microscope image coords
    det.features[0].feature_m = conversions.image_to_microscope_image_coordinates(det.features[0].px, image.data, pixelsize)
    det.features[1].feature_m = conversions.image_to_microscope_image_coordinates(det.features[1].px, image.data, pixelsize)

    return det


def run_validation_ui(
    microscope: FibsemMicroscope, settings: MicroscopeSettings, log_path: Path
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
    \n - Experiment is inserted
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
