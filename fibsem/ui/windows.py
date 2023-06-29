
import logging
from copy import deepcopy
from pathlib import Path

import napari
import numpy as np
from PyQt5.QtWidgets import QMessageBox

from fibsem import acquire, conversions, validation
from fibsem.acquire import take_reference_images
from fibsem.detection import detection
from fibsem.detection.detection import (DetectedFeatures, Feature, ImageCentre,
                                        LamellaCentre, LamellaLeftEdge,
                                        LamellaRightEdge, LandingPost,
                                        NeedleTip, detect_features)
from fibsem.microscope import FibsemMicroscope
from fibsem.segmentation.model import load_model
from fibsem.structures import (BeamType, FibsemStagePosition,
                               MicroscopeSettings, Point)
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.FibsemDetectionUI import FibsemDetectionUI
from fibsem.ui.FibsemDetectionWidget import detection_ui
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.utils import message_box_ui


def detect_features_v2(microscope: FibsemMicroscope, settings: MicroscopeSettings, features: tuple[Feature], validate: bool = True,end_response:str = None) -> DetectedFeatures:

    if settings.image.reduced_area is not None:
        logging.info(f"Reduced area is not compatible with model detection, disabling...")
        settings.image.reduced_area = None

    # take new image
    image = acquire.new_image(microscope, settings.image)

    # load model
    ml_protocol = settings.protocol.get("ml", {})
    checkpoint = ml_protocol.get("weights", None)    
    encoder = ml_protocol.get("encoder", "ResNet18")
    num_classes = int(ml_protocol.get("num_classes", 3))
    cuda = ml_protocol.get("cuda", False)
    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

    det = detection_ui(image=image,model=model,features=features,validate=True)
    
    # calculate features in microscope image coords
    hfw = settings.image.hfw
    pixelsize = hfw/settings.image.resolution[0]
    # pixelsize = image.metadata.pixel_size.x
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


def move_feature_to_image_centre(microscope: FibsemMicroscope, settings: MicroscopeSettings, feature: Feature, validate: bool = True,detection_text: str=None):

    features = [feature,ImageCentre()]

    settings.image.beam_type = BeamType.ION

    det = detect_features_v2(microscope, settings, features, validate)

    is_centred = message_box_ui(title="Feature Centring", text=detection_text, buttons=QMessageBox.Yes | QMessageBox.No)

    feature_centre = det.features[0]

    # move stage to centre of lamella

    if is_centred is False:


        microscope.stable_move(
            settings=settings,
            dx=feature_centre.feature_m.x,
            dy=feature_centre.feature_m.y,
            beam_type=BeamType.ION)
        

        move_feature_to_image_centre(microscope, settings, feature, validate,detection_text=detection_text)


def needle_position_validate(microscope: FibsemMicroscope, settings: MicroscopeSettings):

    ref_images = acquire.take_reference_images(microscope=microscope,image_settings=settings.image)

    viewer = napari.Viewer(ndisplay=2)
    image_widget = FibsemImageSettingsWidget(
        microscope=microscope, 
        image_settings=settings.image,
        viewer=viewer)
    


    image_widget.eb_image = ref_images[0]
    image_widget.ib_image = ref_images[1] 
    
    needle_widget = FibsemManipulatorWidget(
        microscope=microscope,
        settings=settings,
        viewer=viewer,
        image_widget=image_widget,
    )

    viewer.window.add_dock_widget(needle_widget, area="right")

    napari.run()




