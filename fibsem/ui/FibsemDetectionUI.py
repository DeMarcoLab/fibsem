import logging
import os
import sys
from pprint import pprint

import matplotlib.patches as mpatches
from fibsem import conversions, utils
from fibsem.structures import BeamType, MicroscopeSettings
from fibsem.ui import utils as fibsem_ui_utils

from fibsem.detection import utils as det_utils
from fibsem.detection.utils import FeatureType, Point
from fibsem.detection.detection import DetectedFeatures
                                     

from fibsem.ui.qtdesigner_files import detection_dialog as detection_gui
from PyQt5 import QtCore, QtWidgets

# TODO: convert to napari
class FibsemDetectionUI(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope,
        settings: MicroscopeSettings,
        detected_features: DetectedFeatures,
    ):
        super(FibsemDetectionUI, self).__init__()
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        # microscope settings
        self.microscope = microscope
        self.settings = settings
        self.log_path = os.path.dirname(settings.image.save_path)

        # detection data
        self.detected_features = detected_features
        self.selected_feature_idx = 0
        self.selected_feature = self.detected_features.features[
                self.selected_feature_idx
            ].type
        self.logged_detection_types = []

        # images
        self.image = self.detected_features.image
        self._USER_CORRECTED = False

        # pattern drawing
        self.wp = fibsem_ui_utils._WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.wp.canvas.mpl_connect("button_press_event", self.on_click)

        self.setup_connections()
        self.update_display()

        AUTO_CONTINUE = False # TODO:
        if AUTO_CONTINUE:
            self.continue_button_pressed() # automatically continue

    def setup_connections(self):

        self.comboBox_detection_type.clear()
        self.comboBox_detection_type.addItems(
            [feature.type.name for feature in self.detected_features.features]
        )
        self.comboBox_detection_type.setCurrentText(
            self.detected_features.features[0].type.name
        )
        self.comboBox_detection_type.currentTextChanged.connect(
            self.update_detection_type
        )

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)

    def update_detection_type(self):

        self.selected_feature_idx = self.comboBox_detection_type.currentIndex()
        self.selected_feature = self.detected_features.features[
            self.selected_feature_idx
        ].type
        logging.info(
            f"Changed to {self.selected_feature_idx}, {self.selected_feature.name}"
        )

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata

            self.point_m = conversions.image_to_microscope_image_coordinates(Point(self.xclick, self.yclick), self.image, self.detected_features.pixelsize)

            logging.debug(
                f"""DectectedFeature {self.selected_feature.name} | IMAGE COORDINATE | {int(self.xclick)}, {int(self.yclick)} | REAL COORDINATE | {self.point_m.x:.2e}, {self.point_m.y:.2e}"""
                
            )
            logging.debug(f"DectectedFeature {self.selected_feature.name} | COORD {int(self.xclick)}, {int(self.yclick)} | SHIFT | {self.point_m.x:.2e}, {self.point_m.y:.2e} | {self.settings.image.beam_type}")

            # update detection data
            self.detected_features.features[
                self.selected_feature_idx
            ].feature_px = Point(self.xclick, self.yclick)

            # logging statistics
            logging.debug(
                f"Feature | {self.selected_feature} | {False}"
            )

            self.logged_detection_types.append(self.selected_feature)

            # flag that the user corrected a detection.
            self._USER_CORRECTED = True

            self.update_display()

    def update_display(self):
        """Update the window display. Redraw the crosshair"""


        # point position, image coordinates
        point_1 = self.detected_features.features[0].feature_px
        point_2 = self.detected_features.features[1].feature_px

        # colours
        c1 = det_utils.DETECTION_TYPE_COLOURS[
            self.detected_features.features[0].type
        ]
        c2 = det_utils.DETECTION_TYPE_COLOURS[
            self.detected_features.features[1].type
        ]

        # redraw all crosshairs
        self.wp.canvas.ax11.patches.clear()

        # draw cross hairs
        fibsem_ui_utils.draw_crosshair(
            self.image, self.wp.canvas, x=point_1.x, y=point_1.y, colour=c1,
        )
        fibsem_ui_utils.draw_crosshair(
            self.image, self.wp.canvas, x=point_2.x, y=point_2.y, colour=c2,
        )

        # draw arrow
        fibsem_ui_utils.draw_arrow(point_1, point_2, self.wp.canvas)

        # legend
        patch_one = mpatches.Patch(
            color=c1, label=self.detected_features.features[0].type.name
        )
        patch_two = mpatches.Patch(
            color=c2, label=self.detected_features.features[1].type.name
        )
        self.wp.canvas.ax11.legend(handles=[patch_one, patch_two])

        # calculate movement distance
        point_diff_px  = conversions.distance_between_points(point_1, point_2)
        point_diff_m = conversions.convert_point_from_pixel_to_metres(point_diff_px, self.detected_features.pixelsize)
        self.detected_features.distance = point_diff_m  # move from 1 to 2 (reverse direction)

        # update labels
        self.label_movement_header.setText(f"Movement")
        self.label_movement_header.setStyleSheet("font-weight:bold")
        self.label_movement.setText(
            f"""Moving {self.detected_features.features[0].type.name} to {self.detected_features.features[1].type.name}
         \nx distance: {self.detected_features.distance.x*1e6:.2f}um 
         \ny distance: {self.detected_features.distance.y*1e6:.2f}um"""
        )

        self.wp.canvas.draw()
    
    def continue_button_pressed(self):

        self.close() #exit

    def closeEvent(self, event):
        """Override the close event to save the data"""
        # log active learning data...
        if self._USER_CORRECTED:
            path = os.path.join(self.log_path, "label")
            det_utils.write_data_to_disk(path, self.detected_features)

        # log correct detection types
        for feature in self.detected_features.features:
            if feature.type not in self.logged_detection_types:
                logging.debug(
                    f"Feature | {feature.type} | {True}"
                )

        event.accept()




def main():
    from fibsem.detection.detection import Feature, move_based_on_detection
    import fibsem.ui.windows as fibsem_ui_windows
    from fibsem import movement, acquire
    from liftout import actions 


    microscope, settings = utils.setup_session(protocol_path=r"C:\Users\Admin\Github\autoliftout\liftout\protocol\protocol.yaml")
    


    app = QtWidgets.QApplication([])

    import random
    from liftout.autoliftout import AutoLiftoutMode, mill_lamella_edge
    from liftout.structures import ReferenceHFW
    from fibsem.detection import detection
    from fibsem import calibration
    mode = AutoLiftoutMode.Manual
    from liftout.structures import Lamella

    # charge neutralisation  to charge lamella
    settings.image.beam_type = BeamType.ION
    n_iter = int(settings.protocol["options"].get("liftout_charge_neutralisation_iterations", 35))
    calibration.auto_charge_neutralisation(microscope, settings.image, n_iterations=n_iter)
    
    # calibration.auto_charge_neutralisation(microscope, settings.image, n_iterations=20, save_images=True)

    # lamella = Lamella("")
    # lamella = mill_lamella_edge(microscope, settings,  lamella=lamella, validate=True,)

    # above_brightness_threshold = False
    
    # response = fibsem_ui_windows.ask_user_interaction(
    #     msg=f"Above Threshold: {above_brightness_threshold} \nHas the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement.",
    #     image = acquire.last_image(microscope, settings.image.beam_type).data
    # )


    return

    beam_types = [BeamType.ELECTRON, BeamType.ELECTRON, BeamType.ION, BeamType.ION]
    euc_move = [False, False, True, True]
    hfws = [ReferenceHFW.Medium.value, ReferenceHFW.High.value, ReferenceHFW.Medium.value, ReferenceHFW.High.value]
    mask_radii = [512, 256, 512, 256]

    for beam_type, eucentric_move, hfw, m_radius in zip(beam_types, euc_move, hfws, mask_radii):

        settings.image.beam_type = beam_type
        settings.image.hfw = hfw

        det = fibsem_ui_windows.detect_features_v2(
            microscope=microscope,
            settings=settings,
            features=[
                Feature(FeatureType.LamellaCentre),
                Feature(FeatureType.ImageCentre),
            ],
            validate=bool(mode is AutoLiftoutMode.Manual),
            mask_radius=m_radius
        )

        detection.move_based_on_detection(
            microscope, 
            settings, 
            det, 
            beam_type=settings.image.beam_type, 
            eucentric_move=eucentric_move
        )

    settings.image.hfw = ReferenceHFW.Medium.value
    acquire.take_reference_images(microscope, settings.image)

    return

    # beam_type = BeamType.ELECTRON
    features = [Feature(FeatureType.NeedleTip), 
                Feature(FeatureType.LamellaCentre)]

        
    for beam_type in [BeamType.ELECTRON, BeamType.ELECTRON]:

        settings.image.beam_type = beam_type
        settings.image.hfw = 400e-6

    #     # actions.move_needle_to_liftout_position(microscope)

        det = fibsem_ui_windows.detect_features_v2(microscope, settings, features, validate=True)

    #     print("features: ", det.features)
    #     print("distance: ", det.distance)
    #     print("feature 1 position: ", det.features[0].feature_m)

    VALIDATE = True
    from liftout import autoliftout


    # while True:

        # movement.retract_needle(microscope)

        # autoliftout.landing_entry_procedure(microscope, settings, VALIDATE)


    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

