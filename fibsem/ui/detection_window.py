import logging
import os
import sys
from pprint import pprint

import matplotlib.patches as mpatches
from fibsem import conversions, utils
from fibsem.structures import MicroscopeSettings
from fibsem.ui import utils as fibsem_ui_utils

from fibsem.detection import utils as det_utils
from fibsem.detection.utils import DetectionResult, FeatureType, Point
from fibsem.detection.detection import DetectedFeatures
                                     

from fibsem.ui.qtdesigner_files import detection_dialog as detection_gui
from PyQt5 import QtCore, QtWidgets

class GUIDetectionWindow(detection_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope,
        settings: MicroscopeSettings,
        detected_features: DetectedFeatures,
    ):
        super(GUIDetectionWindow, self).__init__()
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        # microscope settings
        self.microscope = microscope
        self.settings = settings
        self.log_path = os.path.dirname(settings.image.save_path)

        # detection data
        self.detected_features = detected_features
        self.current_selected_feature = 0
        self.current_detection_selected = (
            self.detected_features.features[
                self.current_selected_feature
            ].detection_type,
        )
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

        AUTO_CONTINUE = False
        if AUTO_CONTINUE:
            self.continue_button_pressed() # automatically continue

    def setup_connections(self):

        self.comboBox_detection_type.clear()
        self.comboBox_detection_type.addItems(
            [feature.detection_type.name for feature in self.detected_features.features]
        )
        self.comboBox_detection_type.setCurrentText(
            self.detected_features.features[0].detection_type.name
        )
        self.comboBox_detection_type.currentTextChanged.connect(
            self.update_detection_type
        )

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)

    def update_detection_type(self):

        self.current_selected_feature = self.comboBox_detection_type.currentIndex()
        self.current_detection_selected = self.detected_features.features[
            self.current_selected_feature
        ].detection_type
        logging.info(
            f"Changed to {self.current_selected_feature}, {self.current_detection_selected.name}"
        )

    def on_click(self, event):
        """Redraw the patterns and update the display on user click"""
        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = conversions.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), self.image, self.detected_features.pixelsize
            )

            logging.info(
                f"""on_click: {event.button} | {self.current_detection_selected} | IMAGE COORDINATE | {int(self.xclick)}, {int(self.yclick)} | REAL COORDINATE | {self.center_x:.2e}, {self.center_y:.2e}"""
            )

            # update detection data
            self.detected_features.features[
                self.current_selected_feature
            ].feature_px = Point(self.xclick, self.yclick)

            # logging statistics
            logging.info(
                f"detection | {self.current_detection_selected} | {False}"
            )

            self.logged_detection_types.append(self.current_detection_selected)

            # flag that the user corrected a detection.
            self._USER_CORRECTED = True

            self.update_display()

    def update_display(self):
        """Update the window display. Redraw the crosshair"""

        # TODO: consolidate with plot_detected_features

        # point position, image coordinates
        point_1 = self.detected_features.features[0].feature_px
        point_2 = self.detected_features.features[1].feature_px

        # colours
        c1 = det_utils.DETECTION_TYPE_COLOURS[
            self.detected_features.features[0].detection_type
        ]
        c2 = det_utils.DETECTION_TYPE_COLOURS[
            self.detected_features.features[1].detection_type
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
            color=c1, label=self.detected_features.features[0].detection_type.name
        )
        patch_two = mpatches.Patch(
            color=c2, label=self.detected_features.features[1].detection_type.name
        )
        self.wp.canvas.ax11.legend(handles=[patch_one, patch_two])

        # calculate movement distance
        from fibsem import conversions 
        point_diff_px  = conversions.distance_between_points(point_1, point_2)
        point_diff_m = conversions.convert_point_from_pixel_to_metres(point_diff_px, self.detected_features.pixelsize)
        self.detected_features.distance = point_diff_m  # move from 1 to 2 (reverse direction)

        # update labels
        self.label_movement_header.setText(f"Movement")
        self.label_movement_header.setStyleSheet("font-weight:bold")
        self.label_movement.setText(
            f"""Moving {self.detected_features.features[0].detection_type.name} to {self.detected_features.features[1].detection_type.name}
         \nx distance: {self.detected_features.distance.x*1e6:.2f}um 
         \ny distance: {self.detected_features.distance.y*1e6:.2f}um"""
        )

        self.wp.canvas.draw()
    
    def continue_button_pressed(self):

        logging.info(f"Continue button pressed: {self.sender()}")

        self.close() #exit

    def closeEvent(self, event):
        logging.info("Closing Detection Window")

        # log active learning data...
        logging.info(f"Writing machine learning data to disk...")
        if self._USER_CORRECTED:
            path = os.path.join(self.log_path, "label")
            det_utils.write_data_to_disk(path, self.detected_features)

        # log correct detection types
        for feature in self.detected_features.features:
            if feature.detection_type not in self.logged_detection_types:
                logging.info(
                    f" detection | {self.current_detection_selected} | {True}"
                )

        event.accept()

def main():
    from fibsem.detection.detection import Feature
    import fibsem.ui.windows as fibsem_ui_windows

    microscope, settings = utils.setup_session(protocol_path=r"C:\Users\Admin\Github\autoliftout\liftout\protocol\protocol.yaml")
    
    app = QtWidgets.QApplication([])

    # select features
    # features = [Feature(detection_type=FeatureType.ImageCentre, feature_px=None),
    #             Feature(detection_type=FeatureType.LamellaCentre, feature_px=None)]
    # det = fibsem_ui_windows.detect_features(microscope=microscope, 
    #     settings=settings, ref_image=None, features=features, validate=True)

    features = [Feature(FeatureType.NeedleTip), 
                Feature(FeatureType.LamellaCentre)]
    det = fibsem_ui_windows.detect_features_v2(microscope, settings, features, validate=True)

    pprint(det)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
