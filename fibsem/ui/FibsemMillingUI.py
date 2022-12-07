import logging
import time
from pprint import pprint

import liftout.gui.utils as ui_utils
import napari
import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import acquire, constants, conversions, milling, movement
from fibsem import utils as fibsem_utils
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings, Point
from fibsem.ui import utils as fibsem_ui
from fibsem import patterning
from fibsem import config
from fibsem.ui.qtdesigner_files import MillingUI
from fibsem.patterning import MillingPattern
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel


class FibsemMillingUI(MillingUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        microscope: SdbMicroscopeClient,
        settings: MicroscopeSettings,
        milling_pattern: MillingPattern = MillingPattern.Trench,
        point: Point = None,
        change_pattern: bool = False,
        auto_continue: bool = False,
    ):
        super(FibsemMillingUI, self).__init__()

        # setup ui
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.milling_pattern = milling_pattern
        self.point = point
        self.auto_continue = auto_continue
        self.USER_UPDATED_PROTOCOL = False
        self.CHANGE_PATTERN_ENABLED = change_pattern

        self.setup_ui()
        self.setup_connections()

        milling.setup_milling(
            microscope=self.microscope,
            application_file=self.settings.system.application_file,
            hfw=self.settings.image.hfw,
        )

        self.update_milling_stages()

    def setup_connections(self):

        # combobox
        self.comboBox_select_pattern.addItems(
            pattern.name for pattern in MillingPattern
        )
        self.comboBox_select_pattern.setCurrentText(self.milling_pattern.name)
        self.comboBox_select_pattern.currentTextChanged.connect(
            self.update_milling_stages
        )

        self.comboBox_milling_current.addItems(
            [
                f"{current*constants.SI_TO_NANO:.2f}"
                for current in self.microscope.beams.ion_beam.beam_current.available_values
            ]
        )

        # buttons
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_exit_milling.clicked.connect(self.exit_milling)

        # instructions
        self.label_message.setText(
            f"Double-click to move the pattern, adjust parameters to change pattern dimensions. Press Run Milling to start milling."
        )
        self.comboBox_select_pattern.setEnabled(self.CHANGE_PATTERN_ENABLED)

    def setup_ui(self):

        # take image
        self.settings.image.beam_type = BeamType.ION
        self.image = acquire.new_image(self.microscope, self.settings.image)

        # draw image
        self.viewer.layers.clear()
        self.image_layer = self.viewer.add_image(
            ndi.median_filter(self.image.data, 3), name="Image"
        )
        self.image_layer.mouse_double_click_callbacks.append(
            self._double_click
        )  # append callback
        self.viewer.layers.selection.active = self.image_layer

    def _double_click(self, layer, event):

        coords = layer.world_to_data(event.position)

        image_shape = self.image.data.shape

        if (coords[0] > 0 and coords[0] < image_shape[0]) and (
            coords[1] > 0 and coords[1] < image_shape[1]
        ):
            pass

        else:
            napari.utils.notifications.show_info(
                f"Clicked outside image dimensions. Please click inside the image."
            )
            return

        # for trenches, move the stage, not the pattern
        # (trench should always be in centre of image)
        if self.milling_pattern in [MillingPattern.Trench, MillingPattern.Thin]:

            point = conversions.image_to_microscope_image_coordinates(
                Point(x=coords[1], y=coords[0]),
                self.image.data,
                self.image.metadata.binary_result.pixel_size.x,
            )

            movement.move_stage_relative_with_corrected_movement(
                microscope=self.microscope,
                settings=self.settings,
                dx=point.x,
                dy=point.y,
                beam_type=BeamType.ION,
            )

            # update image
            self.setup_ui()
            coords = None

        # get image coordinate
        if coords is None:
            coords = np.asarray(self.image.data.shape) // 2

        self.point = conversions.image_to_microscope_image_coordinates(
            Point(x=coords[1], y=coords[0]),
            self.image.data,
            self.image.metadata.binary_result.pixel_size.x,
        )

        logging.debug(
            f"Milling, {BeamType.ION}, {self.milling_pattern.name}, p=({coords[1]:.2f}, {coords[0]:.2f})  c=({self.point.x:.2e}, {self.point.y:.2e}) "
        )

        self.update_milling_pattern()

    def update_milling_pattern(self):

        try:
            # draw patterns in microscope
            self.microscope.patterning.clear_patterns()
            all_patterns = []
            for stage_name, stage_settings in self.milling_stages.items():

                patterns = patterning.create_milling_patterns(
                    self.microscope, stage_settings, self.milling_pattern, self.point,
                )
                all_patterns.append(patterns)  # 2D

            # # draw patterns in napari
            fibsem_ui._draw_patterns_in_napari(
                viewer=self.viewer,
                ib_image=self.image,
                eb_image=None,
                all_patterns=all_patterns,
            )

            self.update_estimated_time(all_patterns)

            self.viewer.layers.selection.active = self.image_layer

        except Exception as e:
            napari.utils.notifications.show_info(f"Error: {e}")

    def update_estimated_time(self, patterns: list):

        milling_time_seconds = milling.estimate_milling_time_in_seconds(patterns)
        time_str = fibsem_utils._format_time_seconds(milling_time_seconds)
        self.label_milling_time.setText(f"Estimated Time: {time_str}")

    def update_milling_stages(self):

        self.milling_pattern = MillingPattern[
            self.comboBox_select_pattern.currentText()
        ]

        # get all milling stages for the pattern
        milling_protocol_stages = patterning.get_milling_protocol_stage_settings(
            self.settings, self.milling_pattern
        )

        self.milling_stages = {}
        for i, stage_settings in enumerate(milling_protocol_stages, 1):
            self.milling_stages[f"{self.milling_pattern.name}_{i}"] = stage_settings

        try:
            self.comboBox_milling_stage.disconnect()
        except:
            pass
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.addItems(list(self.milling_stages.keys()))
        self.comboBox_milling_stage.currentTextChanged.connect(
            self.setup_milling_parameters_ui
        )

        # setup parameter ui
        self.setup_milling_parameters_ui()

        # draw milling patterns
        self.update_milling_pattern()

    def setup_milling_parameters_ui(self):

        # remove existing elements
        for i in reversed(range(self.gridLayout_2.count())):
            self.gridLayout_2.itemAt(i).widget().setParent(None)

        i = 0
        current_stage = self.comboBox_milling_stage.currentText()

        self.milling_ui_dict = {}

        for (k, v) in self.milling_stages[current_stage].items():

            if k not in config.NON_CHANGEABLE_MILLING_PARAMETERS:
                if k not in config.NON_SCALED_MILLING_PARAMETERS:
                    v = float(v) * constants.METRE_TO_MICRON

                label = QLabel()
                label.setText(str(k))
                spinBox_value = QtWidgets.QDoubleSpinBox()
                spinBox_value.setValue(v)
                spinBox_value.valueChanged.connect(self.update_milling_settings_from_ui)

                if k == "rotation":
                    spinBox_value.setRange(-360, 360)

                self.gridLayout_2.addWidget(label, i, 0)
                self.gridLayout_2.addWidget(spinBox_value, i, 1)

                self.milling_ui_dict[k] = spinBox_value

                i += 1

        milling_current = (
            self.milling_stages[current_stage]["milling_current"] * constants.SI_TO_NANO
        )
        self.comboBox_milling_current.setCurrentText(f"{milling_current:.2f}")
        try:
            self.comboBox_milling_current.disconnect()
        except:
            pass
        self.comboBox_milling_current.currentTextChanged.connect(
            self.update_milling_settings_from_ui
        )

    def update_milling_settings_from_ui(self):

        # flag that user has changed protocol
        self.USER_UPDATED_PROTOCOL = True

        # map keys to ui widgets
        current_stage = self.comboBox_milling_stage.currentText()
        for k, v in self.milling_ui_dict.items():
            value = v.value()
            if k not in config.NON_SCALED_MILLING_PARAMETERS:
                value = float(value) * constants.MICRON_TO_METRE
            self.milling_stages[current_stage][k] = value

        self.milling_stages[current_stage]["milling_current"] = (
            float(self.comboBox_milling_current.currentText()) * constants.NANO_TO_SI
        )

        self.update_milling_pattern()

    def run_milling(self):
        """Run ion beam milling for the selected milling pattern"""

        logging.info(f"Running milling for {len(self.milling_stages)} Stages")

        # clear state
        self.microscope.imaging.set_active_view(BeamType.ION.value)
        self.microscope.imaging.set_active_device(
            BeamType.ION.value
        )  # set ion beam view
        for stage_name, stage_settings in self.milling_stages.items():

            logging.debug(f"MILLING | {self.millling_pattern} | {stage_name} | {stage_settings}")

            # redraw patterns, and run milling
            self.microscope.patterning.clear_patterns()
            self.patterns = patterning.create_milling_patterns(
                self.microscope, stage_settings, self.milling_pattern, self.point,
            )
            
            milling.run_milling(
                microscope=self.microscope,
                milling_current=stage_settings["milling_current"],
                asynch=True,
            )

            # update progress bar
            time.sleep(3)  # wait for milling to start

            milling_time_seconds = milling.estimate_milling_time_in_seconds(
                [self.patterns]
            )
            logging.info(f"milling time: {milling_time_seconds}")
            dt = 0.1
            progressbar = napari.utils.progress(milling_time_seconds * 1 / dt)
            progressbar.display(msg=f"Milling: {stage_name}")

            # TODO: thread https://forum.image.sc/t/napari-progress-bar-modification-on-the-fly/62496/7
            while self.microscope.patterning.state == "Running":

                # elapsed_time += dt
                # prog_val = int(elapsed_time)
                progressbar.update(1)
                time.sleep(dt)

            logging.info(f"Milling finished: {self.microscope.patterning.state}")
            progressbar.clear()
            progressbar.close()

        # reset to imaging mode
        milling.finish_milling(
            microscope=self.microscope,
            imaging_current=self.settings.system.ion.current,
        )

        # update image
        self.setup_ui()

        # confirm finish
        self.finalise_milling()

    def finalise_milling(self) -> bool:

        if self.auto_continue:
            self.close()
            return

        # ask user if the milling succeeded
        response = fibsem_ui.message_box_ui(
            title="Milling Confirmation", text="Do you want to redo milling?"
        )

        if response:
            logging.info("Redoing milling")

            # re-update patterns
            self.update_milling_pattern()
            return response

        # only update if the protocol has been changed...
        if self.USER_UPDATED_PROTOCOL:
            response = fibsem_ui.message_box_ui(
                title="Save Milling Protocol?",
                text="Do you want to update the protocol to use these milling parameters?",
            )

            if response:
                try:
                    ui_utils.update_milling_protocol_ui(
                        self.milling_pattern, self.milling_stages, self
                    )
                except Exception as e:
                    logging.error(f"Unable to update protocol file: {e}")
        self.close()

    def exit_milling(self):
        self.close()

    def closeEvent(self, event):
        self.microscope.patterning.clear_patterns()
        event.accept()
        self.viewer.window.close()


def main():

    from liftout.config import config as liftout_config
    from fibsem.ui.windows import milling_ui

    microscope, settings = fibsem_utils.setup_session()
    settings = fibsem_utils.load_settings_from_config(
        protocol_path=liftout_config.protocol_path
    )
    settings.image.hfw = 50e-6
    milling_pattern = MillingPattern.Polish
    point = None
    change_pattern = True
    auto_continue = False

    settings.image.hfw = 80e-6

    milling_ui(
        microscope,
        settings,
        milling_pattern,
        point=point,
        change_pattern=change_pattern,
        auto_continue=auto_continue,
    )


if __name__ == "__main__":
    main()
