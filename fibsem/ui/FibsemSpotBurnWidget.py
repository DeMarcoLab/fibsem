import logging
from typing import List

import napari
import napari.utils.notifications
from napari.layers import Image as NapariImageLayer
from napari.layers import Points as NapariPointsLayer
from napari.qt.threading import FunctionWorker, thread_worker
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

from fibsem.imaging.spot import run_spot_burn
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point
from fibsem.ui import stylesheets
from fibsem.ui.qtdesigner_files import FibsemSpotBurnWidget as FibsemSpotBurnWidgetUI
from fibsem.utils import format_value

SPOT_BURN_POINTS_LAYER_NAME = "spot-burn-points"
DEFAULT_BEAM_CURRENT = 60e-12  # 60 pA

class FibsemSpotBurnWidget(FibsemSpotBurnWidgetUI.Ui_Form, QWidget):
    spot_burn_progress_signal = pyqtSignal(dict)

    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.parent = parent
        self.viewer: napari.Viewer = parent.viewer
        self.microscope: FibsemMicroscope = parent.microscope
        self.worker: FunctionWorker = None

        # napari layers
        self.pts_layer: NapariPointsLayer = None
        self.image_layer: NapariImageLayer = None

        self.setup_connections()

    def setup_connections(self):
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

        beam_currents = self.microscope.get_available_values("current", BeamType.ION)
        for current in beam_currents:
            label = format_value(current, unit="A", precision=1)
            self.comboBox_beam_current.addItem(label, current)

        # find the current closest to 60pA, set index to that
        closest_current = min(beam_currents, key=lambda x: abs(x - DEFAULT_BEAM_CURRENT))
        closest_index = beam_currents.index(closest_current)
        self.comboBox_beam_current.setCurrentIndex(closest_index)

        # set parameters for exposure time
        self.doubleSpinBox_exposure_time.setSuffix(" s")
        self.doubleSpinBox_exposure_time.setRange(0.1, 60)
        self.doubleSpinBox_exposure_time.setValue(10)
        self.doubleSpinBox_exposure_time.valueChanged.connect(self._on_data_changed)

        # initial state
        self.label_information.setText("No points selected. Please add points to the layer.")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GRAY_PUSHBUTTON_STYLE)
        self.pushButton_run_spot_burn.setEnabled(False)

        # progress bar
        self.progressBar.setVisible(False)
        self.spot_burn_progress_signal.connect(self._update_progress_bar)

    def set_active(self):
        """Called when the widget is activated."""
        # check if the points layer exists, if not create it
        if SPOT_BURN_POINTS_LAYER_NAME not in self.viewer.layers:
            self.pts_layer = self.viewer.add_points(data=[],
                                   name=SPOT_BURN_POINTS_LAYER_NAME,
                                   visible=True,
                                   size=20)
            self.pts_layer.events.data.connect(self._on_data_changed)
        else:
            self.pts_layer = self.viewer.layers[SPOT_BURN_POINTS_LAYER_NAME]

        self.viewer.layers.selection.active = self.pts_layer
        self.pts_layer.visible = True
        self.pts_layer.mode = "add"

    def set_inactive(self):
        """Called when the widget is deactivated."""

        # hide the points layer
        if SPOT_BURN_POINTS_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[SPOT_BURN_POINTS_LAYER_NAME].visible = False

        self.parent.image_widget.restore_active_layer_for_movement()

    def _on_data_changed(self, event = None):
        """Called when the data in the points layer changes."""
        coordinates = self.pts_layer.data
        
        enabled = bool(len(coordinates) > 0)
        self.pushButton_run_spot_burn.setEnabled(enabled)
        if enabled:
            self.label_information.setText(f"Selected {len(coordinates)} points. Estimated time: {len(coordinates) * self.doubleSpinBox_exposure_time.value()} seconds")
            self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        else:
            self.label_information.setText("No points selected. Please add points to the layer.")
            self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GRAY_PUSHBUTTON_STYLE)

    def run_spot_burn_worker(self):
        """Run the spot burn worker."""

        # get the points layer
        if SPOT_BURN_POINTS_LAYER_NAME not in self.viewer.layers:
            napari.utils.notifications.show_warning("No points layer found. Requires 'spot-burn-points' layer.")
            return

        # check if there is a points layer, and that it has points in it
        if self.pts_layer is None:
            napari.utils.notifications.show_warning("No points layer found.")
            return
    
        if len(self.pts_layer.data) == 0:
            napari.utils.notifications.show_warning("No points selected.")
            return

        # get the fib image parameters
        self.image_layer: NapariImageLayer = self.parent.image_widget.ib_layer
        layer_translated = self.pts_layer.data - self.image_layer.translate
        image_shape = self.image_layer.data.shape

        # convert to relative image coordinates (0 - 1)
        coordinates = [Point(x=pt[1]/image_shape[1], y=pt[0] / image_shape[0]) for pt in layer_translated]

        # exclude points outside of image bounds
        coordinates = [pt for pt in coordinates if 0 <= pt.x <= 1 and 0 <= pt.y <= 1]

        if len(coordinates) == 0:
            napari.utils.notifications.show_warning("No points selected within FIB image bounds.")
            return

        beam_current = self.comboBox_beam_current.currentData()     # amps
        exposure_time = self.doubleSpinBox_exposure_time.value()    # seconds

        logging.info(f"Running spot burn with {len(coordinates)} points. Beam current: {beam_current} A, exposure time: {exposure_time} s")

        self.worker = self._spot_burn_worker(microscope=self.microscope,
                                             coordinates=coordinates,
                                             exposure_time=exposure_time,
                                             milling_current=beam_current,)
        self.worker.returned.connect(self.spot_burn_finished)
        self.worker.errored.connect(self.spot_burn_errored)
        self.worker.start()

    @thread_worker
    def _spot_burn_worker(self, microscope: FibsemMicroscope, coordinates: List[Point], exposure_time: float, milling_current: float):
        """Worker function to run the spot burn."""
        self.pushButton_run_spot_burn.setEnabled(False)
        self.pushButton_run_spot_burn.setText("Burning Spot...")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)

        run_spot_burn(microscope=microscope,
                       coordinates=coordinates,
                       exposure_time=exposure_time,
                       milling_current=milling_current,
                       beam_type=BeamType.ION,
                       parent_ui=self)

    def spot_burn_finished(self, result):
        """Called when the spot burn is finished."""
        self.pushButton_run_spot_burn.setEnabled(True)
        self.pushButton_run_spot_burn.setText("Burn Spot")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

    def spot_burn_errored(self, error):
        """Called when the spot burn fails."""
        logging.error(f"Spot burn failed: {error}")
        self.spot_burn_progress_signal.emit({"finished": True})
        self.spot_burn_finished(error)

    def _update_progress_bar(self, ddict: dict):
        """Update the progress bar with the current progress.
        
        Parameters
        ----------
        ddict : dict
            Dictionary with the following keys:
            - total_points (int): Total number of points to burn
            - current_point (int): Current point being burned
            - remaining_time (float): Remaining time for current point in seconds
            - total_remaining_time (float): Total remaining time in seconds
            - total_estimated_time (float): Total estimated time in seconds
            - finished (bool): Whether the spot burn is finished
        """        
        total_points = ddict.get("total_points", 0)
        current_point = ddict.get("current_point", 0)
        remaining_time = ddict.get("remaining_time", 0)
        total_remaining_time = ddict.get("total_remaining_time", 0)
        total_estimated_time = ddict.get("total_estimated_time", 0)
        total_elapsed_time = total_estimated_time - total_remaining_time
        finished = ddict.get("finished", False)

        self.progressBar.setVisible(True)
        self.progressBar.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.progressBar.setMinimum(0)
        
        if total_elapsed_time > 0 and total_estimated_time > 0:
            self.progressBar.setMaximum(total_estimated_time)
            self.progressBar.setValue(total_elapsed_time)
            self.progressBar.setTextVisible(True)
            self.progressBar.setFormat(f"Burning Spot {current_point}/{total_points}... {int(remaining_time)}s remaining")
        else:
            self.progressBar.setValue(0)
            self.progressBar.setFormat("Preparing Spot Burn...")

        if finished:
            self.progressBar.setValue(total_estimated_time)
            self.progressBar.setFormat("Spot Burn Finished")

