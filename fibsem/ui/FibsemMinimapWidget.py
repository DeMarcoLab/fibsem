import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Layer as NapariLayer
from napari.layers import Points as NapariPointsLayer
from napari.qt.threading import thread_worker
from napari.utils.events import Event as NapariEvent
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from scipy.ndimage import median_filter

from fibsem import config as cfg
from fibsem import constants, conversions
from fibsem.imaging import tiled
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.ui import FibsemImageSettingsWidget, FibsemMovementWidget, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    remove_all_napari_shapes_layers,
)
from fibsem.ui.napari.properties import (
    CORRELATION_IMAGE_LAYER_PROPERTIES,
    GRIDBAR_IMAGE_LAYER_PROPERTIES,
    OVERVIEW_IMAGE_LAYER_PROPERTIES,
)
from fibsem.ui.napari.utilities import draw_positions_in_napari, is_inside_image_bounds
from fibsem.ui.qtdesigner_files import FibsemMinimapWidget as FibsemMinimapWidgetUI

try:
    from autolamella.protocol import AutoLamellaProtocol
except ImportError:
    AutoLamellaProtocol = None

TRENCH_KEY, MILL_ROUGH_KEY = "trench", "mill_rough" # TODO: replace with autolamella.protocol.validation
COLOURS = CORRELATION_IMAGE_LAYER_PROPERTIES["colours"]

TILE_COUNTS = [f"{i}x{i}" for i in range(1, 8)]
DEFAULT_TILE_COUNT = TILE_COUNTS[2] # 3x3 grid
DEFAULT_FOV = 500 # um
DEFAULT_DWELL_TIME = 1.0 # us

def generate_gridbar_image(shape: Tuple[int, int], pixelsize: float, spacing: float, width: float) -> FibsemImage:
    """Generate an synthetic image of cryo gridbars."""
    arr = np.zeros(shape=shape, dtype=np.uint8)

    # create grid, grid bars thickness = 10px
    thickness_px = int(width / pixelsize)
    spacing_px = int(spacing / pixelsize)
    for i in range(0, arr.shape[0], spacing_px):
        arr[i:i+thickness_px, :] = 255
        arr[:, i:i+thickness_px] = 255

    # TODO: add metadata
    return FibsemImage(data=arr)

def _get_total_fov(tile_count: int, fov: float) -> float:
    """Calculate the total field of view for the tile collection."""
    return tile_count * fov

# TODO: migrate to properly scaled infinite canvas
# TODO: allow acquiring multiple overview images
# TODO: show stage orientation
class FibsemMinimapWidget(FibsemMinimapWidgetUI.Ui_MainWindow, QtWidgets.QMainWindow):
    stage_position_added_signal = pyqtSignal(FibsemStagePosition)
    stage_position_updated_signal = pyqtSignal(FibsemStagePosition)
    stage_position_removed_signal = pyqtSignal(FibsemStagePosition)
    tile_acquisition_progress_signal = pyqtSignal(dict)
    
    def __init__(
        self,
        viewer: napari.Viewer,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent

        self.microscope: FibsemMicroscope = self.parent.microscope
        self.protocol = None
        if hasattr(self.parent, "protocol"):
            self.protocol: 'AutoLamellaProtocol' = deepcopy(self.parent.protocol)
        self.movement_widget: FibsemMovementWidget = self.parent.movement_widget
        self.image_widget: FibsemImageSettingsWidget = self.parent.image_widget

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.image: FibsemImage = None
        self.image_layer: NapariImageLayer  = None
        self.position_layer: NapariPointsLayer = None
        self.correlation_image_layers: List[str] = []
        self.milling_pattern_layers: List[str] = []

        self.positions: List[FibsemStagePosition] = []
        self.correlation_mode_enabled: bool = False

        self.ADDING_POSITION: bool = False
        self.STOP_ACQUISITION: bool = False

        self.setup_connections()

    def setup_connections(self):
        
        # acquisition buttons
        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)
        self.actionLoad_Image.triggered.connect(self.load_image)
        
        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])
        self.lineEdit_tile_path.setText(str(self.image_widget.image_settings.path)) # set default path
        self.doubleSpinBox_tile_fov.setValue(DEFAULT_FOV)
        self.doubleSpinBox_tile_dwell_time.setValue(DEFAULT_DWELL_TIME)
        self.comboBox_tile_resolution.addItems(cfg.SQUARE_RESOLUTIONS)
        self.comboBox_tile_resolution.setCurrentText(cfg.DEFAULT_SQUARE_RESOLUTION)
        self.comboBox_tile_count.addItems(TILE_COUNTS)
        self.comboBox_tile_count.setCurrentText(DEFAULT_TILE_COUNT)
        self.comboBox_tile_count.currentIndexChanged.connect(self.update_imaging_display)
        self.doubleSpinBox_tile_fov.valueChanged.connect(self.update_imaging_display)
        self.update_imaging_display() # update the total fov

        # TODO: connect event to calculate total fov and time, tile_count * fov
        # TODO: calculate estimate time for acquisition

        # position buttons
        self.pushButton_move_to_position.clicked.connect(self.move_to_position_pressed)
        self.comboBox_tile_position.currentIndexChanged.connect(self.update_current_selected_position)
        self.pushButton_remove_position.clicked.connect(self.remove_selected_position_pressed)
        
        # disable updating position name:
        self.label_position_name.setVisible(False)
        self.lineEdit_tile_position_name.setVisible(False)
        self.pushButton_update_position.setVisible(False)

        # signals
        self.tile_acquisition_progress_signal.connect(self.handle_tile_acquisition_progress)
        
        # refresh stage position ui, when added, updated or removed
        self.stage_position_added_signal.connect(self.handle_stage_position_signals)
        self.stage_position_updated_signal.connect(self.handle_stage_position_signals)
        self.stage_position_removed_signal.connect(self.handle_stage_position_signals)

        # update the positions from the parent   
        if hasattr(self.parent, "sync_positions_to_minimap_signal"):
            self.parent.sync_positions_to_minimap_signal.connect(self.update_positions_from_parent)
        
        if hasattr(self.parent, "lamella_created_signal"):
            self.parent.lamella_created_signal.connect(self.update_from_created_lamella)

        # handle movement progress
        self.movement_widget.movement_progress_signal.connect(self.handle_movement_progress)

        # pattern overlay
        AVAILABLE_MILLING_PATTERNS = []
        if self.protocol is not None:
            AVAILABLE_MILLING_PATTERNS = [k for k in self.protocol.milling.keys()]
            self.comboBox_pattern_overlay.addItems(AVAILABLE_MILLING_PATTERNS)
            if TRENCH_KEY in AVAILABLE_MILLING_PATTERNS:
                self.comboBox_pattern_overlay.setCurrentText(TRENCH_KEY)
            elif MILL_ROUGH_KEY in AVAILABLE_MILLING_PATTERNS:
                self.comboBox_pattern_overlay.setCurrentText(MILL_ROUGH_KEY)
            self.comboBox_pattern_overlay.currentIndexChanged.connect(self.redraw_pattern_overlay)
            self.checkBox_pattern_overlay.stateChanged.connect(self.redraw_pattern_overlay)

        if not AVAILABLE_MILLING_PATTERNS:
            self.checkBox_pattern_overlay.setEnabled(False)
            self.comboBox_pattern_overlay.setToolTip("No milling patterns available.")

        # correlation
        self.actionLoad_Correlation_Image.triggered.connect(self.load_image)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        self.pushButton_enable_correlation.clicked.connect(self._toggle_correlation_mode)
        self.viewer.bind_key("C", self._toggle_correlation_mode)
        self.pushButton_enable_correlation.setEnabled(False) # disabled until correlation images added

        # gridbar controls
        self.groupBox_correlation.setEnabled(True) # only grid-bar overlay enabled
        self.checkBox_gridbar.setEnabled(True)
        self.checkBox_gridbar.stateChanged.connect(self.toggle_gridbar_display)
        self.label_gb_spacing.setVisible(False)
        self.label_gb_width.setVisible(False)
        self.doubleSpinBox_gb_spacing.setVisible(False)
        self.doubleSpinBox_gb_width.setVisible(False)
        self.doubleSpinBox_gb_spacing.setValue(GRIDBAR_IMAGE_LAYER_PROPERTIES["spacing"])
        self.doubleSpinBox_gb_width.setValue(GRIDBAR_IMAGE_LAYER_PROPERTIES["width"])
        self.doubleSpinBox_gb_spacing.setKeyboardTracking(False)
        self.doubleSpinBox_gb_width.setKeyboardTracking(False)
        self.doubleSpinBox_gb_spacing.valueChanged.connect(self.update_gridbar_layer)
        self.doubleSpinBox_gb_width.valueChanged.connect(self.update_gridbar_layer)
        self.groupBox_correlation.setToolTip("Correlation Controls are disabled until an image is acquired or loaded.")

        # set styles
        self.pushButton_update_position.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_run_tile_collection.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_cancel_acquisition.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.progressBar_acquisition.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.pushButton_remove_position.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_move_to_position.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_enable_correlation.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        self.update_positions_combobox()
        self.toggle_interaction(enable=True)

    def update_positions_from_parent(self, positions):
        # TODO: support proper shared data model
        if positions is not None:
            self.positions = positions

        if not self.ADDING_POSITION:
            self.update_positions_combobox()
            self.update_viewer()

    def update_from_created_lamella(self, lamella: 'Lamella') -> None:
        """Update the latest position name from the created lamella (via AutoLamella)."""
        self.positions[-1].name = lamella.name
        self.update_viewer()
        self.update_positions_from_parent(None)

    def get_imaging_parameters(self) -> Dict[str, Any]:
        """Get the imaging parameters from the UI."""
        return {
            "fov": self.doubleSpinBox_tile_fov.value() * constants.MICRO_TO_SI,
            "resolution": list(map(int, self.comboBox_tile_resolution.currentText().split("x"))),
            "tile_count": list(self.comboBox_tile_count.currentText().split("x")),
            "dwell_time": self.doubleSpinBox_tile_dwell_time.value() * constants.MICRO_TO_SI,
            "beam_type": BeamType[self.comboBox_tile_beam_type.currentText()],
            "cryo": self.checkBox_tile_autogamma.isChecked(),
            "autocontrast": self.checkBox_tile_autogamma.isChecked(),
            "path": self.lineEdit_tile_path.text(),
            "filename": self.lineEdit_tile_filename.text(),
        }

    def update_imaging_display(self):
        """Update the imaging parameters based on the field of view and tile count."""
        # update imaging parameters based on fov and tile count
        imaging_params = self.get_imaging_parameters()
        fov = imaging_params["fov"] 
        tile_count = int(imaging_params["tile_count"][0])
        total_fov = int(_get_total_fov(tile_count, fov) * 1e6) # um
        self.label_tile_total_fov.setText(f"Total Field of View: {total_fov} um")
   
    def run_tile_collection(self):
        """Run the tiled acquisition."""
        logging.info("running tile collection")

        imaging_params = self.get_imaging_parameters()
        fov = imaging_params["fov"] 
        resolution = imaging_params["resolution"]
        dwell_time = imaging_params["dwell_time"]
        beam_type = imaging_params["beam_type"]
        cryo = imaging_params["cryo"]
        autocontrast = imaging_params["autocontrast"]
        path = imaging_params["path"]
        filename = imaging_params["filename"]
        tile_count = int(imaging_params["tile_count"][0])

        image_settings = ImageSettings(
            hfw = fov,
            resolution = resolution,
            dwell_time = dwell_time,
            beam_type = beam_type,
            autocontrast = autocontrast,
            save = True,
            path = path,
            filename = filename,
        )
        total_fov = _get_total_fov(tile_count, fov)

        # TODO: support overlap, better stitching (non-existent)
        if image_settings.filename == "":
            napari.utils.notifications.show_error("Please enter a filename for the image")
            return

        # ui feedback
        self.toggle_interaction(enable=False)

        self.STOP_ACQUISITION = False
        worker = self.run_tile_collection_thread(
            microscope=self.microscope, image_settings=image_settings, 
            grid_size=total_fov,
            tile_size=fov, 
            overlap=0, 
            cryo=cryo)

        worker.finished.connect(self.tile_collection_finished)
        worker.errored.connect(self.tile_collection_errored)
        worker.start()


    def tile_collection_finished(self):
        napari.utils.notifications.show_info("Tile collection finished.")
        self.update_viewer(self.image)
        self.toggle_interaction(enable=True)
        self.STOP_ACQUISITION = False

    def tile_collection_errored(self):
        logging.error("Tile collection errored.")
        self.STOP_ACQUISITION = False
        # TODO: handle when acquisition is cancelled halfway, clear viewer, etc

    @thread_worker
    def run_tile_collection_thread(self, 
                                   microscope: FibsemMicroscope, 
                                   image_settings: ImageSettings,
                                    grid_size: float, 
                                    tile_size:float, 
                                    overlap:float=0, 
                                    cryo: bool=True):
        """Threaded worker for tiled acquisition and stitching."""
        try:
            self.image = tiled.tiled_image_acquisition_and_stitch(
                microscope=microscope,
                image_settings=image_settings,
                grid_size=grid_size,
                tile_size=tile_size,
                overlap=overlap,
                cryo=cryo,
                parent_ui=self,
            )
        except Exception as e:
            # TODO: specify the error, user cancelled, or error in acquisition
            logging.error(f"Error in tile collection: {e}")

    def handle_tile_acquisition_progress(self, ddict: dict) -> None:
        """Callback for handling the tile acquisition progress."""

        msg = f"{ddict['msg']} ({ddict['counter']}/{ddict['total']})"
        logging.info(msg)
        napari.utils.notifications.show_info(msg)

        # progress bar
        count, total = ddict["counter"], ddict["total"]
        self.progressBar_acquisition.setMaximum(100)
        self.progressBar_acquisition.setValue(int(count/total*100))

        image = ddict.get("image", None)
        if image is not None:
            self.update_viewer(FibsemImage(data=image), tmp=True) # TODO: this gets too slow when there are lots of tiles, update only the new tile

    def cancel_acquisition(self):
        """Cancel the tiled acquisition."""
        logging.info("Cancelling acquisition...")
        self.STOP_ACQUISITION: bool = True

    def toggle_gridbar_display(self):
        """Toggle the display of the synthetic grid bar overlay."""
        show_gridbar = self.checkBox_gridbar.isChecked()
        self.label_gb_spacing.setVisible(show_gridbar)
        self.label_gb_width.setVisible(show_gridbar)
        self.doubleSpinBox_gb_spacing.setVisible(show_gridbar)
        self.doubleSpinBox_gb_width.setVisible(show_gridbar)

        if show_gridbar:
            self.update_gridbar_layer()
        else:
            layer_name = GRIDBAR_IMAGE_LAYER_PROPERTIES["name"]
            if layer_name in self.viewer.layers:
                self.correlation_image_layers.remove(layer_name)
                self.viewer.layers.remove(layer_name)

            self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
            self.comboBox_correlation_selected_layer.clear()
            self.comboBox_correlation_selected_layer.addItems([layer.name for layer in self.viewer.layers if "correlation-image" in layer.name ])
            self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
            # if no correlation layers left, disable enable correlation
            if len(self.comboBox_correlation_selected_layer) == 0:
                self.pushButton_enable_correlation.setEnabled(False)

    def update_gridbar_layer(self):
        """Update the synthetic grid bar overlay."""
        # update gridbars image
        spacing = self.doubleSpinBox_gb_spacing.value() * constants.MICRO_TO_SI
        width = self.doubleSpinBox_gb_width.value() * constants.MICRO_TO_SI
        gridbars_image = generate_gridbar_image(shape=self.image.data.shape, 
                                                pixelsize=self.image.metadata.pixel_size.x, 
                                                spacing=spacing, width=width)

        # update gridbar layer
        gridbar_layer = GRIDBAR_IMAGE_LAYER_PROPERTIES["name"]
        if gridbar_layer in self.viewer.layers:
            self.viewer.layers[gridbar_layer].data = gridbars_image.data
        else:
            self.add_correlation_image(gridbars_image, is_gridbar=True)

    def toggle_interaction(self, enable: bool = True):
        """Toggle the interactivity of the UI elements."""
        self.pushButton_run_tile_collection.setEnabled(enable)
        self.pushButton_cancel_acquisition.setVisible(not enable)
        self.progressBar_acquisition.setVisible(not enable)
        # reset progress bar
        self.progressBar_acquisition.setValue(0)

        if enable:
            self.pushButton_run_tile_collection.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
            self.pushButton_run_tile_collection.setText("Run Tile Collection")
        else:
            self.pushButton_run_tile_collection.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_run_tile_collection.setText("Running Tile Collection...")

        if self.image is None:
            self.label_instructions.setText("Please take or load an overview image...")

    def load_image(self):
        """Ask the user to select a file to load an image as overview or correlation image."""
        is_correlation = self.sender() == self.actionLoad_Correlation_Image

        path = ui_utils.open_existing_file_dialog(
            msg="Select image to load", 
            path=self.lineEdit_tile_path.text(), 
            _filter="Image Files (*.tif *.tiff)", 
            parent=self)

        if path == "":
            napari.utils.notifications.show_error("No file selected..")
            return

        # load the image
        image = FibsemImage.load(path)
        
        if is_correlation:
            self.add_correlation_image(image)
        else:
            self.update_viewer(image)

    def update_viewer(self, image: FibsemImage = None, tmp: bool = False):
        """Update the viewer with the image and positions."""
        if image is not None:
    
            if not tmp:
                self.image = image
            
            # apply a median filter to the image
            arr = median_filter(image.data, 
                                size=OVERVIEW_IMAGE_LAYER_PROPERTIES["median_filter_size"])

            try:
                self.image_layer.data = arr
            except Exception as e:              
                self.image_layer = self.viewer.add_image(arr, 
                                                         name=OVERVIEW_IMAGE_LAYER_PROPERTIES["name"],
                                                         colormap=OVERVIEW_IMAGE_LAYER_PROPERTIES["colormap"], 
                                                         blending=OVERVIEW_IMAGE_LAYER_PROPERTIES["blending"])

            if tmp:
                return # don't update the rest of the UI, we are just updating the image

            self.image_layer.mouse_drag_callbacks.clear()
            self.image_layer.mouse_double_click_callbacks.clear()
            self.image_layer.mouse_drag_callbacks.append(self.on_single_click)
            self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)

            # NOTE: how to do respace scaling, convert to infinite canvas
            # px = self.image.metadata.pixel_size.x
            # self.image_layer.scale = [px*constants.SI_TO_MICRO, px*constants.SI_TO_MICRO]
            # self.viewer.scale_bar.visible = True
            # self.viewer.scale_bar.unit = "um"

        if self.image:
            self.draw_current_stage_position()  # draw the current stage position on the image
            self.draw_stage_positions()         # draw the reprojected positions on the image
            self.label_instructions.setText("Alt+Click to Add a position, Shift+Click to Update a position \nor Double Click to Move the Stage...")
        self.set_active_layer_for_movement()

    def get_coordinate_in_microscope_coordinates(self, layer: NapariLayer, event: NapariEvent) -> Tuple[np.ndarray, Point]:
        """Validate if event position is inside image, and convert to microscope coords
        Args:
            layer (NapariLayer): The image layer.
            event (Event): The event object.
        Returns:
            Tuple[np.ndarray, Point]: The coordinates in image and microscope image coordinates.
        """         
        # get coords in image coordinates (adjusts for translation, etc)
        coords = layer.world_to_data(event.position)

        # check if clicked point is inside image
        if not is_inside_image_bounds(coords=coords, shape=self.image.data.shape):
            napari.utils.notifications.show_warning(
                "Clicked outside image dimensions. Please click inside the image to move."
            )
            return False, False

        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), 
            image=self.image.data, 
            pixelsize=self.image.metadata.pixel_size.x,
        )

        return coords, point

    def on_single_click(self, layer: NapariImageLayer, event: NapariEvent) -> None:
        """Callback for single click on the image layer. 
        Supports adding and updating positions with Shift and Alt modifiers.
        Args:
            layer: The image layer.
            event: The event object.
        """

        UPDATE_POSITION: bool = "Shift" in event.modifiers
        ADD_NEW_POSITION: bool = "Alt" in event.modifiers
        
        # left click + (alt or shift)
        if event.button != 1 or not (ADD_NEW_POSITION or UPDATE_POSITION):
            logging.warning(f"Invalid mouse click: {event.button} {event.modifiers}")
            return 
        
        coords, point = self.get_coordinate_in_microscope_coordinates(layer, event)

        if point is False: # clicked outside image
            return

        # get the stage position (xyzrt) based on the clicked point and projection
        stage_position = self.microscope.project_stable_move( 
                    dx=point.x, dy=point.y, 
                    beam_type=self.image.metadata.image_settings.beam_type, 
                    base_position=self.image.metadata.microscope_state.stage_position)            

        # TODO: handle case where multiple modifiers are pressed
        if UPDATE_POSITION:
            idx = self.comboBox_tile_position.currentIndex()
            if idx == -1:
                logging.debug("No position selected to update.")
                return

        self.ADDING_POSITION = True # flag prevent double drawing from circular update

        if UPDATE_POSITION:
            stage_position.name = self.positions[idx].name
            self.positions[idx] = stage_position
            self.stage_position_updated_signal.emit(stage_position)
        elif ADD_NEW_POSITION:
            stage_position.name = f"Position {len(self.positions)+1:02d}" 
            self.positions.append(stage_position)
            self.stage_position_added_signal.emit(stage_position)

        self.ADDING_POSITION = False

    def on_double_click(self, layer: NapariImageLayer, event: NapariEvent) -> None:
        """Callback for double click on the image layer.
        Moves the stage to the clicked position.
        Args:
            layer: The image layer.
            event: The event object.
        """

        if event.button != 1: # left click only
            return

        coords, point = self.get_coordinate_in_microscope_coordinates(layer, event)

        if point is False: # clicked outside image
            return
        
        beam_type = self.image.metadata.image_settings.beam_type
        stage_position = self.microscope.project_stable_move( 
            dx=point.x, dy=point.y, 
            beam_type=beam_type, 
            base_position=self.image.metadata.microscope_state.stage_position)   

        self.move_to_stage_position(stage_position)

    def update_current_selected_position(self):
        """Update the currently selected position."""
        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1 or len(self.positions) == 0:
            self.lineEdit_tile_position_name.setText("")
            return

        self.lineEdit_tile_position_name.setText(self.positions[idx].name)
        self.pushButton_move_to_position.setText(f"Move to {self.positions[idx].name}")

        # TODO: should this also update autolamella?
        # TODO: migrate to shared data model, list-view

    def update_positions_combobox(self):
        """Update the positions combobox with the current positions."""

        HAS_POSITIONS = len(self.positions) > 0
        self.pushButton_move_to_position.setEnabled(HAS_POSITIONS)
        self.pushButton_remove_position.setEnabled(HAS_POSITIONS)
        self.pushButton_update_position.setEnabled(HAS_POSITIONS)
        self.groupBox_positions.setVisible(HAS_POSITIONS)

        idx = self.comboBox_tile_position.currentIndex()
        self.comboBox_tile_position.clear()
        self.comboBox_tile_position.addItems([pos.name for pos in self.positions])
        if idx == -1:
            return 
        self.comboBox_tile_position.setCurrentIndex(idx)

        msg = ""
        for pos in self.positions: # TODO: migrate to list-view
            msg += f"{pos.name}:\t x:{pos.x*1e3:.2f}mm, y:{pos.y*1e3:.2f}mm, z:{pos.z*1e3:.2f}m\n"
        self.label_position_info.setText(msg)

    def remove_selected_position_pressed(self):
        """Remove the selected position from the list."""
        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            return
        stage_position = self.positions[idx]
        logging.info(f"Removing position...{stage_position.name}")
        self.positions.remove(stage_position)
        self.stage_position_removed_signal.emit(stage_position)

    def handle_stage_position_signals(self, stage_position: FibsemStagePosition):
        """Handle the stage position signals."""

        # TODO: if autolamella is the parent, this causes a double update. 
        # we should only update the parent, and the parent should update the minimap

        self.update_positions_combobox()
        self.update_viewer()

    def move_to_position_pressed(self) -> None:
        """Move the stage to the selected position."""
        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            return
        stage_position = self.positions[idx] # TODO: migrate to self.selected_stage_position
        self.move_to_stage_position(stage_position)

    def move_to_stage_position(self, stage_position: FibsemStagePosition)->None:
        """Move the stage to the selected position via movement widget."""
        self.movement_widget.move_to_position(stage_position)

    def handle_movement_progress(self, ddict: dict):
        """Handle the movement progress signal from the movement widget."""
        is_finished = ddict.get("finished", False)
        if is_finished:
            self.update_viewer()

    def redraw_pattern_overlay(self):
        """Redraw the selected milling patterns on the image for each saved position."""

        selected_pattern = self.comboBox_pattern_overlay.currentText()
        self.selected_milling_stage = self.protocol.milling[selected_pattern][0]
        self.draw_stage_positions()

    def draw_current_stage_position(self):
        """Draws the current stage position on the image."""
        current_stage_position = deepcopy(self.microscope.get_stage_position())
        points = tiled.reproject_stage_positions_onto_image(self.image, [current_stage_position])
        points[0].name = "Current Position"
        
        draw_positions_in_napari(viewer=self.viewer, 
                                 points=points, 
                                 show_names=True, 
                                 layer_name="current-stage-position")

    def draw_stage_positions(self):
        """Draws the saved positions on the image. Optionally, draws milling patterns."""

        # Performance Note: the reason this is slow is because every time a new position is added, we re-draw every position
        # this is not necessary, we can just add the new position to the existing layer
        # almost all the slow down comes from the linked callbacks from autolamella. probably saving and re-drawing milling patterns
        # we should delay that until the user requests it

        if self.image and self.positions:
            logging.info("Drawing Reprojected Positions...")

            points = tiled.reproject_stage_positions_onto_image(self.image, self.positions)

            position_layer = draw_positions_in_napari(viewer=self.viewer, 
                                points=points, 
                                size_px=75,
                                show_names=True, 
                                layer_name="saved-stage-positions")

            show_patterns: bool = self.checkBox_pattern_overlay.isChecked()
            if show_patterns: 
                milling_stages = []
                stage: FibsemMillingStage
                for point in points:
                    pt = conversions.image_to_microscope_image_coordinates(coord=point, 
                                                                      image=self.image.data, 
                                                                    pixelsize=self.image.metadata.pixel_size.x)
                    stage = deepcopy(self.selected_milling_stage)
                    stage.name = point.name
                    stage.pattern.point = pt
                    milling_stages.append(stage)

                self.milling_pattern_layers = draw_milling_patterns_in_napari(
                    viewer=self.viewer, 
                    image_layer=self.image_layer,
                    pixelsize=self.image.metadata.pixel_size.x,
                    milling_stages = milling_stages,
                    draw_crosshair=False,
                    )
            else:
                # TODO: change this to only remove milling pattern layer now that they are all in a single layer
                remove_all_napari_shapes_layers(viewer=self.viewer)

        # hide the layer if no positions
        if "saved-stage-positions" in self.viewer.layers:
            self.position_layer = self.viewer.layers["saved-stage-positions"]
            self.position_layer.visible = bool(self.positions)

        self.set_active_layer_for_movement()

    def add_correlation_image(self, image: FibsemImage, is_gridbar: bool = False):
        """Add a correlation image to the viewer."""

        basename = CORRELATION_IMAGE_LAYER_PROPERTIES["name"]
        if is_gridbar:
            basename = GRIDBAR_IMAGE_LAYER_PROPERTIES["name"]
        idx = 1
        layer_name = f"{basename}-{idx:02d}"
        while layer_name in self.viewer.layers:
            idx+=1
            layer_name = f"{basename}-{idx:02d}"

        # if grid bar in _name, idx = 3
        if is_gridbar:
            layer_name = basename
            idx = 3

        # add the image layer
        self.viewer.add_image(image.data, 
                        name=layer_name, 
                        colormap=COLOURS[idx%len(COLOURS)], 
                        blending=CORRELATION_IMAGE_LAYER_PROPERTIES["blending"], 
                        opacity=CORRELATION_IMAGE_LAYER_PROPERTIES["opacity"])
        self.correlation_image_layers.append(layer_name)

        # update the combobox
        self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
        idx = self.comboBox_correlation_selected_layer.currentIndex()
        self.comboBox_correlation_selected_layer.clear()
        self.comboBox_correlation_selected_layer.addItems(self.correlation_image_layers)
        if idx != -1:
            self.comboBox_correlation_selected_layer.setCurrentIndex(idx)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        
        # set the image layer as the active layer
        self.set_active_layer_for_movement()
        self.groupBox_correlation.setEnabled(True) # TODO: allow enabling grid-bar overlay separately
        self.checkBox_gridbar.setEnabled(True)
        self.pushButton_enable_correlation.setEnabled(True)

    # do this when image selected is changed
    def update_correlation_ui(self):

        # set ui
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        self.pushButton_enable_correlation.setEnabled(layer_name != "")
        if layer_name == "":
            napari.utils.notifications.show_info("Please select a layer to correlate with update data...")
            return

    def _toggle_correlation_mode(self, event: NapariEvent = None):
        """Toggle correlation mode on or off."""
        if self.image is None:
            napari.utils.notifications.show_warning("Please acquire an image first...")
            return
        
        if not self.correlation_image_layers:
            napari.utils.notifications.show_warning("Please load a correlation image first...")
            return

        # toggle correlation mode
        self.correlation_mode_enabled = not self.correlation_mode_enabled

        if self.correlation_mode_enabled:
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_enable_correlation.setText("Disable Correlation Mode")
            self.comboBox_correlation_selected_layer.setEnabled(False)
        else:
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            self.pushButton_enable_correlation.setText("Enable Correlation Mode")
            self.comboBox_correlation_selected_layer.setEnabled(True)

        # if no correlation layer selected, disable the button
        if self.comboBox_correlation_selected_layer.currentIndex() == -1:
            self.pushButton_enable_correlation.setEnabled(False)
            return

        # get current correlation layer
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        correlation_layer = self.viewer.layers[layer_name]
        
        # set transformation mode on
        if self.correlation_mode_enabled:
            correlation_layer.mode = 'transform'
            self.viewer.layers.selection.active = correlation_layer
        else:
            correlation_layer.mode = 'pan_zoom'
            self.set_active_layer_for_movement()

    def set_active_layer_for_movement(self) -> None:
        """Set the active layer to the image layer for movement."""
        if self.image_layer is not None and self.image_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.image_layer

# TODO: update layer name, set from file?
# TODO: set combobox to all images in viewer 
