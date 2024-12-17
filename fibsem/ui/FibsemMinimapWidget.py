import logging
import os
from copy import deepcopy
from typing import List, Tuple, Dict, Any

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Points as NapariPointsLayer
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from scipy.ndimage import median_filter

from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.imaging import _tile
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import get_milling_stages
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.ui import stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    remove_all_napari_shapes_layers,
)
from fibsem.ui.napari.utilities import draw_crosshair_in_napari
from fibsem.ui.qtdesigner_files import FibsemMinimapWidget as FibsemMinimapWidgetUI
from fibsem.ui import FibsemMovementWidget, FibsemImageSettingsWidget

PATH = os.path.join(cfg.DATA_PATH, "tile")
os.makedirs(PATH, exist_ok=True)

OVERVIEW_IMAGE_LAYER_PROPERTIES = {
    "name": "overview-image",
    "colormap": "gray",
    "blending": "additive",
    "median_filter_size": 3,
}

GRIDBAR_IMAGE_LAYER_PROPERTIES = {
    "name": "gridbar-image",
    "spacing": 100,
    "width": 20,
}

CORRELATION_IMAGE_LAYER_PROPERTIES = {
    "name": "correlation-image",
    "colormap": "green",
    "blending": "translucent",
    "opacity": 0.2,
}

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


class FibsemMinimapWidget(FibsemMinimapWidgetUI.Ui_MainWindow, QtWidgets.QMainWindow):
    _stage_position_moved = pyqtSignal(FibsemStagePosition)
    _stage_position_added = pyqtSignal(FibsemStagePosition)
    tile_acquisition_progress_signal = pyqtSignal(dict)
    _minimap_positions = pyqtSignal(list)
    
    def __init__(
        self,
        viewer: napari.Viewer,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent

        self.microscope: FibsemMicroscope = self.parent.microscope
        self.protocol: Dict[str, Any] = deepcopy(self.parent.settings.protocol)
        self.movement_widget: FibsemMovementWidget = self.parent.movement_widget
        self.image_widget: FibsemImageSettingsWidget = self.parent.image_widget

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.image: FibsemImage = None
        self.image_layer: NapariImageLayer  = None
        self.position_layer: NapariPointsLayer = None
        self.correlation_image_layers: List[NapariImageLayer] = []
        self.milling_pattern_layers: List[str] = []

        self.positions = []
        self.correlation_data = {}
        self.correlation_mode_enabled: bool = False

        self.ADDING_POSITION: bool = False
        self.STOP_ACQUISITION: bool = False

        self.setup_connections()
        self.update_ui()

    def setup_connections(self):
        
        # acquisition buttons
        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)
        self.actionLoad_Image.triggered.connect(self.load_image)
        
        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])
        self.lineEdit_tile_path.setText(str(self.image_widget.image_settings.path)) # set default path

        # position buttons
        self.pushButton_move_to_position.clicked.connect(self.move_to_position_pressed)
        self.comboBox_tile_position.currentIndexChanged.connect(self._update_current_position_info)
        self.pushButton_update_position.clicked.connect(self._update_position_pressed)
        self.pushButton_remove_position.clicked.connect(self._remove_position_pressed)

        # signals
        self.tile_acquisition_progress_signal.connect(self.handle_tile_acquisition_progress)

        # update the positions from the parent   
        if hasattr(self.parent, "_minimap_signal"):
            self.parent._minimap_signal.connect(self.update_positions_from_parent)

        # update the parent when the stage position is moved
        if  hasattr(self.movement_widget, "_stage_position_moved"):
            self._stage_position_moved.connect(self.movement_widget._stage_position_moved)

        # pattern overlay
        AVAILABLE_MILLING_PATTERNS = [k for k in self.protocol.get("milling", {}).keys()]
        self.comboBox_pattern_overlay.addItems(AVAILABLE_MILLING_PATTERNS)
        if "trench" in AVAILABLE_MILLING_PATTERNS:
            self.comboBox_pattern_overlay.setCurrentText("trench")
        elif "mill_rough" in AVAILABLE_MILLING_PATTERNS:
            self.comboBox_pattern_overlay.setCurrentText("mill_rough")
        self.comboBox_pattern_overlay.currentIndexChanged.connect(self.redraw_pattern_overlay)
        self.checkBox_pattern_overlay.stateChanged.connect(self.redraw_pattern_overlay)

        # correlation
        self.actionLoad_Correlation_Image.triggered.connect(self.load_image)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        self.pushButton_enable_correlation.clicked.connect(self._toggle_correlation_mode)
        self.pushButton_enable_correlation.setEnabled(False) # disabled until correlation images added

        # auto update correlation image
        self.doubleSpinBox_correlation_translation_x.valueChanged.connect(self.update_correlation_data)
        self.doubleSpinBox_correlation_translation_y.valueChanged.connect(self.update_correlation_data)
        self.doubleSpinBox_correlation_scale_x.valueChanged.connect(self.update_correlation_data) 
        self.doubleSpinBox_correlation_scale_y.valueChanged.connect(self.update_correlation_data)
        self.doubleSpinBox_correlation_rotation.valueChanged.connect(self.update_correlation_data)

        self.doubleSpinBox_correlation_translation_x.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_translation_y.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_scale_x.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_scale_y.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_rotation.setKeyboardTracking(False)

        # gridbar controls
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

        # set styles
        self.pushButton_update_position.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_run_tile_collection.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_cancel_acquisition.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.progressBar_acquisition.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.pushButton_remove_position.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_enable_correlation.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

        self.toggle_interaction(enable=True)

    def update_positions_from_parent(self, positions):
        
        if positions is not None:
            self.positions = positions

        if not self.ADDING_POSITION:
            self.update_position_info()
            self.update_viewer()

    def run_tile_collection(self):

        logging.info("running tile collection")

        beam_type = BeamType[self.comboBox_tile_beam_type.currentText()]
        grid_size = self.doubleSpinBox_tile_grid_size.value() * constants.MICRO_TO_SI
        tile_size = self.doubleSpinBox_tile_tile_size.value() * constants.MICRO_TO_SI
        resolution = int(self.spinBox_tile_resolution.value())
        dwell_time = self.doubleSpinBox_tile_dwell_time.value() * constants.MICRO_TO_SI
        cryo = self.checkBox_tile_autogamma.isChecked()
        autocontrast = self.checkBox_tile_autogamma.isChecked()
        path = self.lineEdit_tile_path.text()
        filename = self.lineEdit_tile_filename.text()

        image_settings = ImageSettings(        
            hfw = tile_size,
            resolution = [resolution, resolution],
            dwell_time = dwell_time,
            beam_type = beam_type,
            autocontrast = autocontrast,
            save = True,
            path = path,
            filename = filename,
        )

        if image_settings.filename == "":
            napari.utils.notifications.show_error("Please enter a filename for the image")
            return
        
        # ui feedback
        self.toggle_interaction(enable=False)

        self.STOP_ACQUISITION = False
        worker = self.run_tile_collection_thread(
            microscope=self.microscope, image_settings=image_settings, 
            grid_size=grid_size, tile_size=tile_size, 
            overlap=0, cryo=cryo)

        worker.finished.connect(self.tile_collection_finished)
        worker.errored.connect(self.tile_collection_errored)
        worker.start()

    def cancel_acquisition(self):

        logging.info("Cancelling acquisition...")
        self.STOP_ACQUISITION: bool = True

    def toggle_gridbar_display(self):

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
                self.viewer.layers.remove(layer_name)
                self.correlation_image_layers.remove(layer_name)

            self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
            self.comboBox_correlation_selected_layer.clear()
            self.comboBox_correlation_selected_layer.addItems([layer.name for layer in self.viewer.layers if "correlation-image" in layer.name ])
            self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
            # if no correlation layers left, disable enable correlation
            if len(self.comboBox_correlation_selected_layer) == 0:
                self.pushButton_enable_correlation.setEnabled(False)

    def update_gridbar_layer(self):
        
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
            self.update_correlation_image(gridbars_image, is_gridbar=True)

    def toggle_interaction(self, enable: bool = True):
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


    def tile_collection_finished(self):
        napari.utils.notifications.show_info("Tile collection finished.")
        self.update_viewer(self.image)
        self.toggle_interaction(enable=True)
        self.STOP_ACQUISITION = False

    def tile_collection_errored(self):
        logging.error("Tile collection errored.")
        self.STOP_ACQUISITION = False

    @thread_worker
    def run_tile_collection_thread(self, microscope: FibsemMicroscope, image_settings: ImageSettings,
        grid_size: float, tile_size:float, overlap:float=0, cryo: bool=True):

        try:
            self.image = _tile.tiled_image_acquisition_and_stitch(
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
        
        msg = f"{ddict['msg']} ({ddict['counter']}/{ddict['total']})"
        logging.info(msg)
        napari.utils.notifications.show_info(msg)

        # progress bar
        count, total = ddict["counter"], ddict["total"]
        self.progressBar_acquisition.setMaximum(100)
        self.progressBar_acquisition.setValue(int(count/total*100))

        image = ddict.get("image", None)
        if image is not None:
            self.update_viewer(FibsemImage(data=image), tmp=True)
        
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
            self.update_correlation_image(image)
        else:
            self.update_viewer(image)

    def update_ui(self):

        if self.image is not None:
            self.label_instructions.setText("Alt+Click to Add a position, Shift+Click to Update a position \nor Double Click to Move the Stage...")
        else:
            self.label_instructions.setText("Please take or load an overview image...")

    def update_viewer(self, image: FibsemImage = None, tmp: bool = False):

        if image is not None:
    
            if not tmp:
                self.image = image
            
            arr = median_filter(image.data, size=OVERVIEW_IMAGE_LAYER_PROPERTIES["median_filter_size"])

            try:
                self.image_layer.data = arr
            except Exception as e:              
                self.image_layer = self.viewer.add_image(arr, 
                                                         name=OVERVIEW_IMAGE_LAYER_PROPERTIES["name"],
                                                         colormap=OVERVIEW_IMAGE_LAYER_PROPERTIES["colormap"], 
                                                         blending=OVERVIEW_IMAGE_LAYER_PROPERTIES["blending"])

            if tmp:
                return # don't update the rest of the UI, we are just updating the image

            # draw a point on the image at center
            draw_crosshair_in_napari(viewer=self.viewer,
                                    sem_shape=self.image.data.shape, 
                                    fib_shape=None, 
                                    is_checked=True) 

            self.image_layer.mouse_drag_callbacks.clear()
            self.image_layer.mouse_double_click_callbacks.clear()
            self.image_layer.mouse_drag_callbacks.append(self._on_click)
            self.image_layer.mouse_double_click_callbacks.append(self._on_double_click)

            # NOTE: how to do respace scaling, convert to infinite canvas
            # px = self.image.metadata.pixel_size.x
            # self.image_layer.scale = [px*constants.SI_TO_MICRO, px*constants.SI_TO_MICRO]
            # self.viewer.scale_bar.visible = True
            # self.viewer.scale_bar.unit = "um"

        self._draw_positions() # draw the reprojected positions on the image

        self.viewer.layers.selection.active = self.image_layer

        self.update_ui()

    def get_data_from_coord(self, coords: tuple) -> tuple:
        # check inside image dimensions, (y, x)
        shape = self.image.data.shape[0], self.image.data.shape[1]

        if (coords[0] > 0 and coords[0] < shape[0]) and (
            coords[1] > 0 and coords[1] < shape[1]
        ):
            return True
        else:
            return False


    def _validate_mouse_click(self, layer, event):
        """validate if click is inside image, and convert to microscope coords"""                
        # get coords
        coords = layer.world_to_data(event.position)

        _inside_image = self.get_data_from_coord(coords)

        if _inside_image is False:
            napari.utils.notifications.show_warning(
                "Clicked outside image dimensions. Please click inside the image to move."
            )
            return False, False

        point = conversions.image_to_microscope_image_coordinates(
            Point(x=coords[1], y=coords[0]), self.image.data, self.image.metadata.pixel_size.x,
        )

        return coords, point

    def _on_click(self, layer, event):
        
        # left click + (alt or shift)
        if event.button != 1 or ('Alt' not in event.modifiers and 'Shift' not in event.modifiers):
            return 

        coords, point = self._validate_mouse_click(layer, event)

        if point is False: # clicked outside image
            return

        _new_position = self.microscope.project_stable_move( 
                    dx=point.x, dy=point.y, 
                    beam_type=self.image.metadata.image_settings.beam_type, 
                    base_position=self.image.metadata.microscope_state.stage_position)            
       
        self.ADDING_POSITION = True # flag prevent double drawing from circular update
        if 'Shift' in event.modifiers:
            idx = self.comboBox_tile_position.currentIndex()
            _name = self.positions[idx].name
            _new_position.name = _name
            self.positions[idx] = _new_position
        elif 'Alt' in event.modifiers:
            _new_position.name = f"Position {len(self.positions)+1:02d}" 
            self.positions.append(_new_position)
            self._stage_position_added.emit(_new_position)


        # we could save this position as well, use it to pre-select a bunch of lamella positions?
        self.update_position_info()
        self.update_viewer()

        self._minimap_positions.emit(self.positions)
        self.ADDING_POSITION = False

    def _on_double_click(self, layer, event):
        
        if event.button !=1:
            return
        coords, point = self._validate_mouse_click(layer, event)

        if point is False: # clicked outside image
            return
        
        beam_type = self.image.metadata.image_settings.beam_type
        _new_position = self.microscope.project_stable_move( 
            dx=point.x, dy=point.y, 
            beam_type=beam_type, 
            base_position=self.image.metadata.microscope_state.stage_position)   

        self._move_to_position(_new_position)

    def _update_current_position_info(self):

        idx = self.comboBox_tile_position.currentIndex()
        if idx != -1 and len(self.positions) > 0:
            self.lineEdit_tile_position_name.setText(self.positions[idx].name)
            self.pushButton_move_to_position.setText(f"Move to {self.positions[idx].name}")
        else:
            self.lineEdit_tile_position_name.setText("")
        

    def update_position_info(self):
        idx = self.comboBox_tile_position.currentIndex()
        self.comboBox_tile_position.clear()
        self.comboBox_tile_position.addItems([pos.name for pos in self.positions])
        if idx != -1:
            self.comboBox_tile_position.setCurrentIndex(idx)

        _positions_added = len(self.positions) > 0
        self.pushButton_move_to_position.setEnabled(_positions_added)
        self.pushButton_remove_position.setEnabled(_positions_added)
        self.pushButton_update_position.setEnabled(_positions_added)

        msg = ""
        for pos in self.positions:
            msg += f"{pos.name}:\t x:{pos.x*1e3:.2f}mm, y:{pos.y*1e3:.2f}mm, z:{pos.z*1e3:.2f}m\n"
        self.label_position_info.setText(msg)

    def _update_position_pressed(self):

        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            logging.debug(f"No position selected to update.")
            return
        
        name = self.lineEdit_tile_position_name.text()
        if name == "":
            napari.utils.notifications.show_info(f"Please enter a name for the position")
            return

        logging.info(f"Updating position at {idx} to {name}")
        stage_position = self.positions[idx]
        stage_position.name = name 
        self.update_position_info()
        self.update_viewer()

    def _remove_position_pressed(self):
        idx = self.comboBox_tile_position.currentIndex()
        if idx == -1:
            logging.debug("No position selected to remove.")
            return
        logging.info("Removing position...")
        stage__position = self.positions[idx]
        self.positions.remove(stage__position)
        self._minimap_positions.emit(self.positions)
        self.update_position_info()
        self.update_viewer()

    def move_to_position_pressed(self):

        stage_position = self.positions[self.comboBox_tile_position.currentIndex()]
        logging.info(f"Moving To: {stage_position}")
        self._move_to_position(stage_position)

    def _move_to_position(self, _position:FibsemStagePosition)->None:
        self.microscope.safe_absolute_stage_movement(_position)
        self._stage_position_moved.emit(_position)
        self.update_viewer()

    def redraw_pattern_overlay(self):
        """Redraw the milling patterns on the image."""

        selected_pattern = self.comboBox_pattern_overlay.currentText()
        self.selected_milling_stage = get_milling_stages(selected_pattern, 
                                                         protocol=self.protocol["milling"])[0]
        self._draw_positions()

    def _draw_positions(self):
        
        logging.info("Drawing Reprojected Positions...")
        current_position = deepcopy(self.microscope.get_stage_position())
        current_position.name = "Current Position"

        # Performance Note: the reason this is slow is because every time a new position is added, we re-draw every position
        # this is not necessary, we can just add the new position to the existing layer
        # almost all the slow down comes from the linked callbacks from autolamella. probably saving and re-drawing milling patterns
        # we should delay that until the user requests it

        if self.image:
            
            drawn_positions: List[FibsemStagePosition] = self.positions + [current_position]
            points = _tile._reproject_positions(self.image, drawn_positions)

            # convert to napari format (y, x)
            data = [[pt.y, pt.x] for pt in points]
            
            colors = ["lime"] * (len(drawn_positions)-1)
            colors.append("yellow")

            colors_rgba = [[0,1,0,1] for _ in range(len(drawn_positions)-1)]
            colors_rgba.append([1,1,0,1]) # yellow

            text = {
                "string": [pos.name for pos in drawn_positions],
                "color": colors,
                "translation": np.array([-50, 0]),
            }

            OVERVIEW_POSITIONS_LAYER_PROPERTIES = {
                "name": "overview-positions",
                "size": 60,
                "edge_width": 7,
                "edge_width_is_relative": False,
                "edge_color": "transparent",
                "face_color": "lime",
                "blending": "translucent",
                "symbol": "cross",
            }

            if self.position_layer is None:
            
                 self.position_layer = self.viewer.add_points(
                    data=data,
                    name=OVERVIEW_POSITIONS_LAYER_PROPERTIES["name"],
                    text=text,
                    size=OVERVIEW_POSITIONS_LAYER_PROPERTIES["size"],
                    edge_width=OVERVIEW_POSITIONS_LAYER_PROPERTIES["edge_width"],
                    edge_width_is_relative=OVERVIEW_POSITIONS_LAYER_PROPERTIES["edge_width_is_relative"],
                    edge_color=OVERVIEW_POSITIONS_LAYER_PROPERTIES["edge_color"],
                    face_color=OVERVIEW_POSITIONS_LAYER_PROPERTIES["face_color"],
                    blending=OVERVIEW_POSITIONS_LAYER_PROPERTIES["blending"],
                    symbol=OVERVIEW_POSITIONS_LAYER_PROPERTIES["symbol"],
                )
            else:
                self.position_layer.data = data
                self.position_layer.text = text

            self.position_layer.face_color= colors_rgba

            SHOW_PATTERNS: bool = self.checkBox_pattern_overlay.isChecked()
            if SHOW_PATTERNS: # TODO: this is very slow, need to speed up, too many pattern redraws
                points = [conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), 
                                                                            self.image.data, 
                                                                            self.image.metadata.pixel_size.x ) for coords in data[:-1]]
                
                milling_stages = []
                for point, pos in zip(points, drawn_positions[:-1]):
                    stage = deepcopy(self.selected_milling_stage)
                    stage.name = pos.name
                    stage.pattern.point = point
                    milling_stages.append(stage)

                self.milling_pattern_layers = draw_milling_patterns_in_napari(
                    viewer=self.viewer, 
                    image_layer=self.image_layer,
                    pixelsize=self.image.metadata.pixel_size.x,
                    milling_stages = milling_stages,
                    draw_crosshair=False,
                    )
            else:
                remove_all_napari_shapes_layers(viewer=self.viewer)

        self.viewer.layers.selection.active = self.image_layer

    def update_correlation_image(self, image: FibsemImage = None, is_gridbar: bool = False):
            
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

        COLOURS = ["green", "cyan", "magenta", "red", "yellow"]
        correlation_layer = self.viewer.add_image(image.data, 
                        name=layer_name, 
                        colormap=COLOURS[idx%len(COLOURS)], 
                        blending="translucent", opacity=0.2)
        
        self.correlation_image_layers.append(correlation_layer)
        self.correlation_data[layer_name] = deepcopy([0, 0, 1.0, 1.0, 0])

        # update the combobox
        self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
        idx = self.comboBox_correlation_selected_layer.currentIndex()
        self.comboBox_correlation_selected_layer.clear()
        correlation_layer_names = [layer.name for layer in self.correlation_image_layers]
        self.comboBox_correlation_selected_layer.addItems(correlation_layer_names)
        if idx != -1:
            self.comboBox_correlation_selected_layer.setCurrentIndex(idx)
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self.update_correlation_ui)
        
        # set the image layer as the active layer
        self.viewer.layers.selection.active = self.image_layer
        self.pushButton_enable_correlation.setEnabled(True)

    # do this when image selected is changed
    def update_correlation_ui(self):

        # set ui
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        self.pushButton_enable_correlation.setEnabled(layer_name != "")
        if layer_name == "":
            napari.utils.notifications.show_info("Please select a layer to correlate with update data...")
            return

        tx, ty, sx, sy, r = self.correlation_data[layer_name]

        self.doubleSpinBox_correlation_translation_x.setValue(tx)
        self.doubleSpinBox_correlation_translation_y.setValue(ty)
        self.doubleSpinBox_correlation_scale_x.setValue(sx)
        self.doubleSpinBox_correlation_scale_y.setValue(sy)
        self.doubleSpinBox_correlation_rotation.setValue(r)

        self.update_correlation_data()

    # do this when parameters change
    def update_correlation_data(self):

        # select layer
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        if layer_name == "":
            napari.utils.notifications.show_info("Please select a layer to correlate with update ui...")
            return

        tx, ty = self.doubleSpinBox_correlation_translation_x.value(), self.doubleSpinBox_correlation_translation_y.value()
        sx, sy = self.doubleSpinBox_correlation_scale_x.value(), self.doubleSpinBox_correlation_scale_y.value()
        r = self.doubleSpinBox_correlation_rotation.value()

        if sx == 0 or sy == 0:
            sx =1
            sy =1

        angle = np.deg2rad(r)

        rows = self.viewer.layers[layer_name].data.shape[0]*0.5 
        cols = self.viewer.layers[layer_name].data.shape[1]*0.5

        # the proof for this is marvelous but i dont have enough space in the comments

        new_x = int(np.cos(angle) * rows - np.sin(angle) * cols) 
        new_y =  int(np.sin(angle) * rows + np.cos(angle) * cols) 

        trans_x = new_x -rows 
        trans_y = new_y -cols 

        # translation for central rotation + translation
        corrected_tx = tx - trans_y
        corrected_ty = -ty - trans_x


        self.correlation_data[layer_name] = deepcopy([tx, ty, sx, sy, r])

        # apply to selected layer
        self.viewer.layers[layer_name].translate = [corrected_ty, corrected_tx]
        self.viewer.layers[layer_name].scale = [sy, sx]
        self.viewer.layers[layer_name].rotate = r

    def _toggle_correlation_mode(self):
        
        # toggle correlation mode
        self.correlation_mode_enabled = not self.correlation_mode_enabled

        if self.correlation_mode_enabled:
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_enable_correlation.setText("Disable Correlation Mode")
            self.comboBox_correlation_selected_layer.setEnabled(False)
        else:
            self.pushButton_enable_correlation.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
            self.pushButton_enable_correlation.setText("Enable Correlation Mode")
            self.comboBox_correlation_selected_layer.setEnabled(True)

        # if no correlation layer selected, disable the button
        if self.comboBox_correlation_selected_layer.currentText() == "":
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
            self.viewer.layers.selection.active = self.image_layer


# TODO: update layer name, set from file?
# TODO: set combobox to all images in viewer 



def main():

    viewer = napari.Viewer(ndisplay=2)
    microscope, settings = utils.setup_session()
    minimap_widget = FibsemMinimapWidget(microscope, settings, viewer=viewer)
    viewer.window.add_dock_widget(
        minimap_widget, area="right", add_vertical_stretch=False, name="OpenFIBSEM Minimap"
    )
    napari.run()


if __name__ == "__main__":
    main()