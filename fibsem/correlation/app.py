import datetime
import logging
import os
from typing import List, Tuple

import napari
import napari.utils
import numpy as np
import pandas as pd
from napari.layers import Image as NapariImageLayer
from napari.utils import notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from fibsem.correlation import (
    load_and_parse_fib_image,
    multi_channel_get_z_guass,
    multi_channel_zyx_targeting,
    parse_coordinates,
    run_correlation,
    save_correlation_data,
)
from fibsem.correlation.ui import (
    COORDINATE_LAYER_PROPERTIES,
    CORRELATION_PROPERTIES,
    DATAFRAME_PROPERTIES,
    DRAG_DROP_INSTRUCTIONS,
    FILE_FILTERS,
    INSTRUCTIONS,
    LINE_LAYER_PROPERTIES,
    REPROJECTION_LAYER_PROPERTIES,
    RESULTS_LAYER_PROPERTIES,
    TEXT_PROPERTIES,
    USER_PREFERENCES,
    PandasTableModel,
    open_import_wizard,
    tdct_main,
)

logging.basicConfig(level=logging.INFO)

def set_table_properties(table: QtWidgets.QTableView):
    table.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )
    # get number of columns
    for i in range(table.columnCount()):
        table.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)

    # set min height to 200
    table.setMinimumHeight(200)

def check_coordinates_inside_layer(event, target_layers: list):
    for target_layer in target_layers:

        def check_coords(event_position, target_layer):
            coords = target_layer.world_to_data(event_position)

            extent_min = target_layer.extent.data[0]  # (z, y, x)
            extent_max = target_layer.extent.data[1]

            # if they are 4d, remove the first dimension
            if len(coords) == 4:
                logging.warning(f"4D coordinates detected: {coords}, removing first dimension")
                coords = coords[1:]
                extent_min = extent_min[1:]
                extent_max = extent_max[1:]

            # convert the above logs into a json msg
            msgd = {
                "target_layer": target_layer.name,
                "event_position": event_position,
                "coords": coords,
                "extent_min": extent_min,
                "extent_max": extent_max,
            }
            logging.info(msgd)

            for i, coord in enumerate(coords):
                if coord < extent_min[i] or coord > extent_max[i]:
                    logging.debug(
                        f"Coordinate {coord} is out of bounds ({extent_min[i]}, {extent_max[i]})"
                    )
                    return False

            return True

        if check_coords(event.position, target_layer):
            logging.debug(
                f"Coordinates are within bounds of {target_layer.name}"
            )
            return target_layer

    return None


DEV_MODE = False
DEV_PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset"
DEV_FIB_IMAGE = "fib_002.tif"
DEV_FM_IMAGE = "test-image2.ome.tiff"

# TODO:
# support 2D multi-point correlation: works when image is 2D (z=1), but should also support restricting
# display overlay after correlation
# add deconvolution 

class CorrelationUI(tdct_main.Ui_MainWindow, QtWidgets.QMainWindow):
    close_signal = pyqtSignal()
    continue_pressed_signal = pyqtSignal(dict)

    def __init__(self, viewer: napari.Viewer, parent_ui: QtWidgets.QWidget = None):
        super().__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.parent_ui = parent_ui

        self.path = None
        self.project_loaded: bool = False
        self.images_loaded: bool = False
        self.use_z_gauss_optim: bool = False
        self.use_mip: bool = False

        self.fib_image: np.ndarray = None
        self.fm_image: np.ndarray = None

        self.translation = None
        self.fib_image_layer = None
        self.fm_image_layers: List[NapariImageLayer] = []
        self.selected_index = []

        self.rotation_center: tuple = None

        self.df = pd.DataFrame([], columns=DATAFRAME_PROPERTIES["columns"])
        self.correlation_results: dict = None
        self.poi_coordinate: Tuple[float, float] = (0, 0) # TODO: migrate to Point

        # add a point layer for the coordinates (FIB, FM, POI)
        self.coordinates_layer = None
        self.line_layer = None              # corresponding points
        self.results_layer = None           # points of interest results
        self.reprojection_layer = None      # correlation error data
        self.poi_coordinate_layer = None    # points of interest for drag-drop

        self.setup_connections()
        self._show_project_controls()

        if DEV_MODE:
            self.set_project_path(DEV_PATH)
            self._set_fib_image_path(os.path.join(DEV_PATH, DEV_FIB_IMAGE))
            self._set_fm_image_path(os.path.join(DEV_PATH, DEV_FM_IMAGE))

    def closeEvent(self, event):
        self.close_signal.emit()
        super().closeEvent(event)

    def _show_results_widgets(self, enable: bool):
        self.groupBox_results.setVisible(enable)

    def setup_connections(self):
        """Configure the connections for the UI elements"""

        # image / coordinate controls
        self.checkBox_use_mip.clicked.connect(self.toggle_mip)
        self.checkBox_show_corresponding_points.clicked.connect(
            self._show_corresponding_points
        )
        self.checkBox_show_points_thick_dims.clicked.connect(self._toggle_thick_dims)

        # set the default settings
        self.checkBox_use_mip.setChecked(USER_PREFERENCES["use_mip"])
        self.checkBox_show_corresponding_points.setChecked(
            USER_PREFERENCES["show_corresponding_points"]
        )
        self.checkBox_use_zgauss_opt.setChecked(USER_PREFERENCES["use_z_gauss_optim"])
        self.checkBox_show_points_thick_dims.setChecked(
            USER_PREFERENCES["show_thick_dims"]
        )

        # set the default properties for the coordinates layer
        self.comboBox_options_point_symbol.addItems(["ring", "disc", "x"])
        self.comboBox_options_point_symbol.setCurrentText(
            COORDINATE_LAYER_PROPERTIES["symbol"]
        )
        self.spinBox_options_point_size.setValue(COORDINATE_LAYER_PROPERTIES["size"])
        self.comboBox_options_point_symbol.currentIndexChanged.connect(
            self._update_user_preferences
        )
        self.spinBox_options_point_size.valueChanged.connect(
            self._update_user_preferences
        )

        # parameter controls
        self.doubleSpinBox_parameters_pixel_size.setDecimals(4)
        self.doubleSpinBox_parameters_pixel_size.setRange(0, 100)
        self.doubleSpinBox_parameters_pixel_size.valueChanged.connect(
            self._update_parameters
        )
        self.spinBox_parameters_rotation_center_x.valueChanged.connect(
            self._update_parameters
        )
        self.spinBox_parameters_rotation_center_y.valueChanged.connect(
            self._update_parameters
        )
        self.spinBox_parameters_rotation_center_z.valueChanged.connect(
            self._update_parameters
        )
        # hide these, don't allow user to change
        self.label_parameters_rotation_center.setVisible(False)
        self.spinBox_parameters_rotation_center_x.setVisible(False)
        self.spinBox_parameters_rotation_center_y.setVisible(False)
        self.spinBox_parameters_rotation_center_z.setVisible(False)
        self.doubleSpinBox_parameters_pixel_size.setSuffix(" um")

        # menu actions
        self.actionLoad_Load_Coordinates_Old.triggered.connect(self.load_coordinates)
        self.actionLoad_Load_Coordinates.triggered.connect(self.load_coordinates_v2)
        self.actionClear_Coordinates.triggered.connect(self.clear_coordinates)

        # project / image controls
        self.toolButton_project_path.clicked.connect(lambda: self.set_project_path(path=None))
        self.toolButton_fib_image_path.clicked.connect(lambda: self._set_fib_image_path(filename=None))
        self.toolButton_fm_image_path.clicked.connect(lambda: self._set_fm_image_path(filename=None))

        # correlation controls
        self.pushButton_run_correlation.clicked.connect(self.run_correlation)
        self.pushButton_run_correlation.setStyleSheet("background-color: green")
        self.label_instructions.setText(INSTRUCTIONS)

        self.pushButton_continue.setVisible(self.parent_ui is not None)
        self.pushButton_continue.clicked.connect(self.continue_pressed)
        self.continue_pressed_signal.connect(self.handle_continue_signal)
        self.pushButton_continue.setStyleSheet("background-color: blue")
        # method change
        self.comboBox_method.addItems(["Multi-Point", "Drag & Drop"])
        self.comboBox_method.currentIndexChanged.connect(self.on_method_changed)
        self.on_method_changed()

        self.pushButton_toggle_correlation_mode.clicked.connect(self.toggle_correlation_mode)
        self.pushButton_reset_transform.clicked.connect(self.reset_transforms)

        # refractive correction
        # mark surface (alt click?)
        # update poi
        self.pushButton_refreactive_update_poi.clicked.connect(self.apply_refractive_index_correction)
        # TODO: disable this button if no correlation results are available, no surface point available

    def on_method_changed(self):

        # TODO: warn user data will be reset when method is changed

        self.method_name = self.comboBox_method.currentText()
        logging.info(f"Method changed to: {self.method_name}")

        self.is_multi_point = self.method_name == "Multi-Point"
        self.is_drag_drop = self.method_name == "Drag & Drop"

        # display relevant panels
        self.groupBox_controls.setVisible(self.is_drag_drop)
        self.groupBox_options.setVisible(self.is_drag_drop)
        self.groupBox_parameters.setVisible(self.is_drag_drop)
        self.groupBox_coordinates.setVisible(not self.is_drag_drop)
    
        # display relevant controls
        self.label_parameters_rotation_center.setVisible(self.is_multi_point)
        self.spinBox_parameters_rotation_center_x.setVisible(self.is_multi_point)
        self.spinBox_parameters_rotation_center_y.setVisible(self.is_multi_point)
        self.spinBox_parameters_rotation_center_z.setVisible(self.is_multi_point)

        if self.is_drag_drop and self.fm_image_layers:
            self.viewer.layers.link_layers(self.fm_image_layers)
        if self.is_multi_point and self.fm_image_layers:
            self.viewer.layers.unlink_layers(self.fm_image_layers)

        # remove callbacks
        try:
            self.fib_image_layer.mouse_drag_callbacks.remove(self.update_poi_coordinate)
            for fm_layer in self.fm_image_layers:
                fm_layer.mouse_drag_callbacks.remove(self.update_poi_coordinate)
        except Exception as e:
            pass

        if self.is_multi_point:
            self._show_project_controls()
            if self.coordinates_layer is None:
                self.coordinates_layer = self.viewer.add_points(
                [],
                name=COORDINATE_LAYER_PROPERTIES["name"],
                ndim=COORDINATE_LAYER_PROPERTIES["ndim"],
                size=COORDINATE_LAYER_PROPERTIES["size"],
                projection_mode=COORDINATE_LAYER_PROPERTIES["projection_mode"],
                symbol=COORDINATE_LAYER_PROPERTIES["symbol"],
                text=TEXT_PROPERTIES,
                properties=self.df,
            )
            # single click callbacks
            self.coordinates_layer.mouse_drag_callbacks.append(
                self.update_correlation_points
            )
            self.coordinates_layer.events.data.connect(self.coordinates_updated_from_ui)
            self.pushButton_run_correlation.setStyleSheet("background-color: green")
            self.label_instructions.setText(INSTRUCTIONS)

            if self.poi_coordinate_layer is not None and self.poi_coordinate_layer in self.viewer.layers:
                self.viewer.layers.remove(self.poi_coordinate_layer)
                self.poi_coordinate_layer = None

        if self.is_drag_drop:
            if self.coordinates_layer is not None:
                if self.coordinates_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.coordinates_layer)
                self.coordinates_layer = None
            self.clear_coordinates()

            self.label_instructions.setVisible(False) # TODO: add dynamic-instructions
            self.pushButton_run_correlation.setStyleSheet("background-color: gray")

            if self.poi_coordinate_layer is None: # TODO: should this just be the results_layer???
                self.poi_coordinate_layer = self.viewer.add_points(
                    [],
                    name="POI",
                    ndim=2,
                    size=20,
                    symbol="disc",
                    face_color="magenta",
                    blending="additive",
                    opacity=0.9,
                ) # TODO: add to config
            elif self.poi_coordinate_layer in self.viewer.layers:
                self.poi_coordinate_layer.data = []

            self.fib_image_layer.mouse_drag_callbacks.append(self.update_poi_coordinate)
            for fm_layer in self.fm_image_layers:
                fm_layer.mouse_drag_callbacks.append(self.update_poi_coordinate)
                fm_layer.events.mode.connect(self._on_layer_mode_changed)

            self.toggle_correlation_mode()

            self.label_instructions.setText(DRAG_DROP_INSTRUCTIONS)

    def _on_layer_mode_changed(self, event):
        """Update the button text and color based on the layer mode"""
        transform_mode = event.mode == "transform"
        if transform_mode:
            self.pushButton_toggle_correlation_mode.setStyleSheet("background-color: orange")
            self.pushButton_toggle_correlation_mode.setText("Correlation Mode Enabled")
        else:
            self.pushButton_toggle_correlation_mode.setStyleSheet("background-color: green")
            self.pushButton_toggle_correlation_mode.setText("Enable Correlation Mode")

    def toggle_correlation_mode(self):

        if not self.is_drag_drop:
            return
        
        fm_layer = self.fm_image_layers[0]

        enabled = bool(fm_layer.mode == "transform")

        # select first fm layer
        fm_layer.mode = "pan_zoom" if enabled else "transform"

        self.viewer.layers.selection.active = fm_layer

    def reset_transforms(self):

        # can't have mip enabled when we reset, so disable it
        if self.use_mip:
            self.checkBox_use_mip.setChecked(False)
            self.toggle_mip()

        for fm_layer in self.fm_image_layers:
            fm_layer: NapariImageLayer
            fm_layer._reset_affine()

    def continue_pressed(self) -> None:
        # continue with the correlation
        logging.info("Continue Pressed")

        info = {"poi": self.poi_coordinate}

        msg = "Finish correlation and continue?"
        ret = QtWidgets.QMessageBox.question(self,
                                             'Finish Correlation',
                                             msg,
                                             QtWidgets.QMessageBox.Yes, 
                                             QtWidgets.QMessageBox.No)

        if ret == QtWidgets.QMessageBox.Yes:
            self.remove_correlation_layers()
            self.continue_pressed_signal.emit(info)
            self.close()

    def remove_correlation_layers(self):
        """Remove all layers associated with correlation (for embedded workflow)"""
        if self.line_layer is not None:
            if self.line_layer in self.viewer.layers:
                self.viewer.layers.remove(self.line_layer)
                self.line_layer = None

        if self.results_layer is not None:
            if self.results_layer in self.viewer.layers:
                self.viewer.layers.remove(self.results_layer)
                self.results_layer = None

        if self.reprojection_layer is not None:
            if self.reprojection_layer in self.viewer.layers:
                self.viewer.layers.remove(self.reprojection_layer)
                self.reprojection_layer = None
        if self.poi_coordinate_layer is not None:
            if self.poi_coordinate_layer in self.viewer.layers:
                self.viewer.layers.remove(self.poi_coordinate_layer)
                self.poi_coordinate_layer = None

        if self.coordinates_layer is not None:
            if self.coordinates_layer in self.viewer.layers:
                self.viewer.layers.remove(self.coordinates_layer)
                self.coordinates_layer = None

        if self.fm_image_layers:
            self.viewer.layers.unlink_layers(self.fm_image_layers) 
        for layer in self.fm_image_layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.fm_image_layers = []

        if self.fib_image_layer is not None:
            if self.fib_image_layer in self.viewer.layers:
                self.viewer.layers.remove(self.fib_image_layer)
                self.fib_image_layer = None
        
        if "Initial PoI" in self.viewer.layers:
            self.viewer.layers.remove("Initial PoI")

    def handle_continue_signal(self, ddict: dict):

        logging.info("CONTINUE SIGNAL PRESSED")
        logging.info(f"POI: {ddict['poi']}")

    def _show_project_controls(self):
        self.images_loaded = self.fib_image is not None and self.fm_image is not None

        # enable the controls for loading the images
        self.toolButton_fib_image_path.setEnabled(self.project_loaded)
        self.toolButton_fm_image_path.setEnabled(self.project_loaded)
        self.lineEdit_fib_image_path.setEnabled(self.project_loaded)
        self.lineEdit_fm_image_path.setEnabled(self.project_loaded)

        # toggle the image controls based on whether the project is loaded
        self.checkBox_use_mip.setEnabled(self.images_loaded)
        self.checkBox_show_corresponding_points.setEnabled(self.images_loaded)
        self.checkBox_show_points_thick_dims.setEnabled(self.images_loaded)
        self.checkBox_use_zgauss_opt.setEnabled(self.images_loaded)
        self.comboBox_method.setEnabled(self.images_loaded)

        self.groupBox_options.setVisible(self.images_loaded)
        self.groupBox_parameters.setVisible(self.images_loaded)
        self.groupBox_coordinates.setVisible(self.images_loaded)

        # enable controls for loading coordinates
        self.actionLoad_Load_Coordinates_Old.setEnabled(self.images_loaded)
        self.actionLoad_Load_Coordinates.setEnabled(self.images_loaded)
        self.actionClear_Coordinates.setEnabled(self.images_loaded)

        # enable the run correlation button (requires: images loaded, coordinates loaded)
        nfib = len(self.df[self.df["type"] == "FIB"])
        nfm = len(self.df[self.df["type"] == "FM"])
        nfm_poi = len(self.df[self.df["type"] == "POI"])

        # check if there are equal number of fiducial points in both images
        fiducial_points_equal = nfib == nfm
        min_fib_fiducial_points = nfm >= CORRELATION_PROPERTIES["min_fiducial_points"]
        min_fm_fiducial_points = nfib >= CORRELATION_PROPERTIES["min_fiducial_points"]
        min_poi_points = nfm_poi >= CORRELATION_PROPERTIES["min_poi_points"]

        correlation_enabled = bool(
            self.project_loaded
            and self.images_loaded
            and fiducial_points_equal
            and min_fib_fiducial_points
            and min_fm_fiducial_points
            and min_poi_points
        )

        self.pushButton_run_correlation.setEnabled(correlation_enabled)

        # display results
        results_enabled = correlation_enabled and not self.df.empty
        results_enabled = results_enabled and self.results_layer is not None
        self._show_results_widgets(results_enabled)

    def clear_project(self):
        # clear all the project data
        self.path = None
        self.project_loaded = False
        self.fib_image = None
        self.fm_image = None
        # TODO: more to clear?

    def set_project_path(self, path: str = None):
        if path is None:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                caption="Select Project Directory",
                directory=USER_PREFERENCES["default_path"],
            )
            if not path:
                return

        self.path = path
        self.lineEdit_project_path.setText(self.path)
        self.project_loaded = True

        # show the image controls
        self._show_project_controls()

    def _add_image_layer(self, image: np.ndarray, layer, name: str, color=None):
        """Add an image layer to the viewer, and attach a callback to update the correlation points"""
        # TODO: fix the note below
        # NOTE: we can't just update the data, because I can't figure out how to update the dtype on a napari layer properly.
        # so we have to remove the layer and re-add it for the data to be updated properly
        if layer is not None:
            self.viewer.layers.remove(layer)
            layer = None
        layer = self.viewer.add_image(
            image, name=name, blending="additive", colormap=color
        )  # TODO: migrate to properties
        # add callback to update the correlation points
        layer.mouse_drag_callbacks.append(self.update_correlation_points)

        return layer

    def _set_fib_image_path(self, filename: str = None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                caption="Open FIB Image",
                directory=self.path,
                filter=FILE_FILTERS,
            )
        if not filename:
            return

        # load the fib image, set the data, and set the translation
        try:
            fib_image, fib_pixel_size = load_and_parse_fib_image(filename)
        except Exception as e:
            logging.error(f"Error loading FIB Image: {e}")
            return

        self.load_fib_image(image=fib_image, pixel_size=fib_pixel_size, filename=filename)

    def load_fib_image(self, image: np.ndarray, pixel_size: float = None, filename: str = None):
        self.fib_image = image
        self.fib_pixel_size = pixel_size

        if filename is not None:
            self.lineEdit_fib_image_path.setText(filename)

        if self.fib_pixel_size is not None:
            self.doubleSpinBox_parameters_pixel_size.setValue(self.fib_pixel_size * 1e6)
        else:
            logging.warning("Pixel size not found in metadata")
            notifications.show_warning(
                "Pixel size not found in metadata, please set manually"
            )

        self.fib_image_layer = self._add_image_layer(
            image=self.fib_image, layer=self.fib_image_layer, name="FIB Image"
        )

        # update the translation of the FM and POI points
        self.translation = self.fib_image.shape[1]
        COORDINATE_LAYER_PROPERTIES["coordinates"]["FM"]["translation"] = (
            self.translation
        )
        COORDINATE_LAYER_PROPERTIES["coordinates"]["POI"]["translation"] = (
            self.translation
        )

        # apply translation to fm image
        if self.fm_image is not None:
            for layer in self.fm_image_layers:
                layer.translate = [0, 0, self.translation]
        self._show_project_controls()

    def _set_fm_image_path(self, filename: str = None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                caption="Open FM Image",
                directory=self.path,
                filter=FILE_FILTERS,
            )
        if not filename:
            return

        self.wizard = open_import_wizard(filename=filename)
        self.wizard.finished_signal.connect(self.handle_fm_finished_signal)

    def handle_fm_finished_signal(self, data: dict):
        self.fm_image = data["image"]
        self.fm_md = {"pixel_size": data["pixel_size"], 
                      "zstep": data["zstep"], 
                      "colours": data["colours"]}
        filename = data["filename"]

        if self.fm_image.ndim == 3:
            # add a channel dimension 
            self.fm_image = np.expand_dims(self.fm_image, axis=0)

        # rotation center is the centre of the image volume
        halfmax_dim = int(max(self.fm_image.shape) * 0.5)
        self.rotation_center = (halfmax_dim, halfmax_dim, halfmax_dim)

        # set range of the rotation center
        self.spinBox_parameters_rotation_center_z.setRange(0, halfmax_dim * 2)
        self.spinBox_parameters_rotation_center_x.setRange(0, halfmax_dim * 2)
        self.spinBox_parameters_rotation_center_y.setRange(0, halfmax_dim * 2)
        self.spinBox_parameters_rotation_center_z.setValue(halfmax_dim)
        self.spinBox_parameters_rotation_center_x.setValue(halfmax_dim)
        self.spinBox_parameters_rotation_center_y.setValue(halfmax_dim)

        # clear all fm image layers
        for layer in self.fm_image_layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

        # add each channel as a separate layer
        colors = self.fm_md.get("colours", None)
        for i, channel in enumerate(self.fm_image):
            if colors is not None:
                color = colors[i]
            else:
                color = None
            layer = self._add_image_layer(channel, None, f"FM Image Channel {i+1}", color=color)
            self.fm_image_layers.append(layer)

        self.lineEdit_fm_image_path.setText(filename)

        # apply translation to fm image
        if self.translation is not None:
            for layer in self.fm_image_layers:
                layer.translate = [0, 0, self.translation]

        self._show_project_controls()  # TODO: change this to a callback on the data layer?
        self.viewer.reset_view()

        try:
            self.wizard.viewer.close()
        except Exception as e:
            logging.error(f"Error closing wizard: {e}")
        self.wizard = None

    def _update_parameters(self):
        """Update the parameters for the correlation"""
        self.fib_pixel_size = self.doubleSpinBox_parameters_pixel_size.value() * 1e-6

        self.rotation_center = (
            self.spinBox_parameters_rotation_center_x.value(),
            self.spinBox_parameters_rotation_center_y.value(),
            self.spinBox_parameters_rotation_center_z.value(),
        )

        logging.info(
            f"Parameters updated: pixel_size: {self.fib_pixel_size}, Rotation Center: {self.rotation_center}"
        )

    def _update_user_preferences(self):
        # update the user preferences
        USER_PREFERENCES["use_mip"] = self.checkBox_use_mip.isChecked()
        USER_PREFERENCES["show_corresponding_points"] = (
            self.checkBox_show_corresponding_points.isChecked()
        )
        USER_PREFERENCES["show_thick_dims"] = (
            self.checkBox_show_points_thick_dims.isChecked()
        )
        USER_PREFERENCES["use_z_gauss_optim"] = self.checkBox_use_zgauss_opt.isChecked()

        # update the coordinate layer properties
        COORDINATE_LAYER_PROPERTIES["symbol"] = (
            self.comboBox_options_point_symbol.currentText()
        )
        COORDINATE_LAYER_PROPERTIES["size"] = self.spinBox_options_point_size.value()

        if self.coordinates_layer is not None:
            length = len(self.coordinates_layer.data)
            self.coordinates_layer.symbol = [
                COORDINATE_LAYER_PROPERTIES["symbol"]
            ] * length
            self.coordinates_layer.size = [COORDINATE_LAYER_PROPERTIES["size"]] * length

    def _toggle_thick_dims(self):
        """Toggle the thickness of the points layer"""

        show_thick_dims = self.checkBox_show_points_thick_dims.isChecked()

        z_dim = 2
        if show_thick_dims:
            z_dim = self.fm_image.shape[1] * 2 if self.fm_image is not None else 500
        self.viewer.dims.thickness = (z_dim, 1, 1)

    def toggle_mip(self):
        """Toggle the MIP of the fm image"""
        self.use_mip = not self.use_mip

        if self.is_drag_drop:
            self.viewer.layers.unlink_layers(self.fm_image_layers)

        # toggle mip
        for i, layer in enumerate(self.fm_image_layers):
            if self.use_mip:
                layer.data = np.amax(self.fm_image[i], axis=0) # TODO: faster/better way to do this?
            else:
                layer.data = self.fm_image[i]

        if self.is_drag_drop:
            self.viewer.layers.link_layers(self.fm_image_layers)

    def load_coordinates(self):
        fib_coord_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open FIB Coordinates File",
            directory=self.path,
            filter="Text Files (*.txt)",
        )

        if not fib_coord_filename:
            logging.debug("No fib coordinates `file selected")
            return

        fm_coord_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open FM Coordinates File",
            directory=self.path,
            filter="Text Files (*.txt)",
        )

        if not fm_coord_filename:
            logging.debug("No fm coordinates file selected")
            return

        # load the coordinates from the old style coordinate files
        fib_coordinates, fm_coordinates = parse_coordinates(
            fib_coord_filename=fib_coord_filename, fm_coord_filename=fm_coord_filename
        )

        # extract the poi coordinates from the fm coordinates (old style)
        nfib = len(fib_coordinates)
        poi_coordinates = fm_coordinates[nfib:]
        fm_coordinates = fm_coordinates[:nfib]

        # add the coordinates to the dataframe
        [self.add_point("FIB", coord[::-1], True) for coord in fib_coordinates]
        [self.add_point("FM", coord[::-1], True) for coord in fm_coordinates]
        [self.add_point("POI", coord[::-1], True) for coord in poi_coordinates]

        self._dataframe_updated()

    def load_coordinates_v2(self):
        # load data.csv
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open Coordinates File",
            directory=self.path,
            filter="CSV Files (*.csv)",
        )
        if not filename:
            logging.debug("No file selected")
            return

        # load the data, refresh the ui
        self.df = pd.read_csv(filename)
        self._dataframe_updated()


    def run_correlation(self):
        """Run the correlation between FIB and FM coordiantes."""

        self.df = self.df.sort_values(by=["idx"]) # sort by index


        fib_coords = self.df[self.df["type"] == "FIB"][["x", "y", "z"]].values.astype(
            np.float32
        )
        fm_coords = self.df[self.df["type"] == "FM"][["x", "y", "z"]].values.astype(
            np.float32
        )
        poi_coords = self.df[self.df["type"] == "POI"][["x", "y", "z"]].values.astype(
            np.float32
        )

        logging.info(f"FIB Coordinates: {fib_coords}")
        logging.info(f"FM Coordinates: {fm_coords}")
        logging.info(f"POI Coordinates: {poi_coords}")

        fm_image: np.ndarray = self.fm_image[0]  # only use the first channel (for shape, assume all are the same?)
        fib_image: np.ndarray = self.fib_image
        fib_pixel_size: float = self.fib_pixel_size
        rotation_center: Tuple[float, float, float] = self.rotation_center

        if fib_pixel_size is None or fib_pixel_size == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Unknown FIB Pixel Size",
                "FIB Pixel Size must be set before correlating.",
                QtWidgets.QMessageBox.Ok,
            )
            return

        # fib image shape minus metadata, fib_pixelsize (microns), fm_image_shape
        image_props = [fib_image.shape, fib_pixel_size * 1e6, fm_image.shape]
        logging.info(f"Image Properties: {image_props}")

        assert fib_image.ndim == 2, "FIB Image must be 2D"
        assert fm_image.ndim == 3, "FM Image must be 3D"
        assert fib_pixel_size is not None, "FIB Pixel Size must be set"
        assert rotation_center is not None, "Rotation Center must be set"
        assert isinstance(rotation_center, tuple), "Rotation Center must be a tuple"
        assert len(rotation_center) == 3, "Rotation Center must have 3 values"

        # run correlation
        self.correlation_results = run_correlation(
            fib_coords=fib_coords,
            fm_coords=fm_coords,
            poi_coords=poi_coords,
            image_props=image_props,
            rotation_center=rotation_center,
            path=self.path,
            fib_image_filename=os.path.basename(self.lineEdit_fib_image_path.text()),
            fm_image_filename=os.path.basename(self.lineEdit_fm_image_path.text()),
        )

        # set the poi coordinate from the results
        self.poi_coordinate = self.correlation_results["output"]["poi"][0]["px_m"]

        self._show_correlation_results(self.correlation_results)

    def _show_correlation_results(self, correlation_results: dict, refresh_only: bool = False):
        # show the results on the fib image
        dat = []
        poi_image_coordinates = correlation_results["output"]["poi"]
        for i, coord in enumerate(poi_image_coordinates):
            dat.append(coord["image_px"][::-1])

            # image_px: image coordinates (0,0) in top left
            # px: coordinates in microscope image (centre at 0,0)
            # px_m: coordinates in microscope image (m) (centre at 0,0)

        # TODO: add the results to the main dataframe?
        if self.results_layer is not None:
            self.viewer.layers.remove(self.results_layer)
            self.results_layer = None
        text = {
            "text": [f"POI{i+1}" for i in range(len(dat))],
            "size": 10,
            "color": "white",
            "anchor": "upper_right",
        }
        self.results_layer = self.viewer.add_points( # TODO: consolidate this and poi_coordinate_layer, they display the same information
            dat,
            name=RESULTS_LAYER_PROPERTIES["name"],
            ndim=RESULTS_LAYER_PROPERTIES["ndim"],
            size=RESULTS_LAYER_PROPERTIES["size"],
            symbol=RESULTS_LAYER_PROPERTIES["symbol"],
            border_color=RESULTS_LAYER_PROPERTIES["border_color"],
            face_color=RESULTS_LAYER_PROPERTIES["face_color"],
            blending=RESULTS_LAYER_PROPERTIES["blending"],
            opacity=RESULTS_LAYER_PROPERTIES["opacity"],
            text=text,
        )
        self.results_layer.mouse_drag_callbacks.append(self.update_correlation_points)

        # reset selection to coordiantes layer
        self.viewer.layers.selection.active = self.coordinates_layer

        if refresh_only:
            return

        self._show_results_widgets(True)

        # display the transformation data in the results tab
        trans_data = correlation_results["output"]["transformation"]
        error_data = correlation_results["output"]["error"]
        input_data = correlation_results["input"]

        # euler angles
        euler_angles = [f"{i:.2f}" for i in trans_data["rotation_eulers"]]
        # make euler angles a string with the degrees symbol between each angle
        euler_angles = "°, ".join(euler_angles) + "°"  # e.g. 90°, 90°, 90°

        self.label_results_transform_euler_rotation_value.setText(f"{euler_angles}")

        # translation
        rcustom = [f"{i:.1f}" for i in input_data["rotation_center_custom"]]
        trans_rzero = [
            f"{i:.1f}"
            for i in trans_data["translation_around_rotation_center_zero"][:2]
        ]
        trans_rcustom = [
            f"{i:.1f}"
            for i in trans_data["translation_around_rotation_center_custom"][:2]
        ]

        # convert the translation to a string
        rcustom = ", ".join(rcustom)
        trans_rzero = ", ".join(trans_rzero)
        trans_rcustom = ", ".join(trans_rcustom)

        self.label_results_transform_rotation_zero_value.setText(f"{trans_rzero}")
        self.label_results_transform_rotation_center_custom.setText(
            f"Rotation Center @ {rcustom}"
        )
        self.label_results_transform_rotation_custom_value.setText(f"{trans_rcustom}")

        # scale
        self.label_results_scale_value.setText(f"{trans_data['scale']:.2f}")

        # error
        mae = ", ".join([f"{i:.2f}" for i in error_data["mean_absolute_error"]])
        self.label_results_error_mean_absolute_value.setText(f"{mae}")
        self.label_results_error_rms_value.setText(f"{error_data['rms_error']:.2f}")

        # display the error data in the results tab
        delta_2d = np.array(error_data["delta_2d"]).T
        reproj_3d = np.array(error_data["reprojected_3d"]).T

        # format the table
        self.tableWidget_results_error.clear()
        self.tableWidget_results_error.setRowCount(delta_2d.shape[0])
        self.tableWidget_results_error.setColumnCount(2)

        # set column headers
        self.tableWidget_results_error.setHorizontalHeaderLabels(["dx", "dy"])

        set_table_properties(self.tableWidget_results_error) # TODO: change to pandas model

        # loop through all points and and print (x, y)
        for i, (dx, dy) in enumerate(delta_2d):
            dx, dy = f"{dx:.2f}", f"{dy:.2f}"

            self.tableWidget_results_error.setItem(i, 0, QtWidgets.QTableWidgetItem(dx))
            self.tableWidget_results_error.setItem(i, 1, QtWidgets.QTableWidgetItem(dy))

        self._draw_error_data(reproj_3d)
        self.display_milling_stages()

    def update_poi_coordinate(self, layer, event):
        """Update the point of interest for drag-drop method"""

        if not self.is_drag_drop:
            return
        
        if "Control" not in event.modifiers:
            return

        if self.fib_image is None or self.fm_image is None:
            logging.info("No images loaded")
            return

        target_layer = check_coordinates_inside_layer(event, [self.fib_image_layer])

        if target_layer is None:
            logging.info("Coordinates are not within bounds of any layer")
            return

        position = target_layer.world_to_data(event.position)
        logging.info(f"Target Layer: {target_layer.name}, Position: {position}")

        # clear poi layer
        self.poi_coordinate_layer.data = []
        self.poi_coordinate_layer.data = [position]
        
        # convert to microscope coordinates
        shape = self.fib_image.shape
        pixelsize = self.fib_pixel_size
        if pixelsize is None or pixelsize == 0:
            napari.utils.notifications.show_warning("FIB Pixel size not set, correlation result is not valid. " 
                                                    "Please set the pixel-size manually") 
            pixelsize = 0 #3.25e-8
        cy, cx = np.asarray(shape) // 2

        # distance from centre?
        dy = float(-(position[0] - cy)) * pixelsize  # neg = down
        dx = float(position[1] - cx)  * pixelsize # neg = left

        self.poi_coordinate = (dx, dy)
        logging.info(f"POI Coordinate: {self.poi_coordinate}, pixelsize: {pixelsize}")

        # save correlation result
        results = {
            "input": {
                "method": "drag-drop",
                "image_properties": {
                    "fib_image_filename": os.path.basename(self.lineEdit_fib_image_path.text()),
                    "fib_image_shape": list(self.fib_image.shape),
                    "fib_pixel_size_um": self.fib_pixel_size * 1e6,
                    "fm_image_filename": os.path.basename(self.lineEdit_fm_image_path.text()),
                    "fm_image_shape": list(self.fm_image.shape),
                }
            },
            "output": {
                "poi": [
                    {"image_px": [int(p) for p in position[::-1]],
                    "px_m": list(self.poi_coordinate)
                    }],
            },
            # TODO: save affine data from layer transforms?
            "metadata": {
                "project_path": self.path,
                "data_path": self.path,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
        }
        save_correlation_data(results, self.path)

        self.display_milling_stages()

    def apply_refractive_index_correction(self):

        # get surface point
        surface_coords = self.df[self.df["type"] == "Surface"][["x", "y", "z"]].values.astype(
            np.float32
        )
        if len(surface_coords) == 0:
            napari.utils.notifications.show_warning(
                "No surface coordinates found. Please add a surface point to the coordinates."
            )
            return

        surface_coord = surface_coords[0]

        # get result point (poi)
        poi_image_coordinates = self.correlation_results["output"]["poi"][0]["image_px"]
        logging.info(f"Surface Coord: {surface_coord}, PoI Coord: {poi_image_coordinates}")

        # get correction factor
        correction_factor = self.doubleSpinBox_refractive_correction_factor.value()

        # apply correction factor to poi
        depth = poi_image_coordinates[1] - surface_coord[1] # assume poi always below surface, y-axis

        corrected_depth = depth * correction_factor
        logging.info(f"Correction Factor: {correction_factor}, Depth: {depth}, Corrected Depth: {corrected_depth}")

        # update the poi coordinate in poi
        corrected_poi = (poi_image_coordinates[0], surface_coord[1] + corrected_depth)
        logging.info(f"Corrected PoI: {corrected_poi}")

        # show initial point 
        INITIAL_POI_CONFIG = {"name": "Initial PoI", 
                              "face_color": "blue", 
                              "size": 10, 
                              "symbol": "disc",}

        if INITIAL_POI_CONFIG["name"] in self.viewer.layers:
            self.viewer.layers.remove(INITIAL_POI_CONFIG["name"])
        self.viewer.add_points(
            [poi_image_coordinates[::-1]],
            name=INITIAL_POI_CONFIG["name"],
            face_color=INITIAL_POI_CONFIG["face_color"],
            size=INITIAL_POI_CONFIG["size"],
            symbol=INITIAL_POI_CONFIG["symbol"],
            text={"text": [INITIAL_POI_CONFIG["name"]], 
                  "size": 10, 
                  "color": "white", 
                  "anchor": "upper_right"},
            blending="additive",
        )

        # update the poi coordinate in poi
        # update the results dictionary, save the results

        # convert to microscope coordinates
        shape = self.fib_image.shape
        pixelsize = self.fib_pixel_size
        cy, cx = np.asarray(shape) // 2
        logging.info(f"Image Information: Shape: {shape}, Pixelsize: {pixelsize}, Centre: {cy, cx}")

        # distance from centre
        dy = float(-(corrected_poi[1] - cy)) * pixelsize    # neg = down
        dx = float(corrected_poi[0] - cx)  * pixelsize      # neg = left

        self.poi_coordinate = (dx, dy)
        self.correlation_results["output"]["poi"][0]["image_px"] = [float(corrected_poi[0]), float(corrected_poi[1])]
        self.correlation_results["output"]["poi"][0]["px_m"] = list(self.poi_coordinate)
        self.correlation_results["output"]["poi"][0]["px_um"] = [dx*1e6, dy*1e6]
        self.correlation_results["output"]["poi"][0]["px"] = [dx/pixelsize, dy/pixelsize]

        logging.info(f'Final Results: {self.correlation_results["output"]["poi"]}')

        self._show_correlation_results(self.correlation_results, refresh_only=True)
        self.display_milling_stages()

        full_correlation_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "data_path": self.path,
                "csv_path": os.path.join(self.path, "data.csv"),
                "project_path": self.path, # TODO: add project path
            },
            "correlation": self.correlation_results,
        }

        save_correlation_data(full_correlation_data, self.path)

    def display_milling_stages(self):
        """Attempt to display milling stages on the correlated image."""
        try:
            from fibsem.milling.patterning.patterns2 import FiducialPattern
            from fibsem.structures import Point
            from fibsem.ui.napari.patterns import draw_milling_patterns_in_napari

            milling_stages = self.parent_ui.milling_stages

            poi = self.poi_coordinate
            if poi is None:
                return
            point = Point(x=poi[0], y=poi[1])

            for milling_stage in milling_stages:
                # don't move fiducial patterns
                if isinstance(milling_stage.pattern, FiducialPattern):
                    continue
                milling_stage.pattern.point = point

            milling_pattern_layers = draw_milling_patterns_in_napari(
                    viewer=self.viewer,
                    image_layer=self.fib_image_layer,
                    milling_stages=milling_stages,
                    pixelsize=self.fib_pixel_size,
                    draw_crosshair=True,
                    )
            for layer in milling_pattern_layers:
                self.viewer.layers[layer].visible = True
        except Exception as e:
           logging.error(f"Error displaying milling stages: {e}")

    def update_correlation_points(self, layer, event):
        """Update correlation points for multi-point correlation"""
        # event.position  # (z, y, x)

        if self.is_drag_drop:
            return # don't update points in drag-drop view

        if "Control" not in event.modifiers and "Shift" not in event.modifiers and "Alt" not in event.modifiers:
            return

        if self.fib_image is None or self.fm_image is None:
            logging.info("No images loaded")
            return

        target_layers = [layer for layer in self.fm_image_layers] + [self.fib_image_layer]

        target_layer = check_coordinates_inside_layer(event, target_layers)

        if target_layer is None:
            logging.info("Coordinates are not within bounds of any layer")
            return

        position = target_layer.world_to_data(event.position)
        logging.debug(f"Target Layer: {target_layer.name}, Position: {position}")

        # if shift is pressed, add fiducial point to corresponding layer
        # if control is pressed, add poi point (fm)
        if "Shift" in event.modifiers:
            if target_layer == self.fib_image_layer:
                self.add_point("FIB", position)

            elif target_layer in self.fm_image_layers:
                self.add_point("FM", position)
        elif "Control" in event.modifiers:
            if target_layer in self.fm_image_layers:
                self.add_point("POI", position)
        elif "Alt" in event.modifiers:
            if target_layer == self.fib_image_layer:
                self.add_point("Surface", position)

    def add_point(self, layer: str, position: list[float], from_file: bool = False):
        logging.info(f"Adding point to layer: {layer}, Position: {position}")

        # if the position is 2D (FIB), add a z coordinate
        if len(position) == 2:
            position = [0, position[0], position[1]]

        if len(position) == 4:
            position = position[1:]

        # _map the point to the coordinates layer
        clayer_props = COORDINATE_LAYER_PROPERTIES["coordinates"]
        color = clayer_props[layer]["color"]
        translation = clayer_props[layer]["translation"]

        # get the number of points of that type in the dataframe
        n = len(self.df[self.df["type"] == layer]) + 1

        # extract coordinate data
        z, y, x = position

        # z_gauss optimisation
        self.use_z_gauss_optim = self.checkBox_use_zgauss_opt.isChecked()
        if layer in ["FM", "POI"] and self.use_z_gauss_optim and not from_file:
            prev_z = z
            prev_x, prev_y = x, y

            try:
                # getzGauss can fail, so we need to catch the exception
                # zval, z, _ = multi_channel_get_z_guass(image=self.fm_image, x=x, y=y) # TODO: indicate to user which channel is being used
                # logging.info(f"Using Z-Gauss optimisation: {z}, previous z: {prev_z}")
                
                # TODO: enable after more thorough testing
                # logging.info(f"Using multi-channel zyx-targeting")
                ch, (x,y,z) = multi_channel_zyx_targeting(self.fm_image, int(x), int(y), int(z))
                
                if z is None:
                    raise RuntimeError("Z-Gauss optimisation failed: optimisation failed")
                # TODO: also check if the point is within the image bounds
                if z < 0 or z > self.fm_image.shape[1]:
                    raise RuntimeError("Z-Gauss optimisation failed: z out of range")
                if y < 0 or y > self.fm_image.shape[2]:
                    raise RuntimeError("Z-Gauss optimisation failed: y out of range")
                if x < 0 or x > self.fm_image.shape[3]:
                    raise RuntimeError("Z-Gauss optimisation failed: x out of range")


            except RuntimeError as e:
                logging.error(f"Error in z-gauss optimisation: {e}")
                # show a warning to the user
                notifications.show_warning(
                    f"Error in Z-Gauss optimisation, using previous z value: {prev_z}"
                )
                z = prev_z
                x, y = prev_x, prev_y

        if layer == "FIB" and self.use_z_gauss_optim and not from_file:
            from fibsem.correlation.util import hole_fitting_FIB
            x, y = hole_fitting_FIB(img=self.fib_image, x=int(x), y=int(y))

        # add the point to the dataframe
        df_tmp = pd.DataFrame([{"x": x,"y": y, "z": z, 
                                "type": layer,"color": color, 
                                "idx": n, "translation": translation,}
                                ])

        if self.df.empty:
            self.df = df_tmp
        else:
            self.df = pd.concat([self.df, df_tmp], axis=0, ignore_index=True)

        self._dataframe_updated()

    def remove_point(self, layer: str, index: int):
        pass

    def clear_coordinates(
        self,
    ):
        # remove all entries from the dataframe
        self.df = pd.DataFrame([], columns=DATAFRAME_PROPERTIES["columns"])

        # remove results
        if self.results_layer is not None and self.results_layer in self.viewer.layers:
            self.viewer.layers.remove(self.results_layer)
            self.results_layer = None
        if self.reprojection_layer is not None and self.reprojection_layer in self.viewer.layers:
            self.viewer.layers.remove(self.reprojection_layer)
            self.reprojection_layer = None
        # remove the corresponding points
        if self.line_layer is not None and self.line_layer in self.viewer.layers:
            self.viewer.layers.remove(self.line_layer)
            self.line_layer = None

        if self.coordinates_layer is not None:
            self._dataframe_updated()

    def _dataframe_updated(self):
        """"""
        self.df = self.df.sort_values(by=["type", "idx"])
        self.df.reset_index(drop=True, inplace=True)
        self._toggle_thick_dims()
        self._draw_points_to_layer()

        self.setup_table_view()

        self._save_dataframe()
        self._show_corresponding_points()
        self._show_project_controls()

    def setup_table_view(self):

        self.model = PandasTableModel(self.df, display_columns=["type", "idx", "x", "y", "z"])
        self.tableView_coordinates.setModel(self.model) # TODO: this doesn't need to be re-initialized every time

        # set minimum height to stretch the table
        self.tableView_coordinates.verticalHeader().setMinimumSectionSize(25)
        # connect signals
        self.model.dataChanged.connect(self.on_data_changed)

    def on_data_changed(self, df):
        # reindex the dataframe
        self.df = df.reset_index(drop=True)
        self._dataframe_updated()

        # TODO: split into separate table views

    def _show_corresponding_points(self):
        """Show the corresponding points between the two images."""

        # remove the line layer if it exists
        if self.line_layer is not None:
            if self.line_layer in self.viewer.layers:
                self.viewer.layers.remove(self.line_layer)
                self.line_layer = None

        if not self.checkBox_show_corresponding_points.isChecked():
            return

        if self.df.empty:
            logging.info("No data to show corresponding points")
            return

        logging.debug("Showing Corresponding Points")

        # show the matching coordinates in the FIB - FM

        # TODO: show matching points based on idx, not order
        # show the user a warning if there are unmatched points, or points with the same idx
        # TODO: use last_idx to see which points were changed?

        df_sorted = self.df.sort_values(by=["type", "idx"])

        fib_coords = df_sorted[df_sorted["type"] == "FIB"][["x", "y"]].values
        fm_coords = df_sorted[df_sorted["type"] == "FM"][["x", "y", "translation"]].values

        # add the translation to the x coordinates (to account for the translation of the fm image)
        fm_coords[:, 0] += fm_coords[:, 2]
        # drop the translation column
        fm_coords = fm_coords[:, :2]

        line_data = []
        for i, (fib, fm) in enumerate(zip(fib_coords, fm_coords)):
            # reverse the coordinates to match napari
            fib, fm = fib[::-1], fm[::-1]
            # draw a line between the points
            line_data.append([fib, fm])

        # self.line_layer.data = line_data  # TODO: investigate why this doesn't work
        self.line_layer = self.viewer.add_shapes(
            line_data,
            ndim=2,
            name=LINE_LAYER_PROPERTIES["name"],
            shape_type=LINE_LAYER_PROPERTIES["shape_type"],
            edge_color=LINE_LAYER_PROPERTIES["edge_color"],
            edge_width=LINE_LAYER_PROPERTIES["edge_width"],
        )
        self.line_layer.mouse_drag_callbacks.append(self.update_correlation_points)

        # reset selection to coordiantes layer
        self.viewer.layers.selection.active = self.coordinates_layer

    def _save_dataframe(self):
        # save the dataframe to a csv file
        self.df.to_csv(os.path.join(self.path, "data.csv"), index=False)

    def _draw_points_to_layer(self):
        """draw the coordinates to the points layer"""

        # reassign the data to the layer
        pt_data = self.df[["x", "y", "z", "translation"]].values

        # add the translation to the x coordinates (to account for the translation of the fm image)
        pt_data[:, 0] += pt_data[:, 3]

        # cast all the data to float32
        pt_data = pt_data.astype(np.float32)

        layer = self.coordinates_layer
        layer.events.data.disconnect(self.coordinates_updated_from_ui)
        # reverse the coordinates to match napari
        layer.data = pt_data[:, [2, 1, 0]]
        # create a color array that matches the length of the data in each frame
        layer.face_color = self.df["color"].values.astype(str)
        layer.border_color = self.df["color"].values.astype(str)
        layer.properties = self.df  # text
        layer.events.data.connect(self.coordinates_updated_from_ui)

    def _draw_error_data(self, reprojected_points: np.ndarray):
        """Draw the reprojected data to the error layer
        Args:
            reprojected_points (np.ndarray): the reprojection of the FM points

        """
        if self.reprojection_layer is not None:
            try:
                self.viewer.layers.remove(self.reprojection_layer)
            except ValueError as e:
                logging.error(f"Error removing reprojection layer: {e}")
            self.reprojection_layer = None

        if reprojected_points is None:
            return

        if self.df.empty:
            logging.info("No data to show error data")
            return

        # get FIB coordinates
        # fib_coords = self.df[self.df["type"] == "FIB"][["x", "y"]].values

        # add error data to fib coordinates
        # error_coords = fib_coords + error_delta
        reprojected_points = reprojected_points[:, [1, 0]]

        reprojection_text = {
            "text": [f"E{i+1}" for i in range(len(reprojected_points))],
            "size": 8,
            "color": "white",
            "anchor": "lower_left",
        }

        self.reprojection_layer = self.viewer.add_points(
            reprojected_points,
            name=REPROJECTION_LAYER_PROPERTIES["name"],
            ndim=REPROJECTION_LAYER_PROPERTIES["ndim"],
            size=REPROJECTION_LAYER_PROPERTIES["size"],
            symbol=REPROJECTION_LAYER_PROPERTIES["symbol"],
            face_color=REPROJECTION_LAYER_PROPERTIES["face_color"],
            border_color=REPROJECTION_LAYER_PROPERTIES["border_color"],
            opacity=REPROJECTION_LAYER_PROPERTIES["opacity"],
            text=reprojection_text,
        )
        self.reprojection_layer.mouse_drag_callbacks.append(self.update_correlation_points)
        self.viewer.layers.selection.active = self.coordinates_layer

    # @qthrottled(timeout=100)
    def coordinates_updated_from_ui(self, event):
        """Update the dataframe when the coordinates are updated in the UI"""

        if event.action not in ["added", "removed", "changed"]:
            return

        # get which point was moved
        index: tuple[int] = list(event.data_indices)

        if not index:
            return  # no points selected, return

        if self.df.empty:
            return

        logging.debug(
            {
                "event_type": event.type,
                "event_action": event.action,
                "event_source": event.source.name,
                "index": index,
            }
        )

        if event.action == "added":
            logging.info(f"Data Added: {index}")
            return  # don't do anything when data is added, it's handled by the add_point method

        if event.action == "changed":
            # only trigger when the user changes the data

            # TODO: prevent data from being outside of extent of the image
            data = event.source.data

            # update the dataframe
            self.df.loc[index, ["x", "y", "z"]] = data[index][:, ::-1]

            # subtract the translation from the x coordinate, if it's an FM point
            trans_indices = self.df[self.df["type"].isin(["FM", "POI"])].index
            trans_changed_index = np.intersect1d(index, trans_indices)
            self.df.loc[trans_changed_index, "x"] -= self.df.loc[
                trans_changed_index, "translation"
            ]

        if event.action == "removed":
            # remove the point from the dataframe
            self.df.drop(index, inplace=True)

            # reset the index
            self.df.reset_index(inplace=True, drop=True)

            # rename the idx column to match the number of types of points
            self.df["idx"] = self.df.groupby("type").cumcount() + 1

        self._dataframe_updated()

        return





# link the facecolor / bordercolor to the dataframe

# add 'project' concept, save directory, save project file
# add 'open project' concept, load project file, load images, load coordinates
# add an auto-save toggle

# think about how to embed in autolamella with targetting

# display the transformation matrix
# display transformed image onto fib image
# integrate the find beads tool

# add dynamic instructions
# add show instructions toggle

# add a 'help' tab
# add an options tab

# create an iterative optimiser for correlation refinement

# update to use qtableview with pandas model
# https://doc.qt.io/qtforpython-6/examples/example_external_pandas.html
# https://stackoverflow.com/questions/47020995/pyqt5-updating-dataframe-behind-qtablewidget
# allow editting of the coordinates in the table
# allow re-ordering of the coordinates in the table

# DONE:
# load images
# show text on the points (point index)
# add z-gauss fitting to the points in fm
# add data tools: reslice (iterpolate), set pixel size, set origin, set rotation
# when the correlation is run, show the results on the fib image
# add a 'clear' button to clear all the points
# show the corresponding points on the fm image
# add a toggle for showing 'thick' dimensions
# add a toggle for showing the mip of the fm image

# remove the metadata bar from fib image (and auto detect?)
# editable pixelsize for fib image + warning when not set
# editable rotation center

# functionalise correlation code
# add options to set the size / symbol / color of the points
# show the error data on the image

# connect results layer and corresponding points layer to click events, so user doesn't have to select the layer
# when a point is moved, update the dataframe
# when a point is deleted, update the dataframe
# when a point is added, update the dataframe
# save the data as data.csv
# when the correlation is run, save the results to a file

# results
# show results in results tab

# project
# create flow: create / load project -> load images -> select / load coordinates -> run correlation -> save results
# save the project file as a yaml file


def main():
    viewer = napari.Viewer()
    widget = CorrelationUI(viewer)

    # create a button to run correlation
    viewer.window.add_dock_widget(
        widget, area="right", name="3DCT Correlation", tabify=True
    )

    napari.run()


if __name__ == "__main__":
    main()
