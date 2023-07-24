import logging
import os
from copy import deepcopy

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from scipy.ndimage import median_filter

from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.imaging import _tile
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamType, FibsemImage, FibsemStagePosition,
                               MicroscopeSettings, Point)
from fibsem.ui.qtdesigner_files import FibsemTileWidget
from fibsem.ui import utils as ui_utils
from PyQt5.QtCore import pyqtSignal, pyqtSlot


PATH = os.path.join(cfg.DATA_PATH, "tile")
os.makedirs(PATH, exist_ok=True)

class FibsemTileWidget(FibsemTileWidget.Ui_Form, QtWidgets.QWidget):
    _stage_position_moved = pyqtSignal(FibsemStagePosition)
    _stage_position_added = pyqtSignal(FibsemStagePosition)
    
    def __init__(
        self,
        microscope: FibsemMicroscope,
        settings: MicroscopeSettings,
        viewer: napari.Viewer,
        parent=None,
    ):
        super(FibsemTileWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings

        self.viewer = viewer

        self.image = None
        self._image_layer = None
        self._reprojection_layer = None

        self.positions = []

        self.setup_connections()

        self._update_ui()

    def setup_connections(self):

        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_load_image.clicked.connect(self.load_image)

        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])

        self.pushButton_move_to_position.clicked.connect(self._move_position_pressed)

        self.comboBox_tile_position.currentIndexChanged.connect(self._update_current_position_info)
        self.pushButton_update_position.clicked.connect(self._update_position_pressed)
        self.pushButton_update_position.setStyleSheet("background-color: blue")
        self.pushButton_remove_position.clicked.connect(self._remove_position_pressed)
        self.pushButton_remove_position.setStyleSheet("background-color: red")

        self.pushButton_save_positions.clicked.connect(self._save_positions_pressed)
        self.pushButton_load_positions.clicked.connect(self._load_positions)

        # signals
        # self._stage_position_added.connect(self._position_added_callback)

    def run_tile_collection(self):

        print("run_tile_collection")

        beam_type = BeamType[self.comboBox_tile_beam_type.currentText()]
        grid_size = self.doubleSpinBox_tile_grid_size.value() * constants.MICRO_TO_SI
        tile_size = self.doubleSpinBox_tile_tile_size.value() * constants.MICRO_TO_SI

        self.settings.image.hfw = tile_size
        self.settings.image.beam_type = beam_type
        self.settings.image.save = True
        self.settings.image.save_path = PATH
        self.settings.image.label = self.lineEdit_tile_label.text() 

        if self.settings.image.label == "":
            napari.utils.notifications.show_error(f"Please enter a filename for the image")
            return

        image = _tile._tile_image_collection_stitch(self.microscope, self.settings, grid_size, tile_size, overlap=0)

        self._update_viewer(image)


    def load_image(self):

        path = ui_utils._get_file_ui( msg="Select image to load", path=cfg.DATA_TILE_PATH, _filter="Image Files (*.tif *.tiff)", parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        image = FibsemImage.load(path)

        self._update_viewer(image)

    def _update_ui(self):

        _image_loaded = self.image is not None
        self.pushButton_update_position.setEnabled(_image_loaded)
        self.pushButton_remove_position.setEnabled(_image_loaded)
        self.pushButton_move_to_position.setEnabled(_image_loaded)

        if _image_loaded:
            self.label_instructions.setText("Alt+Click to add a position, Double Click to Move the Stage...")
        else:
            self.label_instructions.setText("Please take or load an overview image...")

        _positions_loaded = len(self.positions) > 0
        self.pushButton_move_to_position.setEnabled(_positions_loaded)


    def _update_viewer(self, image: FibsemImage =  None):

        if image is not None:
            self.image = image
        
            arr = median_filter(self.image.data, size=3)
            try:
                self._image_layer.data = arr
            except:
                self._image_layer = self.viewer.add_image(arr, name="tile", colormap="gray", blending="additive")

            # draw a point on the image at center
            ui_utils._draw_crosshair(viewer=self.viewer,eb_image= self.image, ib_image= self.image,is_checked=True) 


            self._image_layer.mouse_drag_callbacks.append(self._on_click)
            self._image_layer.mouse_double_click_callbacks.append(self._on_double_click)
    

        self._draw_positions() # draw the reprojected positions on the image

        self.viewer.layers.selection.active = self._image_layer

        self._update_ui()


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
            napari.utils.notifications.show_info(
                f"Clicked outside image dimensions. Please click inside the image to move."
            )
            return

        point = conversions.image_to_microscope_image_coordinates(
            Point(x=coords[1], y=coords[0]), self.image.data, self.image.metadata.pixel_size.x,
        )

        return coords, point

    def _on_click(self, layer, event):
        
        if event.button != 1 or 'Alt' not in event.modifiers:
            return 

        coords, point = self._validate_mouse_click(layer, event)

        _new_position = self.microscope._calculate_new_position( 
                    settings=self.settings, 
                    dx=point.x, dy=point.y, 
                    beam_type=self.image.metadata.image_settings.beam_type, 
                    base_position=self.image.metadata.microscope_state.absolute_position)            
        _new_position.name = f"Position {len(self.positions)+1:02d}" # TODO: allow user to edit this

        # now should be able to just move to _new_position
        self.positions.append(_new_position)

        # we could save this position as well, use it to pre-select a bunch of lamella positions?
        self._update_position_info()
        self._update_viewer()

        self._stage_position_added.emit(_new_position)

    def _on_double_click(self, layer, event):
        
        if event.button !=1:
            return
        coords, point = self._validate_mouse_click(layer, event)

        _new_position = self.microscope._calculate_new_position( 
            settings=self.settings, 
            dx=point.x, dy=point.y, 
            beam_type=self.image.metadata.image_settings.beam_type, 
            base_position=self.image.metadata.microscope_state.absolute_position)   

        self._move_to_position(_new_position)
        self._update_viewer()

    def _update_current_position_info(self):

        idx = self.comboBox_tile_position.currentIndex()
        if idx != -1:
            self.lineEdit_tile_position_name.setText(self.positions[idx].name)
        else:
            self.lineEdit_tile_position_name.setText("")

    def _update_position_info(self):
        idx = self.comboBox_tile_position.currentIndex()
        self.comboBox_tile_position.clear()
        self.comboBox_tile_position.addItems([pos.name for pos in self.positions])
        if idx != -1:
            self.comboBox_tile_position.setCurrentIndex(idx)

        msg = ""
        for pos in self.positions:
            msg += f"{pos.name}:\t x:{pos.x*1e3:.2f}mm, y:{pos.y*1e3:.2f}mm, z:{pos.z*1e3:.2f}m\n"
        self.label_position_info.setText(msg)


    def _update_position_pressed(self):
        logging.info(f"Updating position...")
        _position = self.positions[self.comboBox_tile_position.currentIndex()]
        
        name = self.lineEdit_tile_position_name.text()
        if name == "":
            napari.utils.notifications.show_info(f"Please enter a name for the position")
            return

        _position.name = name 
        self._update_position_info()
        self._update_viewer()

    def _remove_position_pressed(self):
        logging.info(f"Removing position...")
        _position = self.positions[self.comboBox_tile_position.currentIndex()]
        self.positions.remove(_position)
        self._update_position_info()
        self._update_viewer()


    def _move_position_pressed(self):

        logging.info(f"Moving to position...")
        _position = self.positions[self.comboBox_tile_position.currentIndex()]
        logging.info(f"Moving To: {_position}")
        self._move_to_position(_position)

    def _move_to_position(self, _position:FibsemStagePosition)->None:
        self.microscope._safe_absolute_stage_movement(_position)
        self._stage_position_moved.emit(_position)
        self._update_viewer()


    def _load_positions(self):

        logging.info(f"Loading Positions...")
        path = ui_utils._get_file_ui( msg="Select a position file to load", 
            path=self.settings.image.save_path, 
            _filter= "*yaml", 
            parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        pdict = utils.load_yaml(path)
        
        positions = [FibsemStagePosition.__from_dict__(p) for p in pdict]
        self.positions = self.positions + positions # append? or overwrite

        self._update_position_info()
        self._update_viewer()

    def _save_positions_pressed(self):
        
        logging.info(f"Saving Positions...")
        
        path = ui_utils._get_save_file_ui(msg = "Select a file to save the positions to",
            path = self.settings.image.save_path,
            _filter= "*yaml",
            parent=self,
        )

        if path == "":
            napari.utils.notifications.show_info(f"No file selected, not saving.")
            return

        # save the positions
        pdict = [p.__to_dict__() for p in self.positions]
        utils.save_yaml(path, pdict)

        napari.utils.notifications.show_info(f"Saved positions to {path}")


    def _draw_positions(self):
        
        logging.info(f"Drawing Reprojected Positions...")
        current_position = self.microscope.get_stage_position()
        current_position.name = f"Current Position"

        if self.image:
            
            drawn_positions = self.positions + [current_position]
            points = _tile._reproject_positions(self.image, drawn_positions)

            data = []
            for pt in points:
                # reverse to list
                data.append([pt.y, pt.x])

            text = {
                "string": [pos.name for pos in drawn_positions],
                "color": "lime", # TODO: separate colour for current position
                "translation": np.array([-50, 0]),
            }
            if self._reprojection_layer is None:
            
                 self._reprojection_layer = self.viewer.add_points(
                    data,
                    name="Positions",
                    text=text,
                    size=60,
                    edge_width=7,
                    edge_width_is_relative=False,
                    edge_color="transparent",
                    face_color="lime",
                    blending="translucent",
                    symbol="cross",
                )
            else:
                self._reprojection_layer.data = data
                self._reprojection_layer.text = text

    def _align_image(self):

        # load another image

        # select layer

        # allow translation, rotation, scaling of image

        pass








def main():

    viewer = napari.Viewer(ndisplay=2)
    microscope, settings = utils.setup_session()
    tile_widget = FibsemTileWidget(microscope, settings, viewer=viewer)
    viewer.window.add_dock_widget(
        tile_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()