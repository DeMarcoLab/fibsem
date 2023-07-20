import logging
import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import FibsemMicroscope, TescanMicroscope, ThermoMicroscope, DemoMicroscope
from fibsem.structures import MicroscopeSettings, BeamType, FibsemImage, Point
from fibsem.ui.qtdesigner_files import FibsemTileWidget
import os

from fibsem.imaging import _tile
from scipy.ndimage import median_filter

from copy import deepcopy

PATH = os.path.join(cfg.DATA_PATH, "tile")
os.makedirs(PATH, exist_ok=True)
from fibsem.ui import utils as ui_utils 

from PyQt5.QtCore import pyqtSignal, pyqtSlot


class FibsemTileWidget(FibsemTileWidget.Ui_Form, QtWidgets.QWidget):
    _position_changed = pyqtSignal()
    
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

        self._image_layer = None
        self._features_layer = None

        self.positions = []
        self.image_coords = []

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_load_image.clicked.connect(self.load_image)

        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])

        self.pushButton_move_to_position.clicked.connect(self._move_to_position)



    def run_tile_collection(self):

        print("run_tile_collection")

        beam_type = BeamType[self.comboBox_tile_beam_type.currentText()]
        grid_size = self.doubleSpinBox_tile_grid_size.value() * constants.MICRO_TO_SI
        tile_size = self.doubleSpinBox_tile_tile_size.value() * constants.MICRO_TO_SI

        self.settings.image.hfw = tile_size
        self.settings.image.beam_type = beam_type
        self.settings.image.save = True
        self.settings.image.save_path = PATH
        self.settings.image.resolution = [1024, 1024]
        self.settings.image.dwell_time = 1e-6
        self.settings.image.autocontrast = False
        self.settings.image.label = "stitched_image.tif"

        image = _tile._tile_image_collection_stitch(self.microscope, self.settings, grid_size, tile_size, overlap=0)

        self._update_viewer(image)


    def load_image(self):

        path = ui_utils._get_file_ui( msg="Select image to load", path=cfg.DATA_TILE_PATH, _filter="Image Files (*.tif *.tiff)", parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        image = FibsemImage.load(path)

        self._update_viewer(image)


    def _update_viewer(self, image: FibsemImage =  None):

        if image is not None:
            self.image = image
        
            arr = median_filter(self.image.data, size=3)
            self._image_layer = self.viewer.add_image(arr, name="tile", colormap="gray", blending="additive")

            # draw a point on the image at center
            ui_utils._draw_crosshair(viewer=self.viewer,eb_image= self.image, ib_image= self.image,is_checked=True) 

            # attached click callback to image
            self._image_layer.mouse_drag_callbacks.append(self._on_click)
        
        if self.image_coords:

            # TODO:probably a better way to do this
        

            text = {
                "string": [pos.name for pos in self.positions],
                "color": "white",
                "translation": np.array([-50, 0]),
            }
            if self._features_layer is None:
            
                 self._features_layer = self.viewer.add_points(
                    self.image_coords,
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
                self._features_layer.data = self.image_coords
                self._features_layer.text = text

        self.viewer.layers.selection.active = self._image_layer

    def get_data_from_coord(self, coords: tuple) -> tuple:
        # check inside image dimensions, (y, x)
        shape = self.image.data.shape[0], self.image.data.shape[1]

        if (coords[0] > 0 and coords[0] < shape[0]) and (
            coords[1] > 0 and coords[1] < shape[1]
        ):
            return True
        else:
            return False


    def _on_click(self, layer, event):
        
        if event.button != 1:
            return

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

        dx = point.x
        point_yz = self.microscope._y_corrected_stage_movement(self.settings, point.y, self.image.metadata.image_settings.beam_type)
        dy, dz = point_yz.y, point_yz.z

        _base_position = self.image.metadata.microscope_state.absolute_position

        # calculate the corrected move to reach that point from base-state?
        _new_position = deepcopy(_base_position)
        _new_position.x += dx
        _new_position.y += dy
        _new_position.z += dz
        _new_position.name = f"Position {len(self.positions):02d}"

        logging.info(f"BASE: {_base_position}")
        logging.info(f"POINT: {point}")
        logging.info(f"dx: {dx*1e6}, dy: {dy*1e6}, dz: {dz*1e6}")
        logging.info(f"NEW POSITION: {_new_position}")

        # now should be able to just move to _new_position
        self.positions.append(_new_position)

        # draw the point on the image
        self.image_coords.append(coords)

        # we could save this position as well, use it to pre-select a bunch of lamella positions?

        self._update_position_info()
        self._update_viewer()

    def _update_position_info(self):

        self.comboBox_tile_position.clear()
        self.comboBox_tile_position.addItems([pos.name for pos in self.positions])

        msg = ""
        for pos in self.positions:
            msg += f"{pos.name} - x:{pos.x*1e3:.3f}mm, y:{pos.y*1e3:.3f}mm, z:{pos.z*1e3:.3f}m\n"
        self.label_position_info.setText(msg)


    def _move_to_position(self):

        print("move_to_position")
        _position = self.positions[self.comboBox_tile_position.currentIndex()]
        logging.info(f"Moving To: {_position}")
        self.microscope._safe_absolute_stage_movement(_position)

        self._position_changed.emit()



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