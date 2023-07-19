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



class FibsemTileWidget(FibsemTileWidget.Ui_Form, QtWidgets.QWidget):
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

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_load_image.clicked.connect(self.load_image)

        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])


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

        ddict = _tile._tile_image_collection(self.microscope, self.settings, grid_size, tile_size)
        images = ddict["images"]
        big_image = ddict["big_image"]

        image = _tile._stitch_images(images=images, ddict=ddict, overlap=0)
        image.save(os.path.join(PATH, "stitched_image.tif"))

        self._update_viewer(image)


    def load_image(self):

        print("load_image")

        image = FibsemImage.load(os.path.join(PATH, "stitched_image.tif"))

        self._update_viewer(image)

    def _update_viewer(self, image: FibsemImage):

        self.image = image
        arr = median_filter(image.data, size=3)
        self.viewer.layers.clear()
        self._image_layer = self.viewer.add_image(image.data, name="tile", colormap="gray", blending="additive")

        # draw a point on the image at center
        ui_utils._draw_crosshair(viewer=self.viewer,eb_image= self.image, ib_image= self.image,is_checked=True) 


        # attached click callback to image
        self._image_layer.mouse_double_click_callbacks.append(self._on_click)


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


        logging.info(f"BASE: {_base_position}")
        logging.info(f"POINT: {point}")
        logging.info(f"dx: {dx*1e6}, dy: {dy*1e6}, dz: {dz*1e6}")
        logging.info(f"NEW POSITION: {_new_position}")

        # now should be able to just move to _new_position
        self.microscope._safe_absolute_stage_movement(_new_position)

        # we could save this position as well, use it to pre-select a bunch of lamella positions?


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