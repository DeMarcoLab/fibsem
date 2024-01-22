import logging
import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils, acquire
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, BeamType, FibsemImage, Point, FibsemStagePosition
from fibsem.ui.qtdesigner_files import FibsemMinimapWidget
import os
from fibsem import patterning
from napari.qt.threading import thread_worker
from fibsem.ui import _stylesheets
from fibsem.imaging import _tile
from scipy.ndimage import median_filter,rotate

from copy import deepcopy

PATH = os.path.join(cfg.DATA_PATH, "tile")
os.makedirs(PATH, exist_ok=True)
from fibsem.ui import utils as ui_utils 

from PyQt5.QtCore import pyqtSignal, pyqtSlot


class FibsemMinimapWidget(FibsemMinimapWidget.Ui_MainWindow, QtWidgets.QMainWindow):
    _stage_position_moved = pyqtSignal(FibsemStagePosition)
    _stage_position_added = pyqtSignal(FibsemStagePosition)
    _update_tile_collection = pyqtSignal(dict)
    _minimap_positions = pyqtSignal(list)
    
    def __init__(
        self,
        microscope: FibsemMicroscope,
        settings: MicroscopeSettings,
        viewer: napari.Viewer,
        parent=None,
    ):
        super(FibsemMinimapWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings

        self.parent = parent

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.image = None
        self._image_layer = None
        self._reprojection_layer = None
        self._corr_image_layers = []

        self.positions = []
        self._correlation = {}

        self._tile_info = {}

        self.setup_connections()

        self._update_ui()

    def setup_connections(self):

        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_run_tile_collection.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.actionLoad_Image.triggered.connect(self.load_image)

        self.comboBox_tile_beam_type.addItems([beam_type.name for beam_type in BeamType])

        self.pushButton_move_to_position.clicked.connect(self._move_position_pressed)

        self.comboBox_tile_position.currentIndexChanged.connect(self._update_current_position_info)
        self.pushButton_update_position.clicked.connect(self._update_position_pressed)
        self.pushButton_update_position.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        self.pushButton_remove_position.clicked.connect(self._remove_position_pressed)
        self.pushButton_remove_position.setStyleSheet(_stylesheets._RED_PUSHBUTTON_STYLE)

        self.actionSave_Positions.triggered.connect(self._save_positions_pressed)
        self.actionLoad_Positions.triggered.connect(self._load_positions)

        # checkbox
        self.checkBox_options_move_with_translation.stateChanged.connect(self._update_ui)
        self.checkBox_options_move_with_translation.setVisible(cfg._MINIMAP_MOVE_WITH_TRANSLATION)
        self.label_options_move_with_translate_info.setVisible(cfg._MINIMAP_MOVE_WITH_TRANSLATION)
        self.checkBox_options_acquire_after_movement.setVisible(cfg._MINIMAP_ACQUIRE_AFTER_MOVEMENT)
        self.label_options_header.setVisible(cfg._MINIMAP_ACQUIRE_AFTER_MOVEMENT and cfg._MINIMAP_MOVE_WITH_TRANSLATION)

        # signals
        # self._stage_position_added.connect(self._position_added_callback)
        self._update_tile_collection.connect(self._update_tile_collection_callback)
        self.parent._minimap_signal.connect(self.update_positions_from_parent)


        # pattern overlay
        milling_protocol = self.settings.protocol["milling"]
        milling_patterns = [k for k in milling_protocol if "stages" in milling_protocol[k] or "type" in milling_protocol[k]]
        self.comboBox_pattern_overlay.addItems(milling_patterns)
        if "trench" in milling_patterns:
            self.comboBox_pattern_overlay.setCurrentText("trench")
        elif "lamella" in milling_patterns:
            self.comboBox_pattern_overlay.setCurrentText("lamella")
        self.comboBox_pattern_overlay.currentIndexChanged.connect(self._update_pattern_overlay)
        self.checkBox_pattern_overlay.stateChanged.connect(self._update_pattern_overlay)

        # correlation
        self.actionLoad_Correlation_Image.triggered.connect(self._load_correlation_image)
        self.pushButton_update_correlation_image.clicked.connect(lambda: self._update_correlation_image(None))
        self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self._update_correlation_ui)

        # auto update correlation image
        self.doubleSpinBox_correlation_translation_x.valueChanged.connect(self._update_correlation_data)
        self.doubleSpinBox_correlation_translation_y.valueChanged.connect(self._update_correlation_data)
        self.doubleSpinBox_correlation_scale_x.valueChanged.connect(self._update_correlation_data) 
        self.doubleSpinBox_correlation_scale_y.valueChanged.connect(self._update_correlation_data)
        self.doubleSpinBox_correlation_rotation.valueChanged.connect(self._update_correlation_data)

        self.doubleSpinBox_correlation_translation_x.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_translation_y.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_scale_x.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_scale_y.setKeyboardTracking(False)
        self.doubleSpinBox_correlation_rotation.setKeyboardTracking(False)

        self.lineEdit_tile_path.setText(self.settings.image.path)

        # gridbar

        self.checkBox_gridbar.stateChanged.connect(self._gridbar_set)
        self.label_gb_spacing.setVisible(False)
        self.label_gb_width.setVisible(False)
        self.doubleSpinBox_gb_spacing.setVisible(False)
        self.doubleSpinBox_gb_width.setVisible(False)

        self.doubleSpinBox_gb_spacing.valueChanged.connect(self._update_gridbar)
        self.doubleSpinBox_gb_width.valueChanged.connect(self._update_gridbar)
        self.doubleSpinBox_gb_spacing.setKeyboardTracking(False)
        self.doubleSpinBox_gb_width.setKeyboardTracking(False)


    def update_positions_from_parent(self, positions):
        
        if positions is not None:
            self.positions = positions

        self._update_position_info()
        self._update_viewer()


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

        self._tile_info["grid_size"] = grid_size
        self._tile_info["tile_size"] = tile_size
        self._tile_info["resolution"] = resolution
        self._tile_info["beam_type"] = beam_type
                
        self.settings.image.dwell_time = dwell_time 
        self.settings.image.hfw = tile_size
        self.settings.image.beam_type = beam_type
        self.settings.image.resolution = [resolution, resolution]
        self.settings.image.autocontrast = autocontrast
        self.settings.image.save = True
        self.settings.image.path = path
        self.settings.image.filename = filename

        if self.settings.image.filename == "":
            napari.utils.notifications.show_error(f"Please enter a filename for the image")
            return
        
        # ui feedback
        self.pushButton_run_tile_collection.setStyleSheet(_stylesheets._ORANGE_PUSHBUTTON_STYLE)
        self.pushButton_run_tile_collection.setText("Running Tile Collection...")

        worker = self._run_tile_collection_thread(
            microscope=self.microscope, settings=self.settings, 
            grid_size=grid_size, tile_size=tile_size, 
            overlap=0, cryo=cryo)

        worker.finished.connect(self._tile_collection_finished)
        worker.start()


    def _gridbar_set(self):

        if self.checkBox_gridbar.isChecked():

            grid_shape = self.image.data.shape
            arr = np.zeros(shape=grid_shape, dtype=np.uint8)

            pixelsize = self.image.metadata.pixel_size.x

            # create grid, grid bars thickness = 10px
            BAR_THICKNESS_PX = int(5 * constants.MICRO_TO_SI / pixelsize)
            BAR_SPACING_PX = int(50 * constants.MICRO_TO_SI / pixelsize)   
            for i in range(0, arr.shape[0], BAR_SPACING_PX ):
                arr[i:i+BAR_THICKNESS_PX, :] = 255
                arr[:, i:i+BAR_THICKNESS_PX] = 255

            gridbar_image = FibsemImage(data=arr)

            self._update_correlation_image(gridbar_image, gridbar=True)

            self.label_gb_spacing.setVisible(True)
            self.label_gb_width.setVisible(True)
            self.doubleSpinBox_gb_spacing.setVisible(True)
            self.doubleSpinBox_gb_width.setVisible(True)

            self.doubleSpinBox_gb_spacing.setValue(50)
            self.doubleSpinBox_gb_width.setValue(5)
            

        else:
            
            layer_to_remove = ""
            for layer in self.viewer.layers:
                if "gridbar" in layer.name:
                    layer_to_remove = layer.name
            
            self.label_gb_spacing.setVisible(False)
            self.label_gb_width.setVisible(False)
            self.doubleSpinBox_gb_spacing.setVisible(False)
            self.doubleSpinBox_gb_width.setVisible(False)

            self.viewer.layers.remove(layer_to_remove)
            self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
            self.comboBox_correlation_selected_layer.clear()
            self.comboBox_correlation_selected_layer.addItems([layer.name for layer in self.viewer.layers if "correlation-image" in layer.name ])
            self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self._update_correlation_ui)


    def _update_gridbar(self):

        pixel_size = self.image.metadata.pixel_size.x

        print(f'pixel size: {pixel_size}')


        BAR_THICKNESS_PX = int(self.doubleSpinBox_gb_width.value() * constants.MICRO_TO_SI / pixel_size)
        BAR_SPACING_PX = int(self.doubleSpinBox_gb_spacing.value() * constants.MICRO_TO_SI / pixel_size)

        gridbar_layer = ''

        for layer in self.viewer.layers:
            if "gridbar" in layer.name:
                gridbar_layer = layer.name
        
        grid_shape = self.viewer.layers[gridbar_layer].data.shape

        arr = np.zeros(shape=grid_shape, dtype=np.uint8)
        for i in range(0, arr.shape[0], BAR_SPACING_PX ):
            arr[i:i+BAR_THICKNESS_PX, :] = 255
            arr[:, i:i+BAR_THICKNESS_PX] = 255

        self.viewer.layers[gridbar_layer].data = arr




    def _tile_collection_finished(self):

        napari.utils.notifications.show_info(f"Tile collection finished.")
        self._update_viewer(self.image)
        self.pushButton_run_tile_collection.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_run_tile_collection.setText("Run Tile Collection")

    
    @thread_worker
    def _run_tile_collection_thread(self, microscope: FibsemMicroscope, settings:MicroscopeSettings, 
        grid_size: float, tile_size:float, overlap:float=0, cryo: bool=True):

        self.image = _tile._tile_image_collection_stitch(
            microscope=microscope, settings=settings, 
            grid_size=grid_size, tile_size=tile_size, 
            overlap=overlap, cryo=cryo, parent_ui=self)

    def _update_tile_collection_callback(self, ddict):
        
        # msg = f"{ddict['msg']} ({ddict['i']+1}/{ddict['n_rows']}, {ddict['j']+1}/{ddict['n_cols']})"
        msg = f"{ddict['msg']} ({ddict['counter']}/{ddict['total']})"
        logging.info(msg)
        napari.utils.notifications.show_info(msg)

        if ddict["image"] is not None:

            arr = median_filter(ddict["image"], size=3)
            try:
                self._image_layer.data = arr
            except:
                self._image_layer = self.viewer.add_image(arr, name="overview-image", colormap="gray", blending="additive")

    def load_image(self):

        path = ui_utils._get_file_ui( msg="Select image to load", path=cfg.DATA_TILE_PATH, _filter="Image Files (*.tif *.tiff)", parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        image = FibsemImage.load(path)

        self._tile_info["tile_size"] = image.metadata.image_settings.hfw
        self._tile_info["resolution"] = image.metadata.image_settings.resolution[0]
        self._tile_info["beam_type"] = image.metadata.image_settings.beam_type

        self._update_viewer(image)

    def _update_ui(self):

        _image_loaded = self.image is not None
        self.pushButton_update_position.setEnabled(_image_loaded)
        self.pushButton_remove_position.setEnabled(_image_loaded)
        self.pushButton_move_to_position.setEnabled(_image_loaded)

        if _image_loaded:
            self.label_instructions.setText("Alt+Click to Add a position, Shift+Click to Update a position \nor Double Click to Move the Stage...")
        else:
            self.label_instructions.setText("Please take or load an overview image...")

        _positions_loaded = len(self.positions) > 0
        self.pushButton_move_to_position.setEnabled(_positions_loaded)

        _MOVE_WITH_TRANSLATION = self.checkBox_options_move_with_translation.isChecked()
        self.label_translation_x.setVisible(_MOVE_WITH_TRANSLATION)
        self.label_translation_y.setVisible(_MOVE_WITH_TRANSLATION)
        self.label_translation_z.setVisible(_MOVE_WITH_TRANSLATION)
        self.doubleSpinBox_translation_x.setVisible(_MOVE_WITH_TRANSLATION)
        self.doubleSpinBox_translation_y.setVisible(_MOVE_WITH_TRANSLATION)
        self.doubleSpinBox_translation_z.setVisible(_MOVE_WITH_TRANSLATION)

    def _update_viewer(self, image: FibsemImage =  None):

        if image is not None:
            self.image = image
        
            arr = median_filter(self.image.data, size=3)
            try:
                self._image_layer.data = arr
            except:
                self._image_layer = self.viewer.add_image(arr, name="overview-image", colormap="gray", blending="additive")

            # draw a point on the image at center
            ui_utils._draw_crosshair(viewer=self.viewer,eb_image= self.image, ib_image= self.image,is_checked=True) 

            self._image_layer.mouse_drag_callbacks.clear()
            self._image_layer.mouse_double_click_callbacks.clear()
            self._image_layer.mouse_drag_callbacks.append(self._on_click)
            self._image_layer.mouse_double_click_callbacks.append(self._on_double_click)
    

        self._draw_positions() # draw the reprojected positions on the image

        self.viewer.layers.selection.active = self._image_layer

        self._update_ui()


    def _update_region(self, position:FibsemStagePosition):
    

        resolution = self._tile_info["resolution"]
        self.settings.image.resolution = [resolution, resolution]
        self.settings.image.beam_type = self._tile_info["beam_type"]
        self.settings.image.hfw = self._tile_info["tile_size"]
        self.settings.image.save = False

        # TODO: this assumes the image is taken with the same settings as the tile collection
        self.image = _tile._update_image_region(self.microscope, self.settings.image, self.image, position)

        self._update_viewer(self.image)

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
                f"Clicked outside image dimensions. Please click inside the image to move."
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
        self._update_position_info()
        self._update_viewer()


        self._minimap_positions.emit(self.positions)

    def _on_double_click(self, layer, event):
        
        if event.button !=1:
            return
        coords, point = self._validate_mouse_click(layer, event)

        if point is False: # clicked outside image
            return

        _new_position = self.microscope.project_stable_move( 
            dx=point.x, dy=point.y, 
            beam_type=self.image.metadata.image_settings.beam_type, 
            base_position=self.image.metadata.microscope_state.stage_position)   


        _MOVE_WITH_TRANSLATION = self.checkBox_options_move_with_translation.isChecked()
        if _MOVE_WITH_TRANSLATION:
            dx = self.doubleSpinBox_translation_x.value() * constants.MILLI_TO_SI
            dy = self.doubleSpinBox_translation_y.value() * constants.MILLI_TO_SI
            dz = self.doubleSpinBox_translation_z.value() * constants.MILLI_TO_SI
            translation = FibsemStagePosition(x=dx, y=dy, z=dz)
            logging.info(f"[NOT ENABLED] Moving to position with translation: {translation}")
            napari.utils.notifications.show_warning(f" [NOT ENABLED] Moving to position with translation: {translation}")

            if cfg._MINIMAP_MOVE_WITH_TRANSLATION:
                _new_position += translation

        self._move_to_position(_new_position)
        self._minimap_positions.emit(self.positions)

    def _update_current_position_info(self):

        idx = self.comboBox_tile_position.currentIndex()
        if idx != -1 and len(self.positions) > 0:
            self.lineEdit_tile_position_name.setText(self.positions[idx].name)
            self.pushButton_move_to_position.setText(f"Move to {self.positions[idx].name}")
        else:
            self.lineEdit_tile_position_name.setText("")
        

    def _update_position_info(self):
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
        self._minimap_positions.emit(self.positions)
        self._update_position_info()
        self._update_viewer()
        


    def _move_position_pressed(self):

        logging.info(f"Moving to position...")
        _position = self.positions[self.comboBox_tile_position.currentIndex()]
        logging.info(f"Moving To: {_position}")
        self._move_to_position(_position)

    def _move_to_position(self, _position:FibsemStagePosition)->None:
        self.microscope.safe_absolute_stage_movement(_position)
        self._stage_position_moved.emit(_position)
        if self.checkBox_options_acquire_after_movement.isChecked():
            self._update_region(_position)
        self._update_viewer()


    def _load_positions(self):

        logging.info(f"Loading Positions...")
        path = ui_utils._get_file_ui( msg="Select a position file to load", 
            path=self.settings.image.path, 
            _filter= "*yaml", 
            parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        pdict = utils.load_yaml(path)
        
        positions = [FibsemStagePosition.from_dict(p) for p in pdict]
        # self.positions = self.positions + positions # append? or overwrite
        # overwrite 
        self.positions = positions
        self._minimap_positions.emit(self.positions)

        self._update_position_info()
        self._update_viewer()

    def _save_positions_pressed(self):
        
        logging.info(f"Saving Positions...")
        
        path = ui_utils._get_save_file_ui(msg = "Select a file to save the positions to",
            path = self.settings.image.path,
            _filter= "*yaml",
            parent=self,
        )

        if path == "":
            napari.utils.notifications.show_info(f"No file selected, not saving.")
            return

        # save the positions
        pdict = [p.to_dict() for p in self.positions]
        utils.save_yaml(path, pdict)

        napari.utils.notifications.show_info(f"Saved positions to {path}")

    def _update_pattern_overlay(self):

        self._draw_positions()


    def _draw_positions(self):
        
        logging.info(f"Drawing Reprojected Positions...")
        current_position = deepcopy(self.microscope.get_stage_position())
        current_position.name = f"Current Position"

        if self.image:
            
            drawn_positions = self.positions + [current_position]
            points = _tile._reproject_positions(self.image, drawn_positions)

            data = []
            for pt in points:
                # reverse to list
                data.append([pt.y, pt.x])

            colors = ["lime"] * (len(drawn_positions)-1)
            colors.append("yellow")

            colors_rgba = [[0,1,0,1] for _ in range(len(drawn_positions)-1)]
            colors_rgba.append([1,1,0,1]) # yellow

            text = {
                "string": [pos.name for pos in drawn_positions],
                "color": colors, # TODO: separate colour for current position
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

            self._reprojection_layer.face_color= colors_rgba


            _SHOW_PATTERNS: bool = self.checkBox_pattern_overlay.isChecked()
            if _SHOW_PATTERNS: # TODO: this is very slow, need to speed up, too many pattern redraws
                points = [conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), self.image.data, self.image.metadata.pixel_size.x ) for coords in data[:-1]]
                pattern = self.comboBox_pattern_overlay.currentText() 
                
                milling_stages = [patterning.get_milling_stages(pattern, self.settings.protocol["milling"], Point(point.x, point.y))[0] for point in points]
                
                for stage, pos in zip(milling_stages, drawn_positions[:-1]):
                    stage.name = pos.name

                ui_utils._draw_patterns_in_napari(self.viewer, 
                    ib_image=self.image, 
                    eb_image=None, 
                    milling_stages = milling_stages)
            else:
                ui_utils._remove_all_layers(viewer=self.viewer)

        self.viewer.layers.selection.active = self._image_layer


    def _load_correlation_image(self):

        # load another image
        path = ui_utils._get_file_ui( msg="Select image to load", 
        path=cfg.DATA_TILE_PATH, _filter="Image Files (*.tif *.tiff)", parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        image = FibsemImage.load(path)    

        self._update_correlation_image(image)


    def _update_correlation_image(self, image: FibsemImage = None,gridbar:bool=False):

        if image is not None:
            
            _basename = f"correlation-image" if not gridbar else f"gridbar-image"
            idx = 1
            _name = f"{_basename}-{idx:02d}"
            while _name in self.viewer.layers:
                idx+=1
                _name = f"{_basename}-{idx:02d}"

            # if grid bar in _name, idx = 3
            if gridbar:
                idx = 3

            COLOURS = ["green", "cyan", "magenta", "red", "yellow"]

            self._corr_image_layers.append(
                self.viewer.add_image(image.data, 
                name=_name, colormap=COLOURS[idx%len(COLOURS)], 
                blending="translucent", opacity=0.2)
            )

            self._correlation[_name] = deepcopy([0, 0, 1.0, 1.0, 0])


            self.comboBox_correlation_selected_layer.currentIndexChanged.disconnect()
            idx = self.comboBox_correlation_selected_layer.currentIndex()
            self.comboBox_correlation_selected_layer.clear()
            self.comboBox_correlation_selected_layer.addItems([layer.name for layer in self.viewer.layers if "correlation-image" in layer.name or "gridbar-image" in layer.name])
            if idx != -1:
                self.comboBox_correlation_selected_layer.setCurrentIndex(idx)
            self.comboBox_correlation_selected_layer.currentIndexChanged.connect(self._update_correlation_ui)
            
            self.viewer.layers.selection.active = self._image_layer
    
    # do this when image selected is changed
    def _update_correlation_ui(self):

        # set ui
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        if layer_name == "":
            napari.utils.notifications.show_info(f"Please select a layer to correlate with  update data...")
            return

        tx, ty, sx, sy, r = self._correlation[layer_name]

        self.doubleSpinBox_correlation_translation_x.setValue(tx)
        self.doubleSpinBox_correlation_translation_y.setValue(ty)
        self.doubleSpinBox_correlation_scale_x.setValue(sx)
        self.doubleSpinBox_correlation_scale_y.setValue(sy)
        self.doubleSpinBox_correlation_rotation.setValue(r)

        self._update_correlation_data()

    # do this when parameters change
    def _update_correlation_data(self):

        # select layer
        layer_name = self.comboBox_correlation_selected_layer.currentText()
        if layer_name == "":
            napari.utils.notifications.show_info(f"Please select a layer to correlate with update ui...")
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


        self._correlation[layer_name] = deepcopy([tx, ty, sx, sy, r])

        # apply to selected layer
        self.viewer.layers[layer_name].translate = [corrected_ty, corrected_tx]
        self.viewer.layers[layer_name].scale = [sy, sx]
        self.viewer.layers[layer_name].rotate = r

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