import os
from copy import deepcopy

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets

from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    FibsemImage,
    Point,
)
from fibsem import utils
from fibsem.ui import _stylesheets
from PyQt5.QtCore import pyqtSignal
from fibsem.ui.qtdesigner_files import FibsemEmbeddedDetectionWidget
import logging

from fibsem.segmentation.config import CLASS_COLORS

class FibsemEmbeddedDetectionUI(FibsemEmbeddedDetectionWidget.Ui_Form, QtWidgets.QWidget):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        viewer: napari.Viewer,
        det: DetectedFeatures = None,
        model: fibsem_model.SegmentationModel = None,
        parent=None,
    ):
        super(FibsemEmbeddedDetectionUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.parent = parent
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)
        self.model = model

        self._USER_CORRECTED = False
        self._LABELLING_ENABLED = False
        self._MODEL_ASSIST = False

        self._image_layer = None
        self._mask_layer = None
        self._features_layer = None
        self._cross_hair_layer = None

        self.setup_connections()

        # set detected features
        if det is not None:
            self.set_detected_features(det)

        if model is not None:
            self._set_model(model)

    def setup_connections(self):
        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press Continue when finished."""
        )
        self.pushButton_continue.clicked.connect(self.confirm_button_clicked)
        self.pushButton_continue.setVisible(False)


        # labelling
        self.viewer.bind_key("L", self._toggle_labelling, overwrite=True)
        self.pushButton_enable_labelling.clicked.connect(self._toggle_labelling)
        self.checkBox_labelling_model_assist.stateChanged.connect(self._toggle_labelling)
        self.pushButton_enable_labelling.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)  

        self.checkBox_labelling_model_assist.setVisible(False) # TODO: add model assist
        self.pushButton_labelling_confirm.setVisible(self._MODEL_ASSIST)
        self.pushButton_labelling_cancel.setVisible(self._MODEL_ASSIST)       
        self.label_labelling_class_index.setVisible(self._MODEL_ASSIST)
        self.label_labelling_instructions.setVisible(self._MODEL_ASSIST)
        self.label_labelling_model.setVisible(self._MODEL_ASSIST)
        self.comboBox_labelling_class_index.setVisible(self._MODEL_ASSIST)


    def _toggle_labelling(self, event=None):
        """Toggle labelling mode on/off"""
        if self.sender() != self.checkBox_labelling_model_assist:
            self._LABELLING_ENABLED = not self._LABELLING_ENABLED

        self.checkBox_labelling_model_assist.setVisible(False) # TODO: add model assist
        # self._MODEL_ASSIST = self.checkBox_labelling_model_assist.isChecked()
        self._MODEL_ASSIST = False

        self.pushButton_labelling_confirm.setVisible(self._MODEL_ASSIST)
        self.pushButton_labelling_cancel.setVisible(self._MODEL_ASSIST)       
        self.label_labelling_class_index.setVisible(self._MODEL_ASSIST)
        self.label_labelling_instructions.setVisible(self._MODEL_ASSIST)
        self.label_labelling_model.setVisible(self._MODEL_ASSIST)
        self.comboBox_labelling_class_index.setVisible(self._MODEL_ASSIST)

        # show layer controls
        self.viewer.window._qt_viewer.dockLayerList.setVisible(self._LABELLING_ENABLED)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(self._LABELLING_ENABLED)

        if self._LABELLING_ENABLED:
            self.viewer.layers.selection.active = self._mask_layer
            self._mask_layer.mode = "paint"
            self.pushButton_enable_labelling.setText("Disable Labelling")
            self.pushButton_enable_labelling.setStyleSheet(_stylesheets._ORANGE_PUSHBUTTON_STYLE)        
        else:
            self.viewer.layers.selection.active = self._features_layer
            self._features_layer.mode = "select"
            self.pushButton_enable_labelling.setText("Enable Labelling")
            self.pushButton_enable_labelling.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)  

    def confirm_button_clicked(self, reset_camera=False):
        """Confirm the detected features, save the data and and remove the layers from the viewer."""
        
        # log the difference between initial and final detections
        try:
            fname = self.det.fibsem_image.metadata.image_settings.filename
            beam_type = self.det.fibsem_image.metadata.image_settings.beam_type
        except:
            fname = f"ml-{utils.current_timestamp_v2()}"
            beam_type = "NULL"
        
        fd = [] # feature detections
        for f0, f1 in zip(self.det.features, self._intial_det.features):
            px_diff = f1.px - f0.px
            msgd = {"msg": "feature_detection",
                    "fname": fname,                                             # filename
                    "feature": f0.name,                                         # feature name
                    "px": f0.px.to_dict(),                                      # pixel coordinates
                    "dpx": px_diff.to_dict(),                                   # pixel difference
                    "dm": px_diff._to_metres(self.det.pixelsize).to_dict(),     # metre difference
                    "is_correct": not np.any(px_diff),                          # is the feature correct    
                    "beam_type": beam_type.name,                                # beam type         
                    "pixelsize": self.det.pixelsize,                            # pixelsize
                    "checkpoint": self.det.checkpoint,                          # checkpoint
            }
            logging.debug(msgd)
            fd.append(deepcopy(msgd))                                           # to write to disk

        # save features data
        self.det.mask = self._mask_layer.data.astype(np.uint8) # type: ignore
        det_utils.save_feature_data_to_csv(self.det, features=fd, filename=fname)
            
        # remove feature detection layers
        if self._image_layer is not None:
            if self._image_layer in self.viewer.layers:
                self.viewer.layers.remove(self._image_layer)
            if self._mask_layer in self.viewer.layers:
                self.viewer.layers.remove(self._mask_layer)
            if self._features_layer in self.viewer.layers:
                self.viewer.layers.remove(self._features_layer)
            if self._cross_hair_layer in self.viewer.layers:
                self.viewer.layers.remove(self._cross_hair_layer)

        # reshow all other layers
        for layer in self.viewer.layers:
            layer.visible = True
        
        # reset camera
        self.viewer.camera.center = self.prev_camera.center
        self.viewer.camera.zoom = self.prev_camera.zoom

    def set_detected_features(self, det_features: DetectedFeatures):
        """Set the detected features and update the UI"""
        self.det = det_features
        self._intial_det = deepcopy(det_features)
        self._USER_CORRECTED = False

        self.update_features_ui()

    def update_features_ui(self):
        """Update the UI with the detected features"""
        # hide all other layers?
        for layer in self.viewer.layers:
            layer.visible = False

        self._image_layer = self.viewer.add_image(
            self.det.image, name="image", opacity=0.7, blending="additive",
        )

        # add mask to viewer
        self._mask_layer = self.viewer.add_labels(self.det.mask, 
                                                    name="mask", 
                                                    opacity=0.7,
                                                    blending="additive", 
                                                    color=CLASS_COLORS)

        # add points to viewer
        data = []
        for feature in self.det.features:
            x, y = feature.px
            data.append([y, x])

        text = {
            "string": [feature.name for feature in self.det.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self._features_layer = self.viewer.add_points(
            data,
            name="features",
            text=text,
            size=20,
            edge_width=7,
            edge_width_is_relative=False,
            edge_color="transparent",
            face_color=[feature.color for feature in self.det.features],
            blending="translucent",
        )

        # draw cross hairs
        self._cross_hair_layer = None
        self._draw_crosshairs()

        # set points layer to select mode and active
        self.viewer.layers.selection.active = self._features_layer
        self._features_layer.mode = "select"
        
        # when the point is moved update the feature
        self._features_layer.events.data.connect(self.update_point)
        self.update_info()
            
        # set camera
        self.prev_camera = deepcopy(self.viewer.camera)
        self.viewer.camera.center = [
            0.0,
            self._image_layer.data.shape[0] / 2,
            self._image_layer.data.shape[1] / 2,
        ]
        self.viewer.camera.zoom = 0.7

        if self.det.checkpoint:
            self.label_model.setText(f"Checkpont: {os.path.basename(self.det.checkpoint)}")
        
        napari.utils.notifications.show_info(f"Features ({', '.join([f.name for f in self.det.features])}) Detected")

    def update_info(self):
        """Update the info label with the feature information"""
        if len(self.det.features) > 2:
            self.label_info.setText("Info not available.")
            return
        
        if len(self.det.features) == 1:
            self.label_info.setText(
            f"""{self.det.features[0].name}: {self.det.features[0].px}
            User Corrected: {self._USER_CORRECTED}
            """)
            return
        if len(self.det.features) == 2:
            self.label_info.setText(
                f"""Moving 
                {self.det.features[0].name}: {self.det.features[0].px}
                to 
                {self.det.features[1].name}: {self.det.features[1].px}
                dx={self.det.distance.x*1e6:.2f}um, dy={self.det.distance.y*1e6:.2f}um
                User Corrected: {self._USER_CORRECTED}
                """
                )
            return

    def update_point(self, event):
        """Update the feature when the point is moved"""
        logging.debug(f"{event.source.name} changed its data!")

        layer = self.viewer.layers[f"{event.source.name}"]  # type: ignore

        # get the data
        data = layer.data

        # get which point was moved
        index: list[int] = list(layer.selected_data)  
                
        if len(data) != len(self.det.features):
            # loop backwards to remove the features
            for idx in index[::-1]:
                logging.debug({"msg": "detection_point_deleted",
                               "idx": idx, "data": data[idx], 
                               "feature": self.det.features[idx].name})
                self.det.features.pop(idx)

        else: 
            for idx in index:
                logging.debug({"msg": "detection_point_moved", 
                               "idx": idx, "data": data[idx], 
                               "feature": self.det.features[idx].name})
                
                # update the feature
                self.det.features[idx].px = Point(
                    x=data[idx][1], y=data[idx][0]
                )

        self._draw_crosshairs()
        self._USER_CORRECTED = True
        self.update_info()


    def _draw_crosshairs(self):
        """Draw crosshairs on the image"""

        data = self._features_layer.data

        # for each data point draw two lines from the edge of the image to the point
        line_data, line_colors = [], []
        for idx, point in enumerate(data):
            y, x = point # already flipped
            vline = [[y, 0], [y, self.det.image.data.shape[1]]]
            hline = [[0, x], [self.det.image.data.shape[0], x]]
            
            line_data += [hline, vline]
            color = self.det.features[idx].color
            line_colors += [color, color]
        try:
            self._cross_hair_layer.data = line_data
            self._cross_hair_layer.edge_color = line_colors
        except:
            self._cross_hair_layer = self.viewer.add_shapes(
                data=line_data,
                shape_type="line",
                edge_width=3,
                edge_color=line_colors,
                name="feature_cross_hair",
                opacity=0.7,
                blending="additive",
            )    
    
    def _set_model(self, model: fibsem_model.SegmentationModel):
        self.model = model
        # update model info
        self.label_model.setText(f"Model: {self.model.model.name} \nCheckpont: {os.path.basename(self.model.checkpoint)}")

    def _get_detected_features(self):

        from fibsem import conversions

        for feature in self.det.features:
            feature.feature_m = conversions.image_to_microscope_image_coordinates(
                feature.px, self.det.image.data, self.det.pixelsize
            )

        logging.debug({"msg": "get_detected_features", "detected_features": self.det.to_dict()})

        return self.det


from fibsem.detection.detection import Feature, DetectedFeatures
import fibsem

def main():
    # load model
    checkpoint = "autolamella-mega-20240107.pt"
    model = load_model(checkpoint=checkpoint)
    
    # load image
    image = FibsemImage.load(os.path.join(os.path.dirname(detection.__file__), "test_image.tif"))

    pixelsize = image.metadata.pixel_size.x if image.metadata is not None else 25e-9

    # detect features
    features = [detection.NeedleTip(), detection.LamellaCentre()]
    det = detection.detect_features(
        deepcopy(image.data), model, features=features, pixelsize=pixelsize
    )

    viewer = napari.Viewer(ndisplay=2)
    det_widget_ui = FibsemEmbeddedDetectionUI(
        viewer=viewer, 
        model=model,
        )
    
    det_widget_ui.set_detected_features(det)

    viewer.window.add_dock_widget(
        det_widget_ui, area="right", 
        add_vertical_stretch=False, 
        name=f"OpenFIBSEMv{fibsem.__version__} Feature Detection"
    )
    napari.run()

    det = det_widget_ui.det


if __name__ == "__main__":
    main()