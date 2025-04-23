import logging
import os
from copy import deepcopy
from typing import List, Optional

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Labels as NapariLabelsLayer
from napari.layers import Points as NapariPointsLayer
from napari.layers import Shapes as NapariShapesLayer
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

import fibsem
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation.config import CLASS_COLORS
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    FibsemImage,
    Point,
)
from fibsem.ui.qtdesigner_files import (
    FibsemEmbeddedDetectionWidget as FibsemEmbeddedDetectionWidgetUI,
)


class FibsemEmbeddedDetectionUI(FibsemEmbeddedDetectionWidgetUI.Ui_Form, QtWidgets.QWidget):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.parent = parent
        if parent is not None:
            viewer = parent.viewer
        else:
            viewer = napari.current_viewer()
        self.viewer: napari.Viewer = viewer

        self.has_user_corrected: bool = False

        self.image_layer: NapariImageLayer = None
        self.mask_layer: NapariLabelsLayer = None
        self.features_layer: NapariPointsLayer = None
        self.cross_hair_layer: NapariShapesLayer = None

        self.setup_connections()

    def setup_connections(self):
        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press Continue when finished."""
        )

    def confirm_button_clicked(self):
        """Confirm the detected features, save the data and and remove the layers from the viewer."""
        
        # update the mask as the user may edit it
        self.det.mask = self.mask_layer.data.astype(np.uint8) # type: ignore
        
        # log the difference between initial and final detections
        # TODO: move this to outside the widget, into the same place as the non-supervised logging.
        det_utils.save_ml_feature_data(det=self.det, 
                                       initial_features=self._intial_det.features)
            
        # remove feature detection layers
        if self.image_layer is not None:
            if self.image_layer in self.viewer.layers:
                self.viewer.layers.remove(self.image_layer)
            if self.mask_layer in self.viewer.layers:
                self.viewer.layers.remove(self.mask_layer)
            if self.features_layer in self.viewer.layers:
                self.viewer.layers.remove(self.features_layer)
            if self.cross_hair_layer in self.viewer.layers:
                self.viewer.layers.remove(self.cross_hair_layer)

        # reshow all other layers
        excluded_layers = ["alignment_area"]
        for layer in self.viewer.layers:
            if layer.name in excluded_layers:
                continue
            layer.visible = True
        
        # reset camera
        self.viewer.reset_view()

    def set_detected_features(self, det_features: DetectedFeatures):
        """Set the detected features and update the UI"""
        self.det = det_features
        self._intial_det = deepcopy(det_features)
        self.has_user_corrected = False

        self.update_features_ui()

    def update_features_ui(self):
        """Update the UI with the detected features"""
        # hide all other layers?
        for layer in self.viewer.layers:
            layer.visible = False

        self.image_layer = self.viewer.add_image(
            self.det.image, name="image", opacity=0.7, blending="additive",
        )

        # add mask to viewer
        self.mask_layer = self.viewer.add_labels(self.det.mask, 
                                                    name="mask", 
                                                    opacity=0.3,
                                                    blending="additive", 
                                                    )
        if hasattr(self.mask_layer, "colormap"):
            self.mask_layer.colormap = CLASS_COLORS
        else:
            self.mask_layer.color = CLASS_COLORS

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

        self.features_layer = self.viewer.add_points(
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
        self.cross_hair_layer = None
        self.draw_feature_crosshairs()

        # set points layer to select mode and active
        self.viewer.layers.selection.active = self.features_layer
        self.features_layer.mode = "select"
        
        # when the point is moved update the feature
        self.features_layer.events.data.connect(self.update_point)
        self.update_info()
            
        # set camera
        self.viewer.camera.center = [
            0.0,
            self.image_layer.data.shape[0] / 2,
            self.image_layer.data.shape[1] / 2,
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
            User Corrected: {self.has_user_corrected}
            """)
            return
        if len(self.det.features) == 2:
            self.label_info.setText(
                f"""Moving 
                {self.det.features[0].name}: {self.det.features[0].px}
                to 
                {self.det.features[1].name}: {self.det.features[1].px}
                dx={self.det.distance.x*1e6:.2f}um, dy={self.det.distance.y*1e6:.2f}um
                User Corrected: {self.has_user_corrected}
                """
                )
            return

    def update_point(self, event):
        """Update the feature when the point is moved"""
        # TODO: events have been updated so we can tell which point was deleted/moved/etc. Update this function to use that.
        logging.debug(f"{event.source.name} changed its data!")

        layer = self.viewer.layers[f"{event.source.name}"]  # type: ignore

        # get the data
        data = layer.data

        # get which point was moved
        index: List[int] = list(layer.selected_data)  
                
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

        self.draw_feature_crosshairs()
        self.has_user_corrected = True
        self.update_info()

    def draw_feature_crosshairs(self):
        """Draw crosshairs for each feature on the image"""

        data = self.features_layer.data

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
            self.cross_hair_layer.data = line_data
            self.cross_hair_layer.edge_color = line_colors
        except Exception as e:
            self.cross_hair_layer = self.viewer.add_shapes(
                data=line_data,
                shape_type="line",
                edge_width=3,
                edge_color=line_colors,
                name="feature_cross_hair",
                opacity=0.7,
                blending="additive",
            )    
    
    def _get_detected_features(self):

        from fibsem import conversions

        for feature in self.det.features:
            feature.feature_m = conversions.image_to_microscope_image_coordinates(
                feature.px, self.det.image.data, self.det.pixelsize
            )

        logging.debug({"msg": "get_detected_features", "detected_features": self.det.to_dict()})

        return self.det


def main():
    # load model
    checkpoint = "autolamella-mega-20240107.pt"
    model = load_model(checkpoint=checkpoint)
    
    # load image
    image = FibsemImage.load(os.path.join(os.path.dirname(detection.__file__), "test_image_2.tif"))

    pixelsize = image.metadata.pixel_size.x if image.metadata is not None else 25e-9

    # detect features
    features = [detection.LamellaRightEdge(), detection.LandingPost()]
    det = detection.detect_features(
        deepcopy(image.data), model, features=features, pixelsize=pixelsize
    )

    viewer = napari.Viewer(ndisplay=2)
    det_widget_ui = FibsemEmbeddedDetectionUI()
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