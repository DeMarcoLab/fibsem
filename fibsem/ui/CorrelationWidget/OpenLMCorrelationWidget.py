import scipy.ndimage as ndi
import skimage
from skimage.transform import AffineTransform
import logging

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
import tifffile as tf

logging.basicConfig(level=logging.INFO)

import logging
from copy import deepcopy
import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
# from openlm.microscope import LightMicroscope

from fibsem.ui.CorrelationWidget.qt import OpenLMCorrelationWidget


class OpenLMCorrelationWidget(OpenLMCorrelationWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        viewer: napari.Viewer = None,
        microscope = None,
        parent=None,
    ):
        super(OpenLMCorrelationWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        self.viewer = viewer
        # TODO: add fibsem microscope
        # self.microscope = microscope
        self.points_1 = []
        self.points_2 = []

        # TODO: make better
        self.load_images()
        self.initialise_viewer()
        self.setup_connections()

    def setup_connections(self):
        self.pushButton_correlate.clicked.connect(self.correlate)

    def correlate(self):
        # Need at least 3 points to correlate
        if not (len(self.points_1) > 2 and len(self.points_2) > 2):
            return

        # Check if points are in the same order
        if not (len(self.points_1) == len(self.points_2)):
            return

        print("Correlating")
        matched_points = self.convert_points_to_matched_arrays()
        self.blended_image = correlate_images(
            self.image_1, self.image_2, matched_points
        )
        if "blended_image" in self.viewer.layers:
            self.viewer.layers["blended_image"].data = self.blended_image
        else:
            self.viewer.add_image(
                self.blended_image,
                name="blended_image",
                colormap="gray",
                visible=True,
                translate=[
                    0,
                    self.viewer.layers["image_1"].data.shape[1]
                    + self.viewer.layers["image_2"].data.shape[1],
                ],
            )

    def update_cpoints_table(self):
        n_points = np.max([len(self.points_1), len(self.points_2)])
        self.tableWidget_cpoints.setRowCount(0)

        for i in range(n_points):
            if i > len(self.points_1) - 1:
                x1 = np.NaN
                y1 = np.NaN
            else:
                x1 = self.points_1[i][1]
                y1 = self.points_1[i][0]

            if i > len(self.points_2) - 1:
                x2 = np.NaN
                y2 = np.NaN
            else:
                x2 = (
                    self.points_2[i][1]
                    - self.viewer.layers["image_2"].extent.world[0][1]
                )
                y2 = self.points_2[i][0]

            self.tableWidget_cpoints.insertRow(i)
            self.tableWidget_cpoints.setItem(i, 0, QTableWidgetItem(str(x1)))
            self.tableWidget_cpoints.setItem(i, 1, QTableWidgetItem(str(y1)))
            self.tableWidget_cpoints.setItem(i, 2, QTableWidgetItem(str(x2)))
            self.tableWidget_cpoints.setItem(i, 3, QTableWidgetItem(str(y2)))

    def convert_points_to_matched_arrays(self):
        points_1 = deepcopy(self.points_1)
        points_2 = deepcopy(self.points_2)

        if len(points_1) > len(points_2):
            points_1 = points_1[: len(points_2)]
        elif len(points_2) > len(points_1):
            points_2 = points_2[: len(points_1)]

        matched_points = np.zeros((len(points_1), 5))
        for i in range(len(points_1)):
            matched_points[i, 0] = i
            matched_points[i, 1] = points_1[i][1]
            matched_points[i, 2] = points_1[i][0]
            matched_points[i, 3] = (
                points_2[i][1] - self.viewer.layers["image_2"].extent.world[0][1]
            )
            matched_points[i, 4] = (
                points_2[i][0] - self.viewer.layers["image_2"].extent.world[0][0]
            )

        return matched_points

    def initialise_viewer(self):
        self.viewer.add_image(
            np.zeros(
                (
                    np.max([self.image_1.shape[1], self.image_2.shape[1]]),
                    np.max([self.image_1.shape[0], self.image_2.shape[0]]),
                )
            ),
            name="blended_image",
            colormap="gray",
            visible=True,
        )
        self.viewer.add_image(
            self.image_1, name="image_1", colormap="gray", visible=True
        )
        self.viewer.add_image(
            self.image_2,
            name="image_2",
            colormap="gray",
            visible=True,
            translate=[0, self.viewer.layers["image_1"].data.shape[1]],
        )
        self.viewer.layers["blended_image"].translate = [
            0,
            self.viewer.layers["image_1"].data.shape[1]
            + self.viewer.layers["image_2"].data.shape[1],
        ]
        self.setup_callbacks(self.viewer.layers["image_1"])
        self.setup_callbacks(self.viewer.layers["image_2"])

    def setup_callbacks(self, layer):
        layer.mouse_drag_callbacks.append(self._drag_check)
        layer.events.data.connect(self._changed_data)
        layer.events.data.connect(self.correlate)

    def _changed_data(self):
        if "points_1" in self.viewer.layers:
            self.points_1 = list(self.viewer.layers["points_1"].data)
        if "points_2" in self.viewer.layers:
            self.points_2 = list(self.viewer.layers["points_2"].data)
        self.update_cpoints_table()

    def _check_cpoints_list(self):
        len_points_1 = len(self.points_1)
        len_points_2 = len(self.points_2)
        if len_points_1 == len_points_2:
            return True
        elif len_points_1 > len_points_2:
            return "points_2"
        elif len_points_1 < len_points_2:
            return "points_1"
        else:
            return None

    def _drag_check(self, layer, event):
        dragged = False
        yield
        # on move
        while event.type == "mouse_move":
            dragged = True
            yield
        # on release
        if dragged:
            return
        else:
            self._single_click(layer, event)
            return

    def _single_click(self, layer, event):
        # Use right clicking for adding points
        if event.button == 1:
            return

        # Check if the click is on a layer
        clicked_layer = self._check_layer_clicked(event)

        # Set properties based on layer click
        if clicked_layer == self.viewer.layers[
            "image_1"
        ] and self._check_cpoints_list() in ["points_1", True]:
            new_layer = "points_1"
            data = self.points_1
            color = "blue"
        elif clicked_layer == self.viewer.layers[
            "image_2"
        ] and self._check_cpoints_list() in ["points_2", True]:
            new_layer = "points_2"
            data = self.points_2
            color = "red"
        else:
            return

        # Add point to layer, or create layer if it doesn't exist
        if new_layer in self.viewer.layers:
            data.append(event.position)
            self.viewer.layers[new_layer].data = data
        else:
            self.viewer.add_points(
                event.position,
                name=new_layer,
                symbol="cross",
                size=20,
                face_color=color,
                edge_color=color,
            )
            data = [event.position]
            self.setup_callbacks(self.viewer.layers[new_layer])

        # Update the cpoints table
        self.update_cpoints_table()

    def _check_layer_clicked(self, event):
        extent_image_1 = self.viewer.layers["image_1"].extent.world
        extent_image_2 = self.viewer.layers["image_2"].extent.world

        image_1_x = [extent_image_1[0][1], extent_image_1[1][1]]
        image_1_y = [extent_image_1[0][0], extent_image_1[1][0]]

        image_2_x = [extent_image_2[0][1], extent_image_2[1][1]]
        image_2_y = [extent_image_2[0][0], extent_image_2[1][0]]

        click_x = event.position[1]
        click_y = event.position[0]

        click_within_image_1 = (click_x > image_1_x[0] and click_x < image_1_x[1]) and (
            click_y > image_1_y[0] and click_y < image_1_y[1]
        )
        click_within_image_2 = (click_x > image_2_x[0] and click_x < image_2_x[1]) and (
            click_y > image_2_y[0] and click_y < image_2_y[1]
        )

        if click_within_image_1:
            return self.viewer.layers["image_1"]
        if click_within_image_2:
            return self.viewer.layers["image_2"]
        return None

    def load_images(self):
        # TODO: make this dynamic
        """Forcefully load images from hard drive"""
        # self.image_1 = tf.imread(r"C:\Users\User\Downloads\pairsib.tif")
        # print(self.image_1.shape)
        # self.image_1 = tf.imread(r"Y:\Projects\piescope\piescope_dev\tile\2023-05-08-03-58-01-001792PM.tif")
        # self.image_1 = self.image_1[:, :, 0]

        # self.image_2 = tf.imread(r"C:\Users\User\Downloads\pairsbeforeeb.tif")
        # self.image_2 = tf.imread(r"Y:\Projects\piescope\piescope_dev\tile\2023-05-08-03-58-09-153392PM.tif")
        # print(self.image_2.shape)
        # self.image_2 = self.image_2[:, :, 0]

        # random image 2048x2048
        self.image_1 = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        self.image_2 = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)

def correlate_images(fluorescence_image_rgb, fibsem_image, matched_points_dict):
    """Correlates two images using points chosen by the user

    Parameters
    ----------
    fluorescence_image_rgb :
        umpy array with shape (cols, rows, channels)
    fibsem_image : AdornedImage.
        Expecting .data attribute of shape (cols, rows, channels)
    output : str
        Path to save location
    matched_points_dict : dict
    Dictionary of points selected in the correlation window
    """
    # if matched_points_dict = []:
    #     logging.error('No control points selected, exiting.')
    #     return

    src, dst = point_coords(matched_points_dict)
    transformation = calculate_transform(src, dst)
    fluorescence_image_aligned = apply_transform(fluorescence_image_rgb, transformation)
    result = overlay_images(fluorescence_image_aligned, fibsem_image.data)
    # result = skimage.util.img_as_ubyte(result)

    return result


def point_coords(matched_points_dict):
    """Create source & destination coordinate numpy arrays from cpselect dict.

    Matched points is an array where:
    * the number of rows is equal to the number of points selected.
    * the first column is the point index label.
    * the second and third columns are the source x, y coordinates.
    * the last two columns are the destination x, y coordinates.

    Parameters
    ----------
    matched_points_dict : dict
        Dictionary returned from cpselect containing matched point coordinates.

    Returns
    -------
    (src, dst)
        Row, column coordaintes of source and destination matched points.
        Tuple contains two N x 2 ndarrays, where N is the number of points.
    """

    # matched_points = np.array([list(point.values())
    #                            for point in matched_points_dict])
    matched_points = np.array(matched_points_dict)

    src = np.flip(matched_points[:, 1:3], axis=1)  # flip for row, column index
    dst = np.flip(matched_points[:, 3:], axis=1)  # flip for row, column index

    return src, dst


def calculate_transform(src, dst, model=AffineTransform()):
    """Calculate transformation matrix from matched coordinate pairs.

    Parameters
    ----------
    src : ndarray
        Matched row, column coordinates from source image.
    dst : ndarray
        Matched row, column coordinates from destination image.
    model : scikit-image transformation class, optional.
        By default, model=AffineTransform()


    Returns
    -------
    ndarray
        Transformation matrix.
    """

    model.estimate(src, dst)
    logging.info(f"Transformation matrix: {model.params}")

    return model.params


def apply_transform(image, transformation, inverse=True, multichannel=True):
    """Apply transformation to a 2D image.

    Parameters
    ----------
    image : ndarray
        Input image array. 2D grayscale image expected, or
        2D plus color channels if multichannel kwarg is set to True.
    transformation : ndarray
        Affine transformation matrix. 3 x 3 shape.
    inverse : bool, optional
        Inverse transformation, eg: aligning source image coords to destination
        By default `inverse=True`.
    multichannel : bool, optional
        Treat the last dimension as color, transform each color separately.
        By default `multichannel=True`.

    Returns
    -------
    ndarray
        Image warped by transformation matrix.
    """

    if inverse:
        transformation = np.linalg.inv(transformation)

    if not multichannel:
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        elif image.ndim != transformation.shape[0] - 1:
            raise ValueError(
                "Unexpected number of image dimensions for the "
                "input transformation. Did you need to use: "
                "multichannel=True ?"
            )

    # move channel axis to the front for easier iteration over array
    # image = np.moveaxis(image, -1, 0)
    # make iamge the right shape
    image = np.expand_dims(image, axis=0)
    warped_img = np.array(
        [ndi.affine_transform((img_channel), transformation) for img_channel in image]
    )
    warped_img = warped_img[0]
    # warped_img = np.moveaxis(warped_img, 0, -1)

    return warped_img


def overlay_images(fluorescence_image, fibsem_image, transparency=0.5):
    """Blend two RGB images together.

    Parameters
    ----------
    fluorescence_image : ndarray
        2D RGB image.
    fibsem_image : ndarray
        2D RGB image.
    transparency : float, optional
        Transparency alpha parameter between 0 - 1, by default 0.5

    Returns
    -------
    ndarray
        Blended 2D RGB image.
    """

    fluorescence_image = skimage.img_as_float(fluorescence_image)
    fibsem_image = skimage.img_as_float(fibsem_image)
    blended = (
        transparency * fluorescence_image + (1 - transparency) * fibsem_image# fibsem_image[:, :, 0]
    )
    blended = np.clip(blended, 0, 1)

    return blended


def main():
    viewer = napari.Viewer(ndisplay=2)
    image_settings_ui = OpenLMCorrelationWidget(viewer=viewer)
    viewer.window.add_dock_widget(
        image_settings_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
