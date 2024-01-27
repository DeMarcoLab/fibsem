import logging
from dataclasses import dataclass
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fibsem import constants, conversions
from fibsem.structures import Point, FibsemImage, FibsemRectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QMessageBox, QSizePolicy, QVBoxLayout, QWidget
from fibsem.patterning import FibsemMillingStage
from fibsem.structures import (FibsemPatternSettings, 
                               FibsemRectangleSettings,  FibsemLineSettings, 
                               FibsemCircleSettings, FibsemBitmapSettings)
from fibsem.microscope import FibsemMicroscope
import napari
from fibsem.utils import load_yaml, save_yaml
import fibsem.patterning as patterning

# # TODO: clean up and refactor these (_WidgetPlot and _PlotCanvas)
# class _WidgetPlot(QWidget):
#     def __init__(self, *args, display_image, **kwargs):
#         QWidget.__init__(self, *args, **kwargs)
#         self.setLayout(QVBoxLayout())
#         self.canvas = _PlotCanvas(self, image=display_image)
#         self.layout().addWidget(self.canvas)


# class _PlotCanvas(FigureCanvasQTAgg):
#     def __init__(self, parent=None, image=None):
#         self.fig = Figure()
#         FigureCanvasQTAgg.__init__(self, self.fig)

#         self.setParent(parent)
#         FigureCanvasQTAgg.setSizePolicy(
#             self, QSizePolicy.Expanding, QSizePolicy.Expanding
#         )
#         FigureCanvasQTAgg.updateGeometry(self)
#         self.image = image
#         self.plot()
#         self.createConn()

#         self.figureActive = False
#         self.axesActive = None
#         self.cursorGUI = "arrow"
#         self.cursorChanged = False

#     def plot(self):
#         gs0 = self.fig.add_gridspec(1, 1)

#         self.ax11 = self.fig.add_subplot(gs0[0], xticks=[], yticks=[], title="")

#         if self.image.ndim == 3:
#             self.ax11.imshow(self.image,)
#         else:
#             self.ax11.imshow(self.image, cmap="gray")

#     def updateCanvas(self, event=None):
#         ax11_xlim = self.ax11.get_xlim()
#         ax11_xvis = ax11_xlim[1] - ax11_xlim[0]

#         while len(self.ax11.patches) > 0:
#             [p.remove() for p in self.ax11.patches]
#         while len(self.ax11.texts) > 0:
#             [t.remove() for t in self.ax11.texts]

#         ax11_units = ax11_xvis * 0.003
#         self.fig.canvas.draw()

#     def createConn(self):
#         self.fig.canvas.mpl_connect("figure_enter_event", self.activeFigure)
#         self.fig.canvas.mpl_connect("figure_leave_event", self.leftFigure)
#         self.fig.canvas.mpl_connect("button_press_event", self.mouseClicked)
#         self.ax11.callbacks.connect("xlim_changed", self.updateCanvas)

#     def activeFigure(self, event):
#         self.figureActive = True

#     def leftFigure(self, event):
#         self.figureActive = False
#         if self.cursorGUI != "arrow":
#             self.cursorGUI = "arrow"
#             self.cursorChanged = True

#     def mouseClicked(self, event):
#         x = event.xdata
#         y = event.ydata


# @dataclass
# class Crosshair:
#     rectangle_horizontal: plt.Rectangle
#     rectangle_vertical: plt.Rectangle


# # TODO update with Point
# def draw_crosshair(image, canvas, x: float = None, y: float = None, colour: str ="yellow"):
#     # draw crosshairs
#     crosshair = create_crosshair(image, x, y, colour=colour)
#     for patch in crosshair.__dataclass_fields__:
#         canvas.ax11.add_patch(getattr(crosshair, patch))
#         getattr(crosshair, patch).set_visible(True)

# # draw arrow
# def draw_arrow(p1: Point, p2: Point, canvas) -> None:
#     """Draw an arrow patch from p1 to p2"""
#     x1, y1 = p1.x, p1.y
#     x2, y2 = p2.x, p2.y
#     line = mpatches.Arrow(x1, y1, x2 - x1, y2 - y1, color="white")

#     # draw line
#     canvas.ax11.add_patch(line)
#     return


# def display_error_message(message, title="Error"):
#     """PyQt dialog box displaying an error message."""
#     logging.exception(message)

#     error_dialog = QMessageBox()
#     error_dialog.setIcon(QMessageBox.Critical)
#     error_dialog.setText(message)
#     error_dialog.setWindowTitle(title)
#     error_dialog.exec_()

#     return error_dialog


# def message_box_ui(title: str, text: str, buttons = QMessageBox.Yes | QMessageBox.No):

#     msg = QMessageBox()
#     msg.setWindowTitle(title)
#     msg.setText(text)
#     msg.setStandardButtons(buttons)
#     msg.exec_()

#     response = True if (msg.clickedButton() == msg.button(QMessageBox.Yes)) or (msg.clickedButton() == msg.button(QMessageBox.Ok) ) else False

#     return response


from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGridLayout, QLabel

COLOURS = ["yellow", "cyan", "magenta", "lime", "orange", "hotpink", "green", "blue", "red", "purple"]


def set_arr_as_qlabel(
    arr: np.ndarray,
    label: QLabel,
    shape: tuple = (1536 // 4, 1024 // 4),
) -> QLabel:
    image = QImage(
        arr.data,
        arr.shape[1],
        arr.shape[0],
        QImage.Format_Grayscale8,
    )
    label.setPixmap(QPixmap.fromImage(image).scaled(*shape))

    return label


# def set_arr_as_qlabel_8(
#     arr: np.ndarray,
#     label: QLabel,
#     shape: tuple = (1536//4, 1024//4),
# ) -> QLabel:

#     image = QImage(
#         arr.data,
#         arr.shape[1],
#         arr.shape[0],
#         QImage.Format_Grayscale8,
#     )
#     label.setPixmap(QPixmap.fromImage(image).scaled(*shape))

#     return label


def convert_pattern_to_napari_circle(
    pattern_settings: FibsemCircleSettings, image: FibsemImage
):
    
    if not isinstance(pattern_settings, FibsemCircleSettings):
        raise ValueError("Pattern is not a circle")
    
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y


    # pattern to pixel coords
    r = int(pattern_settings.radius / pixelsize_x)
    cx = int(icx + (pattern_settings.centre_x / pixelsize_y))
    cy = int(icy - (pattern_settings.centre_y / pixelsize_y))

    # create corner coords
    xmin, ymin = cx - r, cy - r
    xmax, ymax = cx + r, cy + r

    # create circle
    shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]  # ??
    return shape


def convert_pattern_to_napari_line(
    pattern_settings: FibsemLineSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y
    
    # extract pattern information from settings
    if not isinstance(pattern_settings, FibsemLineSettings):
        raise ValueError("Pattern is not a line")

    start_x = pattern_settings.start_x
    start_y = pattern_settings.start_y
    end_x = pattern_settings.end_x
    end_y = pattern_settings.end_y

    # pattern to pixel coords
    px0 = int(icx + (start_x / pixelsize_x))
    py0 = int(icy - (start_y / pixelsize_y))
    px1 = int(icx + (end_x / pixelsize_x))
    py1 = int(icy - (end_y / pixelsize_y))

    # napari shape format [[y_start, x_start], [y_end, x_end]])
    shape = [[py0, px0], [py1, px1]]
    return shape

def convert_pattern_to_napari_rect(
    pattern_settings: FibsemRectangleSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y
    # extract pattern information from settings
    if isinstance(pattern_settings, FibsemLineSettings):
        pattern_width = pattern_settings.end_x - pattern_settings.start_x
        pattern_height = max(pattern_settings.end_y - pattern_settings.start_y, 0.5e-6)
        pattern_rotation = np.arctan2(
            pattern_height, pattern_width
        )  # TODO: line rotation doesnt work correctly, fix
        pattern_centre_x = (pattern_settings.end_x + pattern_settings.start_x) / 2
        pattern_centre_y = (pattern_settings.end_y + pattern_settings.start_y) / 2

    
    elif isinstance(pattern_settings, FibsemCircleSettings): #only used for out of bounds check
        pattern_width = 2*pattern_settings.radius
        pattern_height = 2*pattern_settings.radius
        pattern_centre_x = pattern_settings.centre_x
        pattern_centre_y = pattern_settings.centre_y
        pattern_rotation = 0

    else:
        pattern_width = pattern_settings.width
        pattern_height = pattern_settings.height
        pattern_centre_x = pattern_settings.centre_x
        pattern_centre_y = pattern_settings.centre_y
        pattern_rotation = pattern_settings.rotation
    # pattern to pixel coords
    w = int(pattern_width / pixelsize_x)
    h = int(pattern_height / pixelsize_y)
    cx = int(icx + (pattern_centre_x / pixelsize_y))
    cy = int(icy - (pattern_centre_y / pixelsize_y))
    r = -pattern_rotation  #
    xmin, xmax = -w / 2, w / 2
    ymin, ymax = -h / 2, h / 2
    px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
    py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))
    px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
    py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))
    px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
    py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))
    px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
    py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))
    # napari shape format
    shape = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]
    return shape

def create_crosshair_shape(centre_point: Point, image: FibsemImage,eb_image: FibsemImage):

    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2

    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    pattern_centre_x = centre_point.x
    pattern_centre_y = centre_point.y

    cx = int(icx + (pattern_centre_x / pixelsize_y))
    cy = int(icy - (pattern_centre_y / pixelsize_y))

    r_angles = [0,np.deg2rad(90)] #
    w = 40
    h = 1
    crosshair_shapes = []

    for r in r_angles:
        xmin, xmax = -w / 2, w / 2
        ymin, ymax = -h / 2, h / 2
        px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
        py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))
        px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
        py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))
        px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
        py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))
        px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
        py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))
        # napari shape format
        shape = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]
        if eb_image is not None:
                for c in shape:
                    c[1] += eb_image.data.shape[1]
        crosshair_shapes.append(shape)

    return crosshair_shapes






def convert_bitmap_pattern_to_napari_image(
        pattern_settings: FibsemBitmapSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    resize_x = int(pattern_settings.width / pixelsize_x)
    resize_y = int(pattern_settings.height / pixelsize_y)

    
    image_bmp = Image.open(pattern_settings.path)
    image_resized = image_bmp.resize((resize_x, resize_y))
    image_rotated = image_resized.rotate(-pattern_settings.rotation, expand=True)
    img_array = np.array(image_rotated)

    pattern_centre_x = int(icx - pattern_settings.width/pixelsize_x/2) + image.data.shape[1] 
    pattern_centre_y = int(icy - pattern_settings.height/pixelsize_y/2)

    pattern_point_x = int(pattern_centre_x + pattern_settings.centre_x / pixelsize_x)
    pattern_point_y = int(pattern_centre_y - pattern_settings.centre_y / pixelsize_y)

    translate_position = (pattern_point_y,pattern_point_x)

    
    return img_array, translate_position

def convert_pattern_to_napari_image(pattern_settings: FibsemCircleSettings, image: FibsemImage) -> np.ndarray:
    """Convert a circle pattern to a napari image. Note: annulus can only be plotted as image"""
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    resize_x = int(2*pattern_settings.radius / pixelsize_x)
    resize_y = int(2*pattern_settings.radius / pixelsize_y)

    
    inner_radius_ratio = (pattern_settings.radius - pattern_settings.thickness)/pattern_settings.radius

    annulus_shape = _create_annulus_shape(width=resize_x, height=resize_y, inner_radius=inner_radius_ratio, outer_radius=1)
    # annulus_image = np.array(image_rotated)

    pattern_centre_x = int(icx - pattern_settings.radius/pixelsize_x) + image.data.shape[1] 
    pattern_centre_y = int(icy - pattern_settings.radius/pixelsize_y)

    pattern_point_x = int(pattern_centre_x + pattern_settings.centre_x / pixelsize_x)
    pattern_point_y = int(pattern_centre_y - pattern_settings.centre_y / pixelsize_y)

    translate_position = (pattern_point_y,pattern_point_x)

    return annulus_shape, translate_position
   
def _create_annulus_shape(width, height, inner_radius, outer_radius):
    # Create a grid of coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    # Generate the donut shape
    donut = np.logical_and(distance < outer_radius, distance > inner_radius).astype(int)
    return donut


def _remove_all_layers(viewer: napari.Viewer, layer_type = napari.layers.shapes.shapes.Shapes, _ignore: list[str] = []):

    # remove all shapes layers
    layers_to_remove = []
    layers_to_ignore = ["ruler_line","crosshair","scalebar","scalebar_value", "label"] + _ignore
    for layer in viewer.layers:

        if layer.name in layers_to_ignore:
            continue
        if isinstance(layer, layer_type) or layer.name in ["bmp_Image","annulus_Image"]:
            layers_to_remove.append(layer)
    for layer in layers_to_remove:
        viewer.layers.remove(layer)  # Not removing the second layer?


def _draw_patterns_in_napari(
    viewer: napari.Viewer,
    ib_image: FibsemImage,
    eb_image: FibsemImage,
    milling_stages: list[FibsemMillingStage],
    draw_crosshair: bool = True,
):

    # colour wheel
    COLOURS = ["yellow", "cyan", "magenta", "lime", "orange", "hotpink", "green", "blue", "red", "purple"]

    # convert fibsem patterns to napari shapes
    import time

    t_1 = time.time()
    _ignore = []
    for i, stage in enumerate(milling_stages):
        shape_patterns = []
        shape_types = []
        t0 = time.time()

        patterns = stage.pattern.patterns
        point = stage.pattern.point
        name = stage.name
        is_line_pattern = False

        for pattern_settings in patterns:
            if isinstance(pattern_settings, FibsemBitmapSettings):
                if pattern_settings.path == None or pattern_settings.path == '':
                    continue

                bmp_Image, translate_position = convert_bitmap_pattern_to_napari_image(pattern_settings=pattern_settings, image=ib_image)
                if "bmp_Image" in viewer.layers:
                    viewer.layers.remove(viewer.layers["bmp_Image"])
                viewer.add_image(bmp_Image,translate=translate_position,name="bmp_Image")
                shape_patterns = []
                _ignore.append("bmp_Image")
                continue

            elif isinstance(pattern_settings, FibsemCircleSettings):
                if pattern_settings.thickness != 0:
                    annulus_image, translate_position = convert_pattern_to_napari_image(pattern_settings=pattern_settings, image=ib_image)
                    if "annulus_Image" in viewer.layers:
                        viewer.layers.remove(viewer.layers["annulus_Image"])
                    viewer.add_image(annulus_image,translate=translate_position,name="annulus_Image",blending="additive",colormap=COLOURS[i % len(COLOURS)],opacity=0.4)
                    shape_patterns = []
                    _ignore.append("annulus_Image")
                    continue
                else:
                    shape = convert_pattern_to_napari_circle(pattern_settings=pattern_settings, image=ib_image)
                    shape_types.append("ellipse")
                    _ignore.append(name)

            elif isinstance(pattern_settings, FibsemLineSettings):
                shape = convert_pattern_to_napari_line(pattern_settings=pattern_settings, image=ib_image)
                shape_types.append("line")
                _ignore.append(name)
                is_line_pattern = True
            
            else:
                shape = convert_pattern_to_napari_rect(
                    pattern_settings=pattern_settings, image=ib_image
                )
                shape_types.append("rectangle")
                _ignore.append(name)

            # offset the x coord by image width
            if eb_image is not None:
                for c in shape:
                    c[1] += eb_image.data.shape[1]
            shape_patterns.append(shape)
        
        t1 = time.time()

        if len(shape_patterns) > 0:
            
            if draw_crosshair:
                crosshair_shapes = create_crosshair_shape(centre_point=point, image=ib_image, eb_image=eb_image)
                crosshair_shape_types = ["rectangle","rectangle"]
                shape_patterns += crosshair_shapes
                shape_types += crosshair_shape_types

            # _name = f"Stage {i+1:02d}"
            if name in viewer.layers:
                viewer.layers[name].data = []
                viewer.layers[name].data = shape_patterns
                viewer.layers[name].shape_type = shape_types
                viewer.layers[name].edge_color = COLOURS[i % len(COLOURS)]
                viewer.layers[name].face_color=COLOURS[i % len(COLOURS)]
            else:
                viewer.add_shapes(
                    shape_patterns,
                    name=name,
                    shape_type=shape_types,
                    edge_width=0.5,
                    edge_color=COLOURS[i % len(COLOURS)],
                    face_color=COLOURS[i % len(COLOURS)],
                    opacity=0.5,
                    blending="translucent",
                )

            if is_line_pattern:
                viewer.layers[name].edge_width = 3

        t2 = time.time()
        # remove all un-updated layers (assume they have been deleted)        
        _remove_all_layers(viewer=viewer, layer_type=napari.layers.shapes.shapes.Shapes, _ignore=_ignore)#[stage.name for stage in milling_stages])
        t3 = time.time()
        logging.debug(f"_DRAW_SHAPES: CONVERT: {t1-t0}, ADD/UPDATE: {t2-t1}, REMOVE: {t3-t2}")
    t_2 = time.time()
    logging.debug(f"_DRAW_SHAPES: total time: {t_2-t_1}")

def message_box_ui(title: str, text: str, buttons=QMessageBox.Yes | QMessageBox.No):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    msg.exec_()

    response = (
        True
        if (msg.clickedButton() == msg.button(QMessageBox.Yes))
        or (msg.clickedButton() == msg.button(QMessageBox.Ok))
        else False
    )

    return response

from PyQt5 import QtGui
def _display_logo(path, label, shape=[50, 50]):
    label.setScaledContents(True)
    label.setFixedSize(*shape)
    label.setPixmap(QtGui.QPixmap(path))


def create_combobox_message_box(text: str, title: str, options: list, parent = None):
    # create a q message box with combobox
    msg = QtWidgets.QMessageBox(parent=parent)
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

    # create a combobox
    combobox = QtWidgets.QComboBox(msg)
    combobox.addItems(options)

    # add combobox to message box
    msg.layout().addWidget(combobox, 1, 1)

    # show message box
    msg.exec_()

    # get the selected milling pattern

    if msg.result() == QtWidgets.QMessageBox.Ok:
        selected = combobox.currentText()

        return selected
    
    return None


        
def _draw_crosshair(viewer: napari.Viewer, eb_image, ib_image,is_checked=False, width: float = 0.15) -> None:


    layers_in_napari = []
    for layer in viewer.layers:
        layers_in_napari.append(layer.name)


    if is_checked:

        centre_points = [ [eb_image.data.shape[0]//2, eb_image.data.shape[1]//2],[ib_image.data.shape[0]//2, eb_image.data.shape[1] + ib_image.data.shape[1]//2]]

        crosshairs = []

        for i,point in enumerate(centre_points):
            
            eb_location_r,eb_location_c = point[0], point[1]
            crosshair_length = width*point[0]
            horizontal_line = [ [eb_location_r, eb_location_c - crosshair_length],[eb_location_r,eb_location_c + crosshair_length] ]
            vertical_line = [ [eb_location_r - crosshair_length, eb_location_c],[eb_location_r + crosshair_length,eb_location_c] ]

            crosshairs.append(horizontal_line)
            crosshairs.append(vertical_line)

        if f"crosshair" not in layers_in_napari:

            viewer.add_shapes(data=crosshairs, shape_type='line', edge_width=5, edge_color='yellow', face_color='yellow', opacity=0.8, blending='translucent', name=f'crosshair')
        else:
            viewer.layers[f"crosshair"].data = crosshairs
            viewer.layers[f"crosshair"].opacity = 0.8

    else:
        
        if f"crosshair" in layers_in_napari :
            viewer.layers["crosshair"].opacity = 0
           
    return 

def _scale_length_value(hfw: float) -> float:

    scale_length_value = hfw*constants.METRE_TO_MICRON*0.2

    if scale_length_value > 0 and scale_length_value < 100:
        scale_length_value = round(scale_length_value/5)*5
    if scale_length_value > 100 and scale_length_value < 500:
        scale_length_value = round(scale_length_value/25)*25
    if scale_length_value > 500 and scale_length_value < 1000:
        scale_length_value = round(scale_length_value/50)*50

    scale_ratio = scale_length_value/(hfw*constants.METRE_TO_MICRON)

    return scale_ratio,scale_length_value

def _draw_scalebar(viewer: napari.Viewer, eb_image, ib_image,is_checked=False, width: float = 0.1) -> np.ndarray:

    layers_in_napari = []
    for layer in viewer.layers:
        layers_in_napari.append(layer.name)


    if is_checked:

        location_points = [ [int(eb_image.data.shape[0]*0.9), int(eb_image.data.shape[1]*0.15)],[int(ib_image.data.shape[0]*0.9), int(eb_image.data.shape[1] + ib_image.data.shape[1]*0.15)]]

        if is_checked:
            
            # making the scale bar line
            scale_bar_shape = []

            for i,point in enumerate(location_points):
                
                

                if i == 0:
                    scale_ratio,eb_scale = _scale_length_value(eb_image.metadata.image_settings.hfw)
                    length = scale_ratio*eb_image.data.shape[1]
                else:
                    scale_ratio,ib_scale = _scale_length_value(ib_image.metadata.image_settings.hfw)
                    length = scale_ratio*ib_image.data.shape[1]

                main_line = [[point[0]+25, int(point[1]-0.5*length)], [point[0]+25, int(point[1]+0.5*length)]]
        
                left_line = [[point[0]+50, int(point[1]-0.5*length)], [point[0], int(point[1]-0.5*length)]]
                right_line = [[point[0]+50, int(point[1]+0.5*length)], [point[0], int(point[1]+0.5*length)]]

                scale_bar_shape.append(main_line)
                scale_bar_shape.append(left_line)
                scale_bar_shape.append(right_line)

            if "scalebar" not in layers_in_napari:
                
                viewer.add_shapes(
                    data=scale_bar_shape,
                    shape_type='line',
                    edge_width=5,
                    edge_color='yellow',
                    name='scalebar'
                )
            else:

                viewer.layers["scalebar"].data = scale_bar_shape
                viewer.layers["scalebar"].opacity = 1

            ## making the scale bar value

            

            text = {
                "string": [f"{eb_scale} um",f"{ib_scale} um"],
                "color":"white"
            }

            if "scalebar" not in layers_in_napari:

                viewer.add_points(
                    data=location_points,
                    text=text,
                    size=20,
                    name="scalebar_value",
                    edge_color='transparent',
                    face_color='transparent',

                )
            else:
                viewer.layers["scalebar_value"].data = location_points
                viewer.layers["scalebar_value"].text = text
                viewer.layers["scalebar_value"].opacity = 1

    else:

        if "scalebar" in layers_in_napari:
            viewer.layers["scalebar"].opacity = 0
            viewer.layers["scalebar_value"].opacity = 0


    


def convert_point_to_napari(resolution: list, pixel_size: float, centre: Point):
    icy, icx = resolution[1] // 2, resolution[0] // 2

    cx = int(icx + (centre.x / pixel_size))
    cy = int(icy - (centre.y / pixel_size))

    return Point(cx, cy)


def validate_pattern_placement(
    patterns: list[FibsemPatternSettings], resolution: list, shape: list[list[float]]
):
    x_lim = resolution[0]
    y_lim = resolution[1]

    for coordinate in shape:
        x_coord = coordinate[1]
        y_coord = coordinate[0]

        if x_coord < 0 or x_coord > x_lim:
            return False
        if y_coord < 0 or y_coord > y_lim:
            return False

    return True


from fibsem import config as cfg
from pathlib import Path


# TODO: add filters for file types


def _get_directory_ui(
    msg: str = "Select a directory", path: Path = cfg.LOG_PATH, parent=None
) -> Path:
    path = QtWidgets.QFileDialog.getExistingDirectory(parent, msg, directory=path)
    return path


def _get_file_ui(
    msg: str = "Select a file",
    path: Path = cfg.LOG_PATH,
    _filter: str = "*yaml",
    parent=None,
) -> Path:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, msg, directory=path, filter=_filter
    )
    return path


def _get_save_file_ui(
    msg: str = "Select a file",
    path: Path = cfg.LOG_PATH,
    _filter: str = "*yaml",
    parent=None,
) -> Path:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        msg,
        directory=path,
        filter=_filter,
    )
    return path


def _get_text_ui(
    msg: str = "Enter text",
    title: str = "Text Entry",
    default: str = "UserText",
    parent=None,
) -> tuple[str, bool]:
    text, okPressed = QtWidgets.QInputDialog.getText(
        parent,
        title,
        msg,
        QtWidgets.QLineEdit.Normal,
        default,
    )

    return text, okPressed

def _draw_milling_stages_on_image(image: FibsemImage, milling_stages: list[FibsemMillingStage], show: bool = True):

    viewer = napari.Viewer()
    viewer.add_image(image.data, name='test_image')
    _draw_patterns_in_napari(viewer=viewer,ib_image=image,eb_image=None,milling_stages=milling_stages)
    screenshot = viewer.screenshot()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot)
    viewer.close()

    for i,stage in enumerate(milling_stages):
    
        plt.plot(0,0,'-',color=COLOURS[i % len(COLOURS)],label=stage.name)

    ax.axis('off')
    ax.legend()
    if show:
        plt.show()
    
    return fig

def _calculate_fiducial_area_v2(image: FibsemImage, fiducial_centre: Point, fiducial_length:float)->tuple[FibsemRectangle, bool]:
    pixelsize = image.metadata.pixel_size.x
    
    fiducial_centre.y = -fiducial_centre.y
    fiducial_centre_px = conversions.convert_point_from_metres_to_pixel(
        fiducial_centre, pixelsize
    )

    rcx = fiducial_centre_px.x / image.metadata.image_settings.resolution[0] + 0.5
    rcy = fiducial_centre_px.y / image.metadata.image_settings.resolution[1] + 0.5

    fiducial_length_px = (
        conversions.convert_metres_to_pixels(fiducial_length, pixelsize) * 2 # SCALE_FACTOR
    )
    h_offset = fiducial_length_px / image.metadata.image_settings.resolution[0] / 2
    v_offset = fiducial_length_px / image.metadata.image_settings.resolution[1] / 2

    left = rcx - h_offset
    top = rcy - v_offset
    width = 2 * h_offset
    height = 2 * v_offset

    if left < 0 or (left + width) > 1 or top < 0 or (top + height) > 1:
        flag = True
    else:
        flag = False

    fiducial_area = FibsemRectangle(left, top, width, height)

    return fiducial_area, flag

    
def show_information_dialog(microscope: FibsemMicroscope, parent=None):
    import fibsem
    
    fibsem_version = fibsem.__version__
    autolamella_version = "Not Installed"
    try:
        import autolamella
        autolamella_version = autolamella.__version__
    except:
        pass
    
    info = microscope.system.info

    text = f"""
    OpenFIBSEM Information:
    OpenFIBSEM: {fibsem_version}
    AutoLamella: {autolamella_version}

    Microscope Information:
    Name: {info.name}
    Manufacturer: {info.manufacturer}
    Model: {info.model}
    Serial Number: {info.serial_number}
    Firmware Version: {info.hardware_version}
    Software Version: {info.software_version}
    """

    # create a qdialog box with information
    msg = QtWidgets.QMessageBox(parent=parent)
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setWindowTitle("Information")
    msg.setText(text)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

    # exec
    msg.exec_()
