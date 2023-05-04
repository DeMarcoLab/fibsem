import logging
from dataclasses import dataclass
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fibsem.config import load_microscope_manufacturer

from fibsem.structures import Point, FibsemImage, FibsemPatternSettings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QMessageBox, QSizePolicy, QVBoxLayout, QWidget
import napari

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

def set_arr_as_qlabel(
    arr: np.ndarray,
    label: QLabel,
    shape: tuple = (1536//4, 1024//4),
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

def convert_pattern_to_napari_circle(pattern_settings: FibsemPatternSettings, image: FibsemImage):

    # image centre
    icy, icx = image.metadata.image_settings.resolution[1] // 2, image.metadata.image_settings.resolution[0] // 2 # TODO; this should be the actual shape of the image

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
    shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]] #??
    return shape

def convert_pattern_to_napari_rect(
    pattern_settings: FibsemPatternSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.metadata.image_settings.resolution[1] // 2, image.metadata.image_settings.resolution[0] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y
    # extract pattern information from settings
    from fibsem.structures import FibsemPattern
    if pattern_settings.pattern is FibsemPattern.Line:
        pattern_width = pattern_settings.end_x - pattern_settings.start_x
        pattern_height = max(pattern_settings.end_y - pattern_settings.start_y, 0.5e-6)
        pattern_rotation = np.arctan2(pattern_height, pattern_width) # TODO: line rotation doesnt work correctly, fix
        pattern_centre_x = (pattern_settings.end_x + pattern_settings.start_x) / 2
        pattern_centre_y = (pattern_settings.end_y + pattern_settings.start_y) / 2
    
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

def _remove_all_layers(viewer: napari.Viewer, layer_type = napari.layers.shapes.shapes.Shapes):
    # remove all shapes layers
    layers_to_remove = []
    for layer in viewer.layers:
        if isinstance(layer, layer_type):
            layers_to_remove.append(layer)
    for layer in layers_to_remove:
        viewer.layers.remove(layer)  # Not removing the second layer?

def _draw_patterns_in_napari(
    viewer: napari.Viewer,
    ib_image: FibsemImage,
    eb_image: FibsemImage,
    all_patterns: list[FibsemPatternSettings],
):

    _remove_all_layers(viewer=viewer, layer_type=napari.layers.shapes.shapes.Shapes)
    
    # colour wheel
    # colour = ["orange", "yellow", "red", "green", "purple"]
    colour = ["yellow", "cyan", "magenta", "green", "orange"]
    from fibsem.structures import FibsemPattern
   
    # convert fibsem patterns to napari shapes

    for i, stage in enumerate(all_patterns):
        shape_patterns = []
        shape_types = []
        for pattern_settings in stage:
            if pattern_settings.pattern is FibsemPattern.Circle:
                shape = convert_pattern_to_napari_circle(pattern_settings=pattern_settings, image=ib_image)
                shape_types.append("ellipse")
            else:
                shape = convert_pattern_to_napari_rect(pattern_settings=pattern_settings, image=ib_image)
                shape_types.append("rectangle")

            # offset the x coord by image width
            if eb_image is not None:
                for c in shape:
                    c[1] += eb_image.data.shape[1]
            shape_patterns.append(shape)
        viewer.add_shapes(
            shape_patterns,
            name=f"Stage {i+1}",
            shape_type=shape_types,
            edge_width=0.5,
            edge_color=colour[i % 5],
            face_color=colour[i % 5],
            opacity=0.5,
        )

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
        
def _draw_crosshair(arr: np.ndarray, width: float = 0.1) -> np.ndarray:
    # add crosshair
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    from PIL import Image, ImageDraw
    im = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(im)
    # 10% of img width in pixels
    length = int(im.size[0] * width / 2)
    draw.line((cx, cy-length) + (cx, cy+length), fill="yellow", width=3)
    draw.line((cx-length, cy) + (cx+length, cy), fill="yellow", width=3)

    arr = np.array(im)
    return arr

def convert_point_to_napari(resolution: list, pixel_size: float, centre: Point):

    icy, icx = resolution[1] // 2, resolution[0] // 2

    cx = int(icx + (centre.x / pixel_size))
    cy = int(icy - (centre.y / pixel_size))
    
    return Point(cx, cy)