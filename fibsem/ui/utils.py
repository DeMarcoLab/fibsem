import logging
from dataclasses import dataclass
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from autoscript_sdb_microscope_client._dynamic_object_proxies import (
    CleaningCrossSectionPattern, RectanglePattern)
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem.structures import Point
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QMessageBox, QSizePolicy, QVBoxLayout, QWidget

# TODO: clean up and refactor these (_WidgetPlot and _PlotCanvas)
class _WidgetPlot(QWidget):
    def __init__(self, *args, display_image, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = _PlotCanvas(self, image=display_image)
        self.layout().addWidget(self.canvas)


class _PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, image=None):
        self.fig = Figure()
        FigureCanvasQTAgg.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self)
        self.image = image
        self.plot()
        self.createConn()

        self.figureActive = False
        self.axesActive = None
        self.cursorGUI = "arrow"
        self.cursorChanged = False

    def plot(self):
        gs0 = self.fig.add_gridspec(1, 1)

        self.ax11 = self.fig.add_subplot(gs0[0], xticks=[], yticks=[], title="")

        if self.image.ndim == 3:
            self.ax11.imshow(self.image,)
        else:
            self.ax11.imshow(self.image, cmap="gray")

    def updateCanvas(self, event=None):
        ax11_xlim = self.ax11.get_xlim()
        ax11_xvis = ax11_xlim[1] - ax11_xlim[0]

        while len(self.ax11.patches) > 0:
            [p.remove() for p in self.ax11.patches]
        while len(self.ax11.texts) > 0:
            [t.remove() for t in self.ax11.texts]

        ax11_units = ax11_xvis * 0.003
        self.fig.canvas.draw()

    def createConn(self):
        self.fig.canvas.mpl_connect("figure_enter_event", self.activeFigure)
        self.fig.canvas.mpl_connect("figure_leave_event", self.leftFigure)
        self.fig.canvas.mpl_connect("button_press_event", self.mouseClicked)
        self.ax11.callbacks.connect("xlim_changed", self.updateCanvas)

    def activeFigure(self, event):
        self.figureActive = True

    def leftFigure(self, event):
        self.figureActive = False
        if self.cursorGUI != "arrow":
            self.cursorGUI = "arrow"
            self.cursorChanged = True

    def mouseClicked(self, event):
        x = event.xdata
        y = event.ydata


@dataclass
class Crosshair:
    rectangle_horizontal: plt.Rectangle
    rectangle_vertical: plt.Rectangle


def create_crosshair(
    image: Union[np.ndarray, AdornedImage], x=None, y=None, colour="xkcd:yellow"
):
    if type(image) == AdornedImage:
        image = image.data

    midx = int(image.shape[1] / 2) if x is None else x
    midy = int(image.shape[0] / 2) if y is None else y

    cross_width = int(0.05 / 100 * image.shape[1])
    cross_length = int(5 / 100 * image.shape[1])

    rect_horizontal = plt.Rectangle(
        (midx - cross_length / 2, midy - cross_width / 2), cross_length, cross_width
    )
    rect_vertical = plt.Rectangle(
        (midx - cross_width, midy - cross_length / 2), cross_width * 2, cross_length
    )

    # set colours
    rect_horizontal.set_color(colour)
    rect_vertical.set_color(colour)

    return Crosshair(
        rectangle_horizontal=rect_horizontal, rectangle_vertical=rect_vertical
    )

# TODO update with Point
def draw_crosshair(image, canvas, x: float = None, y: float = None, colour: str ="yellow"):
    # draw crosshairs
    crosshair = create_crosshair(image, x, y, colour=colour)
    for patch in crosshair.__dataclass_fields__:
        canvas.ax11.add_patch(getattr(crosshair, patch))
        getattr(crosshair, patch).set_visible(True)

# draw arrow
def draw_arrow(p1: Point, p2: Point, canvas) -> None:
    """Draw an arrow patch from p1 to p2"""
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    line = mpatches.Arrow(x1, y1, x2 - x1, y2 - y1, color="white")

    # draw line
    canvas.ax11.add_patch(line)
    return


def draw_crosshair_v2(image: AdornedImage, canvas: _PlotCanvas, point: Point, colour: str = "yellow"):

    markersize = max(image.data.shape) // 20
    
    canvas.ax11.plot(point.x, point.y, marker="+", color=colour, ms=markersize, markeredgewidth=2)

    return 

def display_error_message(message, title="Error"):
    """PyQt dialog box displaying an error message."""
    logging.exception(message)

    error_dialog = QMessageBox()
    error_dialog.setIcon(QMessageBox.Critical)
    error_dialog.setText(message)
    error_dialog.setWindowTitle(title)
    error_dialog.exec_()

    return error_dialog


def message_box_ui(title: str, text: str, buttons = QMessageBox.Yes | QMessageBox.No):

    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    msg.exec_()
    
    response = True if (msg.clickedButton() == msg.button(QMessageBox.Yes)) or (msg.clickedButton() == msg.button(QMessageBox.Ok) ) else False

    return response



def draw_rectangle_pattern(adorned_image: AdornedImage, pattern: Union[RectanglePattern, CleaningCrossSectionPattern], colour: str ="yellow") -> Rectangle:
    """Draw a AutoSCript Rectangle Pattern as Matplotib Rectangle"""
    rectangle = Rectangle(
        (0, 0),
        0.2,
        0.2,
        color=colour,
        fill=None,
        alpha=1,
        angle=np.rad2deg(-pattern.rotation),
    )
    rectangle.set_visible(False)
    rectangle.set_hatch("//////")

    image_width = adorned_image.width
    image_height = adorned_image.height
    pixel_size = adorned_image.metadata.binary_result.pixel_size.x

    width = pattern.width / pixel_size
    height = pattern.height / pixel_size
    rotation = -pattern.rotation
    rectangle_left = (
        (image_width / 2)
        + (pattern.center_x / pixel_size)
        - (width / 2) * np.cos(rotation)
        + (height / 2) * np.sin(rotation)
    )
    rectangle_bottom = (
        (image_height / 2)
        - (pattern.center_y / pixel_size)
        - (height / 2) * np.cos(rotation)
        - (width / 2) * np.sin(rotation)
    )
    rectangle.set_width(width)
    rectangle.set_height(height)
    rectangle.set_xy((rectangle_left, rectangle_bottom))
    rectangle.set_visible(True)

    return rectangle


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
        QImage.Format_Grayscale16,
    )
    label.setPixmap(QPixmap.fromImage(image).scaled(*shape))

    return label

def set_arr_as_qlabel_8(
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