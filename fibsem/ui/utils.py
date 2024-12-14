import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import napari
import numpy as np
from napari.layers import Layer as NapariLayer
from napari.layers.shapes.shapes import Shapes as NapariShapes
from PIL import Image
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QMessageBox,
)

from fibsem import config as cfg
from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    Point,
)


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
) -> Tuple[str, bool]:
    text, okPressed = QtWidgets.QInputDialog.getText(
        parent,
        title,
        msg,
        QtWidgets.QLineEdit.Normal,
        default,
    )

    return text, okPressed

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
