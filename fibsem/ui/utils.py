
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QMessageBox,
    QWidget,
)

from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope

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


def message_box_ui(title: str, 
                   text: str, 
                   buttons=QMessageBox.Yes | QMessageBox.No, 
                   parent: Optional[QWidget] = None) -> bool:
    msg = QMessageBox(parent=parent)
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



# TODO: add filters for file types

def open_existing_directory_dialog(
    msg: str = "Select a directory", path: Path = cfg.LOG_PATH, parent=None
) -> Path:
    path = QtWidgets.QFileDialog.getExistingDirectory(parent=parent, caption=msg, directory=path)
    return path


def open_existing_file_dialog(
    msg: str = "Select a file",
    path: Path = cfg.LOG_PATH,
    _filter: str = "*yaml",
    parent=None,
) -> Path:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent=parent, 
        caption=msg, 
        directory=path, 
        filter=_filter
    )
    return path


def open_save_file_dialog(
    msg: str = "Select a file",
    path: Path = cfg.LOG_PATH,
    _filter: str = "*yaml",
    parent=None,
) -> Path:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent=parent,
        caption=msg,
        directory=path,
        filter=_filter,
    )
    return path

def open_text_input_dialog(
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

def open_information_dialog(microscope: FibsemMicroscope, parent=None):
    import fibsem
    
    fibsem_version = fibsem.__version__
    autolamella_version = "Not Installed"
    try:
        import autolamella
        autolamella_version = autolamella.__version__
    except ImportError:
        pass
    from fibsem.structures import SystemInfo
    info: SystemInfo = microscope.system.info

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


import numpy as np
import matplotlib.pyplot as plt

def create_nested_squares(array_size: int = 1000, 
                          orange_size: int = 800, 
                          green_size: int = 600) -> np.ndarray:
    """
    Create a 2D numpy array with nested squares.
    
    Colors:
      white = 0 (background)
      orange = 1 (outer square)
      green = 2 (inner square)
      
    Args:
        array_size (int): Size of the output square array (array_size x array_size).
        orange_size (int): Side length of the orange square.
        green_size (int): Side length of the green square.
        
    Returns:
        np.ndarray: 2D array representing the nested squares.
    """
    if green_size > orange_size:
        raise ValueError("green_size must be less than or equal to orange_size")
        
    # Create white background 
    img = np.zeros((array_size, array_size), dtype=np.uint8)
    
    center = array_size // 2
    
    # Draw the orange square
    half_orange = orange_size // 2
    orange_top = center - half_orange
    orange_bottom = orange_top + orange_size
    img[orange_top:orange_bottom, orange_top:orange_bottom] = 1  # set orange square to 1
    
    # Draw the green square inside the orange square
    half_green = green_size // 2
    green_top = center - half_green
    green_bottom = green_top + green_size
    img[green_top:green_bottom, green_top:green_bottom] = 2  # set green square to 2
    
    return img

def tile_nested_squares(tile_rows: int, tile_cols: int,
                        array_size: int = 1000, 
                        orange_size: int = 800, 
                        green_size: int = 600) -> np.ndarray:
    """
    Create a large grid by tiling nested squares.
    
    Args:
        tile_rows (int): Number of tiles vertically.
        tile_cols (int): Number of tiles horizontally.
        array_size (int): Size of each nested square array.
        orange_size (int): Side length of the orange square.
        green_size (int): Side length of the green square.
    
    Returns:
        np.ndarray: Tiled grid of nested squares.
    """
    tile = create_nested_squares(array_size, orange_size, green_size)
    # Use np.tile to replicate the tile
    tiled_grid = np.tile(tile, (tile_rows, tile_cols))
    return tiled_grid


# Example usage:
if __name__ == "__main__":

    gridbar_thickness = 200
    mesh_size = 700
    keepout = 100
    pixelsize = 1e-6

    array_size = gridbar_thickness + mesh_size
    orange_size = mesh_size
    green_size = mesh_size - 2*keepout
    # Create a 3x3 grid of nested squares
    grid = tile_nested_squares(tile_rows=5,
                               tile_cols=5,
                               array_size=array_size,
                               orange_size=orange_size,
                               green_size=green_size)
    import napari

    cmap = {0: 'red', 1: 'orange', 2: 'green'}

    viewer = napari.view_labels(grid,
                                name="Grid Overlay",
                                scale=(pixelsize, pixelsize),
                                colormap=cmap)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "m"

    napari.run()