
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
