import logging
import os
import sys
import glob

# import fibsem
import napari
import napari.utils.notifications
from PIL import Image
from PyQt5 import QtWidgets
import dask.array as da
from fibsem.segmentation import config as cfg
from fibsem.segmentation import utils as seg_utils
from fibsem.segmentation import train as seg_train

from fibsem.ui.qtdesigner_files import FibsemModelTrainingWidget
import yaml
from fibsem.segmentation.config import CLASS_COLORS


# BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")
from PyQt5.QtCore import pyqtSignal

from napari.qt.threading import thread_worker


class FibsemModelTrainingWidget(FibsemModelTrainingWidget.Ui_Form, QtWidgets.QWidget):
    ui_signal = pyqtSignal(dict)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent=None,
    ):
        super(FibsemModelTrainingWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.viewer = viewer

        self.setup_connections()

    def setup_connections(self):
        self.pushButton_train_model.clicked.connect(self.train_model)

        self.comboBox_encoder.addItems(seg_utils.unet_encoders)

        self.ui_signal.connect(self.train_model_yielded)

        # NOTE: Setup your config.yml file
        with open(cfg.CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        checkpoint = config["checkpoint"] if config["checkpoint"] is not None else ""

        self.lineEdit_data_path.setText(config["data_paths"][0])
        self.lineEdit_label_path.setText(config["label_paths"][0])
        self.lineEdit_save_path.setText(config["save_path"])
        self.comboBox_encoder.setCurrentText(config["encoder"])
        self.lineEdit_checkpoint.setText(checkpoint)
        self.spinBox_num_classes.setValue(config["num_classes"])
        self.spinBox_batch_size.setValue(config["batch_size"])
        self.spinBox_epochs.setValue(config["epochs"])
        self.doubleSpinBox_lr.setValue(config["lr"])
        self.checkBox_use_wandb.setChecked(config["wandb"])
        self.lineEdit_wandb_project.setText(config["wandb_project"])
        self.lineEdit_wandb_entity.setText(config["wandb_entity"])

    def train_model(self):
        checkpoint = self.lineEdit_checkpoint.text()
        checkpoint = None if checkpoint == "" else checkpoint

        config = {
            "data_paths": [self.lineEdit_data_path.text()],
            "label_paths": [self.lineEdit_label_path.text()],
            "save_path": self.lineEdit_save_path.text(),
            "encoder": self.comboBox_encoder.currentText(),
            "checkpoint": checkpoint,
            "num_classes": int(self.spinBox_num_classes.value()),
            "batch_size": self.spinBox_batch_size.value(),
            "epochs": self.spinBox_epochs.value(),
            "lr": self.doubleSpinBox_lr.value(),
            "wandb": self.checkBox_use_wandb.isChecked(),
            "wandb_project": self.lineEdit_wandb_project.text(),
            "wandb_entity": self.lineEdit_wandb_entity.text(),
        }

        self.label_info.setText("Model Training...")
        self.pushButton_train_model.setEnabled(False)
        worker = self.train_worker(config)
        worker.finished.connect(self.train_model_finished)
        worker.start()

    @thread_worker
    def train_worker(self, config: dict):
        ################################## VALIDATE CONFIG ##################################
        seg_utils.validate_config(config)

        ################################## SETUP W&B ##################################
        seg_train._setup_wandb(config)

        ################################## LOAD DATASET ##################################
        train_data_loader, val_data_loader = seg_train._setup_dataset(config)

        ################################## LOAD MODEL ##################################

        model, optimizer, device = seg_train._setup_model(config)
        
        self.viewer.layers.clear()
        # self.viewer.grid.enabled = True

        seg_train.train_model(
            model,
            device,
            optimizer,
            train_data_loader,
            val_data_loader,
            config=config,
        )

    def train_model_yielded(self, info: dict):

        if info["stage"] == "end":
            self.epoch = info["epoch"]
            self.epochs = info["epochs"]
            self.train_loss = info["train_loss"]
            self.val_loss = info["val_loss"]
        if info["stage"] == "train":
            self.train_loss = info["train_loss"]
        if info["stage"] == "val":
            self.val_loss = info["val_loss"]
        msg = f"Epoch: {self.epoch}/{self.epochs} | Train Loss: {self.train_loss:.4f} | Val Loss: {self.val_loss:.4f}"
        self.label_info.setText(msg)

        if info["stage"] in ["train", "val"]:
            
            try:
                self._image_layer.data = info["image"]
            except:
                self._image_layer = self.viewer.add_image(
                    info["image"],
                    name="Image",
                    opacity=0.7,
                    blending="additive",
                )
            try:
                self._label_layer.data = info["gt"]
            except:
                self._label_layer = self.viewer.add_labels(
                    info["gt"],
                    name="Mask",    
                    opacity=0.7,
                    blending="additive",
                    color=CLASS_COLORS,
                )
            try:
                self._pred_layer.data = info["pred"]
            except:
                self._pred_layer = self.viewer.add_labels(
                    info["pred"],
                    name="Prediction",
                    opacity=0.7,
                    blending="additive",
                    color=CLASS_COLORS,
                )

    def train_model_finished(self):
        msg = "Train Model Finished!"
        napari.utils.notifications.show_info(msg)
        self.label_info.setText(msg)
        self.pushButton_train_model.setEnabled(True)

    def label_info(self, text: str):
        self.label_info.setText(text)


def main():
    viewer = napari.Viewer()
    widget = FibsemModelTrainingWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


if __name__ == "__main__":
    main()


# TODO: stop training button
