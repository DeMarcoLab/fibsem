# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemTileWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(463, 449)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 17, 0, 1, 2)
        self.doubleSpinBox_tile_tile_size = QtWidgets.QDoubleSpinBox(Form)
        self.doubleSpinBox_tile_tile_size.setMaximum(100000.0)
        self.doubleSpinBox_tile_tile_size.setProperty("value", 100.0)
        self.doubleSpinBox_tile_tile_size.setObjectName("doubleSpinBox_tile_tile_size")
        self.gridLayout.addWidget(self.doubleSpinBox_tile_tile_size, 3, 1, 1, 1)
        self.label_tile_overlap = QtWidgets.QLabel(Form)
        self.label_tile_overlap.setObjectName("label_tile_overlap")
        self.gridLayout.addWidget(self.label_tile_overlap, 4, 0, 1, 1)
        self.pushButton_add_position = QtWidgets.QPushButton(Form)
        self.pushButton_add_position.setObjectName("pushButton_add_position")
        self.gridLayout.addWidget(self.pushButton_add_position, 11, 0, 1, 1)
        self.label_tile_tile_size = QtWidgets.QLabel(Form)
        self.label_tile_tile_size.setObjectName("label_tile_tile_size")
        self.gridLayout.addWidget(self.label_tile_tile_size, 3, 0, 1, 1)
        self.pushButton_load_image = QtWidgets.QPushButton(Form)
        self.pushButton_load_image.setObjectName("pushButton_load_image")
        self.gridLayout.addWidget(self.pushButton_load_image, 7, 0, 1, 2)
        self.label_tile_label = QtWidgets.QLabel(Form)
        self.label_tile_label.setObjectName("label_tile_label")
        self.gridLayout.addWidget(self.label_tile_label, 5, 0, 1, 1)
        self.label_instructions = QtWidgets.QLabel(Form)
        self.label_instructions.setObjectName("label_instructions")
        self.gridLayout.addWidget(self.label_instructions, 16, 0, 1, 2)
        self.comboBox_tile_position = QtWidgets.QComboBox(Form)
        self.comboBox_tile_position.setObjectName("comboBox_tile_position")
        self.gridLayout.addWidget(self.comboBox_tile_position, 10, 1, 1, 1)
        self.label_tile_grid_size = QtWidgets.QLabel(Form)
        self.label_tile_grid_size.setObjectName("label_tile_grid_size")
        self.gridLayout.addWidget(self.label_tile_grid_size, 2, 0, 1, 1)
        self.pushButton_remove_position = QtWidgets.QPushButton(Form)
        self.pushButton_remove_position.setObjectName("pushButton_remove_position")
        self.gridLayout.addWidget(self.pushButton_remove_position, 11, 1, 1, 1)
        self.comboBox_tile_beam_type = QtWidgets.QComboBox(Form)
        self.comboBox_tile_beam_type.setObjectName("comboBox_tile_beam_type")
        self.gridLayout.addWidget(self.comboBox_tile_beam_type, 1, 1, 1, 1)
        self.label_position_info = QtWidgets.QLabel(Form)
        self.label_position_info.setObjectName("label_position_info")
        self.gridLayout.addWidget(self.label_position_info, 12, 0, 1, 2)
        self.label_position_header = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_position_header.setFont(font)
        self.label_position_header.setObjectName("label_position_header")
        self.gridLayout.addWidget(self.label_position_header, 10, 0, 1, 1)
        self.doubleSpinBox_tile_grid_size = QtWidgets.QDoubleSpinBox(Form)
        self.doubleSpinBox_tile_grid_size.setMaximum(1000000.0)
        self.doubleSpinBox_tile_grid_size.setProperty("value", 400.0)
        self.doubleSpinBox_tile_grid_size.setObjectName("doubleSpinBox_tile_grid_size")
        self.gridLayout.addWidget(self.doubleSpinBox_tile_grid_size, 2, 1, 1, 1)
        self.pushButton_move_to_position = QtWidgets.QPushButton(Form)
        self.pushButton_move_to_position.setObjectName("pushButton_move_to_position")
        self.gridLayout.addWidget(self.pushButton_move_to_position, 13, 0, 1, 2)
        self.pushButton_run_tile_collection = QtWidgets.QPushButton(Form)
        self.pushButton_run_tile_collection.setObjectName("pushButton_run_tile_collection")
        self.gridLayout.addWidget(self.pushButton_run_tile_collection, 6, 0, 1, 2)
        self.lineEdit_tile_label = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_tile_label.sizePolicy().hasHeightForWidth())
        self.lineEdit_tile_label.setSizePolicy(sizePolicy)
        self.lineEdit_tile_label.setObjectName("lineEdit_tile_label")
        self.gridLayout.addWidget(self.lineEdit_tile_label, 5, 1, 1, 1)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.label_tile_beam_type = QtWidgets.QLabel(Form)
        self.label_tile_beam_type.setObjectName("label_tile_beam_type")
        self.gridLayout.addWidget(self.label_tile_beam_type, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 8, 0, 1, 2)
        self.pushButton_load_positions = QtWidgets.QPushButton(Form)
        self.pushButton_load_positions.setObjectName("pushButton_load_positions")
        self.gridLayout.addWidget(self.pushButton_load_positions, 15, 0, 1, 1)
        self.pushButton_save_positions = QtWidgets.QPushButton(Form)
        self.pushButton_save_positions.setObjectName("pushButton_save_positions")
        self.gridLayout.addWidget(self.pushButton_save_positions, 15, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_tile_overlap.setText(_translate("Form", "Overlap (%)"))
        self.pushButton_add_position.setText(_translate("Form", "Add Position"))
        self.label_tile_tile_size.setText(_translate("Form", "Tile Size (um)"))
        self.pushButton_load_image.setText(_translate("Form", "Load Image"))
        self.label_tile_label.setText(_translate("Form", "Tile Filename"))
        self.label_instructions.setText(_translate("Form", "Please take or load an overview image."))
        self.label_tile_grid_size.setText(_translate("Form", "Grid Size (um)"))
        self.pushButton_remove_position.setText(_translate("Form", "Remove Position"))
        self.label_position_info.setText(_translate("Form", "No Positions saved."))
        self.label_position_header.setText(_translate("Form", "Saved Positions"))
        self.pushButton_move_to_position.setText(_translate("Form", "Move to Position"))
        self.pushButton_run_tile_collection.setText(_translate("Form", "Run Tile Collection"))
        self.lineEdit_tile_label.setText(_translate("Form", "stitched-image"))
        self.label_title.setText(_translate("Form", "Minimap "))
        self.label_tile_beam_type.setText(_translate("Form", "Beam Type"))
        self.pushButton_load_positions.setText(_translate("Form", "Load Positions from File"))
        self.pushButton_save_positions.setText(_translate("Form", "Save Positions to File"))
