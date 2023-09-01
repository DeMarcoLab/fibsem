# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemMovementWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(477, 540)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_movement = QtWidgets.QWidget()
        self.tab_movement.setObjectName("tab_movement")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_movement)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_movement_stage_coordinate_system = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_stage_coordinate_system.setObjectName("label_movement_stage_coordinate_system")
        self.gridLayout_2.addWidget(self.label_movement_stage_coordinate_system, 6, 0, 1, 1)
        self.doubleSpinBox_movement_stage_tilt = QtWidgets.QDoubleSpinBox(self.tab_movement)
        self.doubleSpinBox_movement_stage_tilt.setObjectName("doubleSpinBox_movement_stage_tilt")
        self.gridLayout_2.addWidget(self.doubleSpinBox_movement_stage_tilt, 5, 1, 1, 1)
        self.pushButton_move_flat_ion = QtWidgets.QPushButton(self.tab_movement)
        self.pushButton_move_flat_ion.setObjectName("pushButton_move_flat_ion")
        self.gridLayout_2.addWidget(self.pushButton_move_flat_ion, 8, 1, 1, 1)
        self.doubleSpinBox_movement_stage_y = QtWidgets.QDoubleSpinBox(self.tab_movement)
        self.doubleSpinBox_movement_stage_y.setDecimals(5)
        self.doubleSpinBox_movement_stage_y.setMinimum(-1e+20)
        self.doubleSpinBox_movement_stage_y.setMaximum(1e+25)
        self.doubleSpinBox_movement_stage_y.setObjectName("doubleSpinBox_movement_stage_y")
        self.gridLayout_2.addWidget(self.doubleSpinBox_movement_stage_y, 2, 1, 1, 1)
        self.pushButton_move = QtWidgets.QPushButton(self.tab_movement)
        self.pushButton_move.setObjectName("pushButton_move")
        self.gridLayout_2.addWidget(self.pushButton_move, 9, 0, 1, 2)
        self.pushButton_continue = QtWidgets.QPushButton(self.tab_movement)
        self.pushButton_continue.setObjectName("pushButton_continue")
        self.gridLayout_2.addWidget(self.pushButton_continue, 13, 0, 1, 2)
        self.doubleSpinBox_movement_stage_x = QtWidgets.QDoubleSpinBox(self.tab_movement)
        self.doubleSpinBox_movement_stage_x.setDecimals(5)
        self.doubleSpinBox_movement_stage_x.setMinimum(-10000000000.0)
        self.doubleSpinBox_movement_stage_x.setMaximum(1e+17)
        self.doubleSpinBox_movement_stage_x.setObjectName("doubleSpinBox_movement_stage_x")
        self.gridLayout_2.addWidget(self.doubleSpinBox_movement_stage_x, 1, 1, 1, 1)
        self.label_movement_images = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_images.setObjectName("label_movement_images")
        self.gridLayout_2.addWidget(self.label_movement_images, 10, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 14, 0, 1, 2)
        self.comboBox_movement_mode = QtWidgets.QComboBox(self.tab_movement)
        self.comboBox_movement_mode.setObjectName("comboBox_movement_mode")
        self.gridLayout_2.addWidget(self.comboBox_movement_mode, 0, 1, 1, 1)
        self.label_movement_stage_rotation = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_stage_rotation.setObjectName("label_movement_stage_rotation")
        self.gridLayout_2.addWidget(self.label_movement_stage_rotation, 4, 0, 1, 1)
        self.checkBox_movement_acquire_electron = QtWidgets.QCheckBox(self.tab_movement)
        self.checkBox_movement_acquire_electron.setChecked(True)
        self.checkBox_movement_acquire_electron.setObjectName("checkBox_movement_acquire_electron")
        self.gridLayout_2.addWidget(self.checkBox_movement_acquire_electron, 11, 0, 1, 1)
        self.label_movement_stage_y = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_stage_y.setObjectName("label_movement_stage_y")
        self.gridLayout_2.addWidget(self.label_movement_stage_y, 2, 0, 1, 1)
        self.checkBox_movement_acquire_ion = QtWidgets.QCheckBox(self.tab_movement)
        self.checkBox_movement_acquire_ion.setChecked(True)
        self.checkBox_movement_acquire_ion.setObjectName("checkBox_movement_acquire_ion")
        self.gridLayout_2.addWidget(self.checkBox_movement_acquire_ion, 11, 1, 1, 1)
        self.label_movement_stage_x = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_stage_x.setObjectName("label_movement_stage_x")
        self.gridLayout_2.addWidget(self.label_movement_stage_x, 1, 0, 1, 1)
        self.pushButton_move_flat_electron = QtWidgets.QPushButton(self.tab_movement)
        self.pushButton_move_flat_electron.setObjectName("pushButton_move_flat_electron")
        self.gridLayout_2.addWidget(self.pushButton_move_flat_electron, 8, 0, 1, 1)
        self.label_movement_instructions = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_instructions.setMinimumSize(QtCore.QSize(0, 50))
        self.label_movement_instructions.setWordWrap(True)
        self.label_movement_instructions.setObjectName("label_movement_instructions")
        self.gridLayout_2.addWidget(self.label_movement_instructions, 12, 0, 1, 2)
        self.label_movement_mode = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_mode.setObjectName("label_movement_mode")
        self.gridLayout_2.addWidget(self.label_movement_mode, 0, 0, 1, 1)
        self.label_movement_stage_tilt = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_stage_tilt.setObjectName("label_movement_stage_tilt")
        self.gridLayout_2.addWidget(self.label_movement_stage_tilt, 5, 0, 1, 1)
        self.doubleSpinBox_movement_stage_z = QtWidgets.QDoubleSpinBox(self.tab_movement)
        self.doubleSpinBox_movement_stage_z.setDecimals(5)
        self.doubleSpinBox_movement_stage_z.setMinimum(-1e+17)
        self.doubleSpinBox_movement_stage_z.setMaximum(1e+23)
        self.doubleSpinBox_movement_stage_z.setObjectName("doubleSpinBox_movement_stage_z")
        self.gridLayout_2.addWidget(self.doubleSpinBox_movement_stage_z, 3, 1, 1, 1)
        self.label_movement_stage_z = QtWidgets.QLabel(self.tab_movement)
        self.label_movement_stage_z.setObjectName("label_movement_stage_z")
        self.gridLayout_2.addWidget(self.label_movement_stage_z, 3, 0, 1, 1)
        self.comboBox_movement_stage_coordinate_system = QtWidgets.QComboBox(self.tab_movement)
        self.comboBox_movement_stage_coordinate_system.setObjectName("comboBox_movement_stage_coordinate_system")
        self.gridLayout_2.addWidget(self.comboBox_movement_stage_coordinate_system, 6, 1, 1, 1)
        self.doubleSpinBox_movement_stage_rotation = QtWidgets.QDoubleSpinBox(self.tab_movement)
        self.doubleSpinBox_movement_stage_rotation.setMinimum(-360.0)
        self.doubleSpinBox_movement_stage_rotation.setMaximum(360.0)
        self.doubleSpinBox_movement_stage_rotation.setObjectName("doubleSpinBox_movement_stage_rotation")
        self.gridLayout_2.addWidget(self.doubleSpinBox_movement_stage_rotation, 4, 1, 1, 1)
        self.tabWidget.addTab(self.tab_movement, "")
        self.tab_positions = QtWidgets.QWidget()
        self.tab_positions.setObjectName("tab_positions")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_positions)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_2 = QtWidgets.QLabel(self.tab_positions)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.tab_positions)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 1, 1, 1)
        self.pushButton_load_image_minimap = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_load_image_minimap.setObjectName("pushButton_load_image_minimap")
        self.gridLayout_3.addWidget(self.pushButton_load_image_minimap, 7, 2, 1, 1)
        self.pushButton_update_position = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_update_position.setObjectName("pushButton_update_position")
        self.gridLayout_3.addWidget(self.pushButton_update_position, 5, 2, 1, 1)
        self.label_minimap = QtWidgets.QLabel(self.tab_positions)
        self.label_minimap.setText("")
        self.label_minimap.setObjectName("label_minimap")
        self.gridLayout_3.addWidget(self.label_minimap, 9, 1, 1, 3)
        self.comboBox_positions = QtWidgets.QComboBox(self.tab_positions)
        self.comboBox_positions.setObjectName("comboBox_positions")
        self.gridLayout_3.addWidget(self.comboBox_positions, 2, 2, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.tab_positions)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 1, 1, 1)
        self.pushButton_import = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_import.setObjectName("pushButton_import")
        self.gridLayout_3.addWidget(self.pushButton_import, 6, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem1, 10, 1, 1, 3)
        self.lineEdit_position_name = QtWidgets.QLineEdit(self.tab_positions)
        self.lineEdit_position_name.setObjectName("lineEdit_position_name")
        self.gridLayout_3.addWidget(self.lineEdit_position_name, 4, 2, 1, 2)
        self.pushButton_remove_position = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_remove_position.setStyleSheet("color: white; background-color: rgb(170, 0, 0)")
        self.pushButton_remove_position.setObjectName("pushButton_remove_position")
        self.gridLayout_3.addWidget(self.pushButton_remove_position, 5, 3, 1, 1)
        self.label_current_position = QtWidgets.QLabel(self.tab_positions)
        self.label_current_position.setText("")
        self.label_current_position.setObjectName("label_current_position")
        self.gridLayout_3.addWidget(self.label_current_position, 3, 2, 1, 2)
        self.pushButton_export = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_export.setObjectName("pushButton_export")
        self.gridLayout_3.addWidget(self.pushButton_export, 6, 3, 1, 1)
        self.pushButton_go_to = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_go_to.setObjectName("pushButton_go_to")
        self.gridLayout_3.addWidget(self.pushButton_go_to, 3, 1, 1, 1)
        self.pushButton_save_position = QtWidgets.QPushButton(self.tab_positions)
        self.pushButton_save_position.setStyleSheet("color: white; background-color: rgb(85, 170, 0)")
        self.pushButton_save_position.setObjectName("pushButton_save_position")
        self.gridLayout_3.addWidget(self.pushButton_save_position, 4, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab_positions)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 8, 1, 1, 3)
        self.tabWidget.addTab(self.tab_positions, "")
        self.gridLayout.addWidget(self.tabWidget, 2, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.tabWidget, self.comboBox_movement_mode)
        Form.setTabOrder(self.comboBox_movement_mode, self.doubleSpinBox_movement_stage_x)
        Form.setTabOrder(self.doubleSpinBox_movement_stage_x, self.doubleSpinBox_movement_stage_y)
        Form.setTabOrder(self.doubleSpinBox_movement_stage_y, self.doubleSpinBox_movement_stage_z)
        Form.setTabOrder(self.doubleSpinBox_movement_stage_z, self.doubleSpinBox_movement_stage_rotation)
        Form.setTabOrder(self.doubleSpinBox_movement_stage_rotation, self.doubleSpinBox_movement_stage_tilt)
        Form.setTabOrder(self.doubleSpinBox_movement_stage_tilt, self.comboBox_movement_stage_coordinate_system)
        Form.setTabOrder(self.comboBox_movement_stage_coordinate_system, self.pushButton_move_flat_electron)
        Form.setTabOrder(self.pushButton_move_flat_electron, self.pushButton_move_flat_ion)
        Form.setTabOrder(self.pushButton_move_flat_ion, self.pushButton_move)
        Form.setTabOrder(self.pushButton_move, self.checkBox_movement_acquire_electron)
        Form.setTabOrder(self.checkBox_movement_acquire_electron, self.checkBox_movement_acquire_ion)
        Form.setTabOrder(self.checkBox_movement_acquire_ion, self.pushButton_continue)
        Form.setTabOrder(self.pushButton_continue, self.comboBox_positions)
        Form.setTabOrder(self.comboBox_positions, self.pushButton_go_to)
        Form.setTabOrder(self.pushButton_go_to, self.pushButton_save_position)
        Form.setTabOrder(self.pushButton_save_position, self.lineEdit_position_name)
        Form.setTabOrder(self.lineEdit_position_name, self.pushButton_update_position)
        Form.setTabOrder(self.pushButton_update_position, self.pushButton_remove_position)
        Form.setTabOrder(self.pushButton_remove_position, self.pushButton_import)
        Form.setTabOrder(self.pushButton_import, self.pushButton_export)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_movement_stage_coordinate_system.setText(_translate("Form", "Coordinate System"))
        self.pushButton_move_flat_ion.setText(_translate("Form", "Move Flat to ION Beam"))
        self.pushButton_move.setText(_translate("Form", "Move to Position"))
        self.pushButton_continue.setText(_translate("Form", "Continue"))
        self.label_movement_images.setText(_translate("Form", "Acquire images after moving:"))
        self.label_movement_stage_rotation.setText(_translate("Form", "Rotation (deg)"))
        self.checkBox_movement_acquire_electron.setText(_translate("Form", "Electron Beam"))
        self.label_movement_stage_y.setText(_translate("Form", "Y Coordinate (mm)"))
        self.checkBox_movement_acquire_ion.setText(_translate("Form", "Ion Beam"))
        self.label_movement_stage_x.setText(_translate("Form", "X Coordinate (mm)"))
        self.pushButton_move_flat_electron.setText(_translate("Form", "Move Flat to ELECTRON Beam"))
        self.label_movement_instructions.setText(_translate("Form", "Double click to move. Press continue when complete."))
        self.label_movement_mode.setText(_translate("Form", "Mode"))
        self.label_movement_stage_tilt.setText(_translate("Form", "Tilt (deg)"))
        self.label_movement_stage_z.setText(_translate("Form", "Z Coordinate (mm)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_movement), _translate("Form", "Movement"))
        self.label_2.setText(_translate("Form", "Saved positions"))
        self.label.setText(_translate("Form", "Positions "))
        self.pushButton_load_image_minimap.setText(_translate("Form", "Load Minimap Image"))
        self.pushButton_update_position.setText(_translate("Form", "Update position"))
        self.label_3.setText(_translate("Form", "All positions in mm and degrees"))
        self.pushButton_import.setText(_translate("Form", "Import Positions"))
        self.pushButton_remove_position.setText(_translate("Form", "Remove Position"))
        self.pushButton_export.setText(_translate("Form", "Export positions"))
        self.pushButton_go_to.setText(_translate("Form", "Go To"))
        self.pushButton_save_position.setText(_translate("Form", "Save position"))
        self.label_4.setText(_translate("Form", "Please load grid image to display minimap "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_positions), _translate("Form", "Positions"))
