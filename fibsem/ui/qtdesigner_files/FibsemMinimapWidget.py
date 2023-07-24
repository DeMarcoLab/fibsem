# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemMinimapWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(525, 637)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_tile_tile_size = QtWidgets.QLabel(self.tab)
        self.label_tile_tile_size.setObjectName("label_tile_tile_size")
        self.gridLayout_2.addWidget(self.label_tile_tile_size, 2, 0, 1, 1)
        self.label_tile_beam_type = QtWidgets.QLabel(self.tab)
        self.label_tile_beam_type.setObjectName("label_tile_beam_type")
        self.gridLayout_2.addWidget(self.label_tile_beam_type, 0, 0, 1, 1)
        self.label_tile_label = QtWidgets.QLabel(self.tab)
        self.label_tile_label.setObjectName("label_tile_label")
        self.gridLayout_2.addWidget(self.label_tile_label, 4, 0, 1, 1)
        self.label_tile_overlap = QtWidgets.QLabel(self.tab)
        self.label_tile_overlap.setObjectName("label_tile_overlap")
        self.gridLayout_2.addWidget(self.label_tile_overlap, 3, 0, 1, 1)
        self.doubleSpinBox_tile_grid_size = QtWidgets.QDoubleSpinBox(self.tab)
        self.doubleSpinBox_tile_grid_size.setMaximum(1000000.0)
        self.doubleSpinBox_tile_grid_size.setProperty("value", 400.0)
        self.doubleSpinBox_tile_grid_size.setObjectName("doubleSpinBox_tile_grid_size")
        self.gridLayout_2.addWidget(self.doubleSpinBox_tile_grid_size, 1, 1, 1, 1)
        self.lineEdit_tile_label = QtWidgets.QLineEdit(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_tile_label.sizePolicy().hasHeightForWidth())
        self.lineEdit_tile_label.setSizePolicy(sizePolicy)
        self.lineEdit_tile_label.setObjectName("lineEdit_tile_label")
        self.gridLayout_2.addWidget(self.lineEdit_tile_label, 4, 1, 1, 1)
        self.doubleSpinBox_tile_tile_size = QtWidgets.QDoubleSpinBox(self.tab)
        self.doubleSpinBox_tile_tile_size.setMaximum(100000.0)
        self.doubleSpinBox_tile_tile_size.setProperty("value", 100.0)
        self.doubleSpinBox_tile_tile_size.setObjectName("doubleSpinBox_tile_tile_size")
        self.gridLayout_2.addWidget(self.doubleSpinBox_tile_tile_size, 2, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 6, 0, 1, 2)
        self.label_tile_grid_size = QtWidgets.QLabel(self.tab)
        self.label_tile_grid_size.setObjectName("label_tile_grid_size")
        self.gridLayout_2.addWidget(self.label_tile_grid_size, 1, 0, 1, 1)
        self.comboBox_tile_beam_type = QtWidgets.QComboBox(self.tab)
        self.comboBox_tile_beam_type.setObjectName("comboBox_tile_beam_type")
        self.gridLayout_2.addWidget(self.comboBox_tile_beam_type, 0, 1, 1, 1)
        self.pushButton_run_tile_collection = QtWidgets.QPushButton(self.tab)
        self.pushButton_run_tile_collection.setObjectName("pushButton_run_tile_collection")
        self.gridLayout_2.addWidget(self.pushButton_run_tile_collection, 5, 0, 1, 2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_position_header = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_position_header.setFont(font)
        self.label_position_header.setObjectName("label_position_header")
        self.gridLayout_4.addWidget(self.label_position_header, 0, 0, 1, 1)
        self.label_position_info = QtWidgets.QLabel(self.tab_4)
        self.label_position_info.setObjectName("label_position_info")
        self.gridLayout_4.addWidget(self.label_position_info, 3, 0, 1, 2)
        self.pushButton_move_to_position = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_move_to_position.setObjectName("pushButton_move_to_position")
        self.gridLayout_4.addWidget(self.pushButton_move_to_position, 4, 0, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(483, 269, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem1, 6, 0, 1, 2)
        self.comboBox_tile_position = QtWidgets.QComboBox(self.tab_4)
        self.comboBox_tile_position.setObjectName("comboBox_tile_position")
        self.gridLayout_4.addWidget(self.comboBox_tile_position, 0, 1, 1, 1)
        self.label_instructions = QtWidgets.QLabel(self.tab_4)
        self.label_instructions.setObjectName("label_instructions")
        self.gridLayout_4.addWidget(self.label_instructions, 5, 0, 1, 2)
        self.lineEdit_tile_position_name = QtWidgets.QLineEdit(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_tile_position_name.sizePolicy().hasHeightForWidth())
        self.lineEdit_tile_position_name.setSizePolicy(sizePolicy)
        self.lineEdit_tile_position_name.setObjectName("lineEdit_tile_position_name")
        self.gridLayout_4.addWidget(self.lineEdit_tile_position_name, 1, 1, 1, 1)
        self.pushButton_remove_position = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_remove_position.setObjectName("pushButton_remove_position")
        self.gridLayout_4.addWidget(self.pushButton_remove_position, 2, 1, 1, 1)
        self.label_position_name = QtWidgets.QLabel(self.tab_4)
        self.label_position_name.setObjectName("label_position_name")
        self.gridLayout_4.addWidget(self.label_position_name, 1, 0, 1, 1)
        self.pushButton_update_position = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_update_position.setObjectName("pushButton_update_position")
        self.gridLayout_4.addWidget(self.pushButton_update_position, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(483, 261, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem2, 8, 0, 1, 2)
        self.label_correlation_translation_x = QtWidgets.QLabel(self.tab_2)
        self.label_correlation_translation_x.setObjectName("label_correlation_translation_x")
        self.gridLayout_3.addWidget(self.label_correlation_translation_x, 2, 0, 1, 1)
        self.doubleSpinBox_correlation_translation_y = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_correlation_translation_y.setDecimals(0)
        self.doubleSpinBox_correlation_translation_y.setMinimum(-100000000.0)
        self.doubleSpinBox_correlation_translation_y.setMaximum(1000000000.0)
        self.doubleSpinBox_correlation_translation_y.setObjectName("doubleSpinBox_correlation_translation_y")
        self.gridLayout_3.addWidget(self.doubleSpinBox_correlation_translation_y, 3, 1, 1, 1)
        self.label_correlation_selected_layer = QtWidgets.QLabel(self.tab_2)
        self.label_correlation_selected_layer.setObjectName("label_correlation_selected_layer")
        self.gridLayout_3.addWidget(self.label_correlation_selected_layer, 1, 0, 1, 1)
        self.label_correlation_scale_y = QtWidgets.QLabel(self.tab_2)
        self.label_correlation_scale_y.setObjectName("label_correlation_scale_y")
        self.gridLayout_3.addWidget(self.label_correlation_scale_y, 5, 0, 1, 1)
        self.comboBox_correlation_selected_layer = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_correlation_selected_layer.setObjectName("comboBox_correlation_selected_layer")
        self.gridLayout_3.addWidget(self.comboBox_correlation_selected_layer, 1, 1, 1, 1)
        self.doubleSpinBox_correlation_scale_x = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_correlation_scale_x.setDecimals(3)
        self.doubleSpinBox_correlation_scale_x.setMinimum(-100000000.0)
        self.doubleSpinBox_correlation_scale_x.setMaximum(1000000000.0)
        self.doubleSpinBox_correlation_scale_x.setProperty("value", 1.0)
        self.doubleSpinBox_correlation_scale_x.setObjectName("doubleSpinBox_correlation_scale_x")
        self.gridLayout_3.addWidget(self.doubleSpinBox_correlation_scale_x, 4, 1, 1, 1)
        self.doubleSpinBox_correlation_rotation = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_correlation_rotation.setDecimals(2)
        self.doubleSpinBox_correlation_rotation.setMinimum(-360.0)
        self.doubleSpinBox_correlation_rotation.setMaximum(360.0)
        self.doubleSpinBox_correlation_rotation.setObjectName("doubleSpinBox_correlation_rotation")
        self.gridLayout_3.addWidget(self.doubleSpinBox_correlation_rotation, 6, 1, 1, 1)
        self.doubleSpinBox_correlation_translation_x = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_correlation_translation_x.setDecimals(0)
        self.doubleSpinBox_correlation_translation_x.setMinimum(-100000000.0)
        self.doubleSpinBox_correlation_translation_x.setMaximum(1000000000.0)
        self.doubleSpinBox_correlation_translation_x.setObjectName("doubleSpinBox_correlation_translation_x")
        self.gridLayout_3.addWidget(self.doubleSpinBox_correlation_translation_x, 2, 1, 1, 1)
        self.label_correlation_rotation = QtWidgets.QLabel(self.tab_2)
        self.label_correlation_rotation.setObjectName("label_correlation_rotation")
        self.gridLayout_3.addWidget(self.label_correlation_rotation, 6, 0, 1, 1)
        self.label_correlation_translation_y = QtWidgets.QLabel(self.tab_2)
        self.label_correlation_translation_y.setObjectName("label_correlation_translation_y")
        self.gridLayout_3.addWidget(self.label_correlation_translation_y, 3, 0, 1, 1)
        self.doubleSpinBox_correlation_scale_y = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_correlation_scale_y.setDecimals(3)
        self.doubleSpinBox_correlation_scale_y.setMinimum(-100000000.0)
        self.doubleSpinBox_correlation_scale_y.setMaximum(1000000000.0)
        self.doubleSpinBox_correlation_scale_y.setProperty("value", 1.0)
        self.doubleSpinBox_correlation_scale_y.setObjectName("doubleSpinBox_correlation_scale_y")
        self.gridLayout_3.addWidget(self.doubleSpinBox_correlation_scale_y, 5, 1, 1, 1)
        self.label_correlation_scale_x = QtWidgets.QLabel(self.tab_2)
        self.label_correlation_scale_x.setObjectName("label_correlation_scale_x")
        self.gridLayout_3.addWidget(self.label_correlation_scale_x, 4, 0, 1, 1)
        self.pushButton_update_correlation_image = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_update_correlation_image.setObjectName("pushButton_update_correlation_image")
        self.gridLayout_3.addWidget(self.pushButton_update_correlation_image, 7, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 525, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Image = QtWidgets.QAction(MainWindow)
        self.actionLoad_Image.setObjectName("actionLoad_Image")
        self.actionLoad_Correlation_Image = QtWidgets.QAction(MainWindow)
        self.actionLoad_Correlation_Image.setObjectName("actionLoad_Correlation_Image")
        self.actionSave_Positions = QtWidgets.QAction(MainWindow)
        self.actionSave_Positions.setObjectName("actionSave_Positions")
        self.actionLoad_Positions = QtWidgets.QAction(MainWindow)
        self.actionLoad_Positions.setObjectName("actionLoad_Positions")
        self.menuFile.addAction(self.actionLoad_Image)
        self.menuFile.addAction(self.actionLoad_Correlation_Image)
        self.menuFile.addAction(self.actionSave_Positions)
        self.menuFile.addAction(self.actionLoad_Positions)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_title.setText(_translate("MainWindow", "Minimap "))
        self.label_tile_tile_size.setText(_translate("MainWindow", "Tile Size (um)"))
        self.label_tile_beam_type.setText(_translate("MainWindow", "Beam Type"))
        self.label_tile_label.setText(_translate("MainWindow", "Tile Filename"))
        self.label_tile_overlap.setText(_translate("MainWindow", "Overlap (%)"))
        self.lineEdit_tile_label.setText(_translate("MainWindow", "stitched-image"))
        self.label_tile_grid_size.setText(_translate("MainWindow", "Grid Size (um)"))
        self.pushButton_run_tile_collection.setText(_translate("MainWindow", "Run Tile Collection"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Collection"))
        self.label_position_header.setText(_translate("MainWindow", "Saved Positions"))
        self.label_position_info.setText(_translate("MainWindow", "No Positions saved."))
        self.pushButton_move_to_position.setText(_translate("MainWindow", "Move to Position"))
        self.label_instructions.setText(_translate("MainWindow", "Please take or load an overview image."))
        self.pushButton_remove_position.setText(_translate("MainWindow", "Remove Position"))
        self.label_position_name.setText(_translate("MainWindow", "Position Name"))
        self.pushButton_update_position.setText(_translate("MainWindow", "Update Position"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Positions"))
        self.label_correlation_translation_x.setText(_translate("MainWindow", "Translation X (px)"))
        self.label_correlation_selected_layer.setText(_translate("MainWindow", "Selected Layer"))
        self.label_correlation_scale_y.setText(_translate("MainWindow", "Scale Y (%)"))
        self.label_correlation_rotation.setText(_translate("MainWindow", "Rotation (deg)"))
        self.label_correlation_translation_y.setText(_translate("MainWindow", "Translation Y (px)"))
        self.label_correlation_scale_x.setText(_translate("MainWindow", "Scale (X) (%)"))
        self.pushButton_update_correlation_image.setText(_translate("MainWindow", "Update Correlation Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Correlation"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_Image.setText(_translate("MainWindow", "Load Image"))
        self.actionLoad_Correlation_Image.setText(_translate("MainWindow", "Load Correlation Image"))
        self.actionSave_Positions.setText(_translate("MainWindow", "Save Positions"))
        self.actionLoad_Positions.setText(_translate("MainWindow", "Load Positions"))
