# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemLabellingUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 469)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_model_info = QtWidgets.QLabel(Dialog)
        self.label_model_info.setObjectName("label_model_info")
        self.gridLayout.addWidget(self.label_model_info, 2, 1, 1, 1)
        self.pushButton_next = QtWidgets.QPushButton(Dialog)
        self.pushButton_next.setObjectName("pushButton_next")
        self.gridLayout.addWidget(self.pushButton_next, 14, 1, 1, 1)
        self.comboBox_model_class_index = QtWidgets.QComboBox(Dialog)
        self.comboBox_model_class_index.setObjectName("comboBox_model_class_index")
        self.gridLayout.addWidget(self.comboBox_model_class_index, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 10, 0, 1, 1)
        self.pushButton_model_clear = QtWidgets.QPushButton(Dialog)
        self.pushButton_model_clear.setObjectName("pushButton_model_clear")
        self.gridLayout.addWidget(self.pushButton_model_clear, 4, 1, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_raw_data = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_raw_data.setObjectName("lineEdit_raw_data")
        self.gridLayout_2.addWidget(self.lineEdit_raw_data, 0, 1, 1, 1)
        self.lineEdit_save_path = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_save_path.setObjectName("lineEdit_save_path")
        self.gridLayout_2.addWidget(self.lineEdit_save_path, 1, 1, 1, 1)
        self.label_raw_data = QtWidgets.QLabel(self.tab)
        self.label_raw_data.setObjectName("label_raw_data")
        self.gridLayout_2.addWidget(self.label_raw_data, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem1, 4, 0, 1, 2)
        self.label_save_path = QtWidgets.QLabel(self.tab)
        self.label_save_path.setObjectName("label_save_path")
        self.gridLayout_2.addWidget(self.label_save_path, 1, 0, 1, 1)
        self.label_num_classes = QtWidgets.QLabel(self.tab)
        self.label_num_classes.setObjectName("label_num_classes")
        self.gridLayout_2.addWidget(self.label_num_classes, 2, 0, 1, 1)
        self.pushButton_load_data = QtWidgets.QPushButton(self.tab)
        self.pushButton_load_data.setObjectName("pushButton_load_data")
        self.gridLayout_2.addWidget(self.pushButton_load_data, 3, 0, 1, 2)
        self.spinBox_num_classes = QtWidgets.QSpinBox(self.tab)
        self.spinBox_num_classes.setMaximum(255)
        self.spinBox_num_classes.setProperty("value", 1)
        self.spinBox_num_classes.setObjectName("spinBox_num_classes")
        self.gridLayout_2.addWidget(self.spinBox_num_classes, 2, 1, 1, 1)
        self.rawData_button = QtWidgets.QToolButton(self.tab)
        self.rawData_button.setObjectName("rawData_button")
        self.gridLayout_2.addWidget(self.rawData_button, 0, 2, 1, 1)
        self.savePath_button = QtWidgets.QToolButton(self.tab)
        self.savePath_button.setObjectName("savePath_button")
        self.gridLayout_2.addWidget(self.savePath_button, 1, 2, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 2)
        self.label_title = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 16, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 17, 0, 1, 2)
        self.pushButton_previous = QtWidgets.QPushButton(Dialog)
        self.pushButton_previous.setObjectName("pushButton_previous")
        self.gridLayout.addWidget(self.pushButton_previous, 14, 0, 1, 1)
        self.pushButton_model_confirm = QtWidgets.QPushButton(Dialog)
        self.pushButton_model_confirm.setObjectName("pushButton_model_confirm")
        self.gridLayout.addWidget(self.pushButton_model_confirm, 4, 0, 1, 1)
        self.checkBox_model_assist = QtWidgets.QCheckBox(Dialog)
        self.checkBox_model_assist.setObjectName("checkBox_model_assist")
        self.gridLayout.addWidget(self.checkBox_model_assist, 2, 0, 1, 1)
        self.label_model_class_index = QtWidgets.QLabel(Dialog)
        self.label_model_class_index.setObjectName("label_model_class_index")
        self.gridLayout.addWidget(self.label_model_class_index, 3, 0, 1, 1)
        self.label_progress = QtWidgets.QLabel(Dialog)
        self.label_progress.setAlignment(QtCore.Qt.AlignCenter)
        self.label_progress.setObjectName("label_progress")
        self.gridLayout.addWidget(self.label_progress, 12, 0, 1, 2)
        self.label_instructions = QtWidgets.QLabel(Dialog)
        self.label_instructions.setObjectName("label_instructions")
        self.gridLayout.addWidget(self.label_instructions, 11, 0, 1, 2)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_model_info.setText(_translate("Dialog", "No Model"))
        self.pushButton_next.setText(_translate("Dialog", "Next"))
        self.pushButton_model_clear.setText(_translate("Dialog", "Clear"))
        self.label_raw_data.setText(_translate("Dialog", "Raw Data"))
        self.label_save_path.setText(_translate("Dialog", "Save Path"))
        self.label_num_classes.setText(_translate("Dialog", "Num Classes"))
        self.pushButton_load_data.setText(_translate("Dialog", "Load Data"))
        self.rawData_button.setText(_translate("Dialog", "..."))
        self.savePath_button.setText(_translate("Dialog", "..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Data"))
        self.label_title.setText(_translate("Dialog", "FIBSEM Labelling"))
        self.pushButton_previous.setText(_translate("Dialog", "Previous"))
        self.pushButton_model_confirm.setText(_translate("Dialog", "Confirm"))
        self.checkBox_model_assist.setText(_translate("Dialog", "Model Assisted"))
        self.label_model_class_index.setText(_translate("Dialog", "Class Index"))
        self.label_progress.setText(_translate("Dialog", "Progress"))
        self.label_instructions.setText(_translate("Dialog", "TextLabel"))
