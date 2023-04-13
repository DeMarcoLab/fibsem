# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemDetectionWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(467, 646)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label_feature = QtWidgets.QLabel(Form)
        self.label_feature.setObjectName("label_feature")
        self.gridLayout.addWidget(self.label_feature, 12, 0, 1, 1)
        self.label_info = QtWidgets.QLabel(Form)
        self.label_info.setMinimumSize(QtCore.QSize(0, 150))
        self.label_info.setText("")
        self.label_info.setObjectName("label_info")
        self.gridLayout.addWidget(self.label_info, 14, 0, 1, 3)
        self.label_instructions = QtWidgets.QLabel(Form)
        self.label_instructions.setWordWrap(True)
        self.label_instructions.setObjectName("label_instructions")
        self.gridLayout.addWidget(self.label_instructions, 16, 0, 1, 3)
        self.pushButton_continue = QtWidgets.QPushButton(Form)
        self.pushButton_continue.setObjectName("pushButton_continue")
        self.gridLayout.addWidget(self.pushButton_continue, 17, 0, 1, 3)
        self.label_2 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 10, 0, 1, 3)
        self.label_encoder = QtWidgets.QLabel(Form)
        self.label_encoder.setObjectName("label_encoder")
        self.gridLayout.addWidget(self.label_encoder, 6, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.spinBox_num_classes = QtWidgets.QSpinBox(Form)
        self.spinBox_num_classes.setObjectName("spinBox_num_classes")
        self.gridLayout.addWidget(self.spinBox_num_classes, 8, 1, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 19, 0, 1, 2)
        self.pushButton_test_function = QtWidgets.QPushButton(Form)
        self.pushButton_test_function.setObjectName("pushButton_test_function")
        self.gridLayout.addWidget(self.pushButton_test_function, 18, 0, 1, 1)
        self.comboBox_feature_1 = QtWidgets.QComboBox(Form)
        self.comboBox_feature_1.setObjectName("comboBox_feature_1")
        self.gridLayout.addWidget(self.comboBox_feature_1, 12, 1, 1, 1)
        self.pushButton_load_model = QtWidgets.QPushButton(Form)
        self.pushButton_load_model.setObjectName("pushButton_load_model")
        self.gridLayout.addWidget(self.pushButton_load_model, 9, 0, 1, 3)
        self.checkBox_show_mask = QtWidgets.QCheckBox(Form)
        self.checkBox_show_mask.setObjectName("checkBox_show_mask")
        self.gridLayout.addWidget(self.checkBox_show_mask, 13, 0, 1, 1)
        self.label_checkpoint = QtWidgets.QLabel(Form)
        self.label_checkpoint.setObjectName("label_checkpoint")
        self.gridLayout.addWidget(self.label_checkpoint, 7, 0, 1, 1)
        self.pushButton_run_feature_detection = QtWidgets.QPushButton(Form)
        self.pushButton_run_feature_detection.setObjectName("pushButton_run_feature_detection")
        self.gridLayout.addWidget(self.pushButton_run_feature_detection, 15, 0, 1, 3)
        self.lineEdit_encoder = QtWidgets.QLineEdit(Form)
        self.lineEdit_encoder.setObjectName("lineEdit_encoder")
        self.gridLayout.addWidget(self.lineEdit_encoder, 6, 1, 1, 2)
        self.comboBox_feature_2 = QtWidgets.QComboBox(Form)
        self.comboBox_feature_2.setObjectName("comboBox_feature_2")
        self.gridLayout.addWidget(self.comboBox_feature_2, 12, 2, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_num_classes = QtWidgets.QLabel(Form)
        self.label_num_classes.setObjectName("label_num_classes")
        self.gridLayout.addWidget(self.label_num_classes, 8, 0, 1, 1)
        self.lineEdit_checkpoint = QtWidgets.QLineEdit(Form)
        self.lineEdit_checkpoint.setObjectName("lineEdit_checkpoint")
        self.gridLayout.addWidget(self.lineEdit_checkpoint, 7, 1, 1, 2)
        self.label_beam_type = QtWidgets.QLabel(Form)
        self.label_beam_type.setObjectName("label_beam_type")
        self.gridLayout.addWidget(self.label_beam_type, 11, 0, 1, 1)
        self.comboBox_beam_type = QtWidgets.QComboBox(Form)
        self.comboBox_beam_type.setObjectName("comboBox_beam_type")
        self.gridLayout.addWidget(self.comboBox_beam_type, 11, 1, 1, 2)
        self.checkBox_move_features = QtWidgets.QCheckBox(Form)
        self.checkBox_move_features.setObjectName("checkBox_move_features")
        self.gridLayout.addWidget(self.checkBox_move_features, 13, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_feature.setText(_translate("Form", "Features"))
        self.label_instructions.setText(_translate("Form", "Select the feature, and click to change the feature position."))
        self.pushButton_continue.setText(_translate("Form", "Continue"))
        self.label_2.setText(_translate("Form", "Feature Detection"))
        self.label_encoder.setText(_translate("Form", "Encoder"))
        self.label_title.setText(_translate("Form", "Feature Detection"))
        self.pushButton_test_function.setText(_translate("Form", "Test 1"))
        self.pushButton_load_model.setText(_translate("Form", "Load Model"))
        self.checkBox_show_mask.setText(_translate("Form", "Show Mask"))
        self.label_checkpoint.setText(_translate("Form", "Checkpoint"))
        self.pushButton_run_feature_detection.setText(_translate("Form", "Run Feature Detection"))
        self.label.setText(_translate("Form", "Segmentation Model"))
        self.label_num_classes.setText(_translate("Form", "No. Classes"))
        self.label_beam_type.setText(_translate("Form", "Beam Type"))
        self.checkBox_move_features.setText(_translate("Form", "Move Features"))
