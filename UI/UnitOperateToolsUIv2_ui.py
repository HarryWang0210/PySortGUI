# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\harry\Desktop\Lab\Project_spikesorter\try\UI\UnitOperateToolsUIv2.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UnitOperateTools(object):
    def setupUi(self, UnitOperateTools):
        UnitOperateTools.setObjectName("UnitOperateTools")
        UnitOperateTools.resize(268, 396)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(UnitOperateTools.sizePolicy().hasHeightForWidth())
        UnitOperateTools.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        UnitOperateTools.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(UnitOperateTools)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableView = QtWidgets.QTableView(UnitOperateTools)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableView.setTabKeyNavigation(False)
        self.tableView.setProperty("showDropIndicator", True)
        self.tableView.setDragDropOverwriteMode(False)
        self.tableView.setObjectName("tableView")
        self.tableView.verticalHeader().setVisible(False)
        self.verticalLayout.addWidget(self.tableView)
        self.line = QtWidgets.QFrame(UnitOperateTools)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.unit_actions_label = QtWidgets.QLabel(UnitOperateTools)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.unit_actions_label.setFont(font)
        self.unit_actions_label.setAlignment(QtCore.Qt.AlignCenter)
        self.unit_actions_label.setObjectName("unit_actions_label")
        self.verticalLayout.addWidget(self.unit_actions_label)
        self.is_multiunit_checkBox = QtWidgets.QCheckBox(UnitOperateTools)
        self.is_multiunit_checkBox.setTristate(False)
        self.is_multiunit_checkBox.setObjectName("is_multiunit_checkBox")
        self.verticalLayout.addWidget(self.is_multiunit_checkBox)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.add_unit_pushButton = QtWidgets.QPushButton(UnitOperateTools)
        self.add_unit_pushButton.setMinimumSize(QtCore.QSize(45, 20))
        self.add_unit_pushButton.setObjectName("add_unit_pushButton")
        self.horizontalLayout.addWidget(self.add_unit_pushButton)
        self.remove_unit_pushButton = QtWidgets.QPushButton(UnitOperateTools)
        self.remove_unit_pushButton.setMinimumSize(QtCore.QSize(45, 20))
        self.remove_unit_pushButton.setObjectName("remove_unit_pushButton")
        self.horizontalLayout.addWidget(self.remove_unit_pushButton)
        self.merge_units_pushButton = QtWidgets.QPushButton(UnitOperateTools)
        self.merge_units_pushButton.setMinimumSize(QtCore.QSize(45, 20))
        self.merge_units_pushButton.setObjectName("merge_units_pushButton")
        self.horizontalLayout.addWidget(self.merge_units_pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.wav_actions_label = QtWidgets.QLabel(UnitOperateTools)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.wav_actions_label.setFont(font)
        self.wav_actions_label.setAlignment(QtCore.Qt.AlignCenter)
        self.wav_actions_label.setObjectName("wav_actions_label")
        self.verticalLayout.addWidget(self.wav_actions_label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.add_wav_pushButton = QtWidgets.QPushButton(UnitOperateTools)
        self.add_wav_pushButton.setMinimumSize(QtCore.QSize(45, 20))
        self.add_wav_pushButton.setCheckable(True)
        self.add_wav_pushButton.setObjectName("add_wav_pushButton")
        self.horizontalLayout_2.addWidget(self.add_wav_pushButton)
        self.remove_wav_pushButton = QtWidgets.QPushButton(UnitOperateTools)
        self.remove_wav_pushButton.setMinimumSize(QtCore.QSize(45, 20))
        self.remove_wav_pushButton.setCheckable(True)
        self.remove_wav_pushButton.setObjectName("remove_wav_pushButton")
        self.horizontalLayout_2.addWidget(self.remove_wav_pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.line_1 = QtWidgets.QFrame(UnitOperateTools)
        self.line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_1.setObjectName("line_1")
        self.verticalLayout.addWidget(self.line_1)
        self.features_label = QtWidgets.QLabel(UnitOperateTools)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.features_label.setFont(font)
        self.features_label.setAlignment(QtCore.Qt.AlignCenter)
        self.features_label.setObjectName("features_label")
        self.verticalLayout.addWidget(self.features_label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.feature1_comboBox = QtWidgets.QComboBox(UnitOperateTools)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.feature1_comboBox.sizePolicy().hasHeightForWidth())
        self.feature1_comboBox.setSizePolicy(sizePolicy)
        self.feature1_comboBox.setMinimumSize(QtCore.QSize(50, 20))
        self.feature1_comboBox.setObjectName("feature1_comboBox")
        self.feature1_comboBox.addItem("")
        self.feature1_comboBox.addItem("")
        self.feature1_comboBox.addItem("")
        self.feature1_comboBox.addItem("")
        self.feature1_comboBox.addItem("")
        self.feature1_comboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.feature1_comboBox)
        self.feature2_comboBox = QtWidgets.QComboBox(UnitOperateTools)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.feature2_comboBox.sizePolicy().hasHeightForWidth())
        self.feature2_comboBox.setSizePolicy(sizePolicy)
        self.feature2_comboBox.setMinimumSize(QtCore.QSize(50, 20))
        self.feature2_comboBox.setObjectName("feature2_comboBox")
        self.feature2_comboBox.addItem("")
        self.feature2_comboBox.addItem("")
        self.feature2_comboBox.addItem("")
        self.feature2_comboBox.addItem("")
        self.feature2_comboBox.addItem("")
        self.feature2_comboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.feature2_comboBox)
        self.feature3_comboBox = QtWidgets.QComboBox(UnitOperateTools)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.feature3_comboBox.sizePolicy().hasHeightForWidth())
        self.feature3_comboBox.setSizePolicy(sizePolicy)
        self.feature3_comboBox.setMinimumSize(QtCore.QSize(50, 20))
        self.feature3_comboBox.setObjectName("feature3_comboBox")
        self.feature3_comboBox.addItem("")
        self.feature3_comboBox.addItem("")
        self.feature3_comboBox.addItem("")
        self.feature3_comboBox.addItem("")
        self.feature3_comboBox.addItem("")
        self.feature3_comboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.feature3_comboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.features_on_selection_pushButton = QtWidgets.QPushButton(UnitOperateTools)
        self.features_on_selection_pushButton.setMinimumSize(QtCore.QSize(102, 0))
        self.features_on_selection_pushButton.setCheckable(True)
        self.features_on_selection_pushButton.setObjectName("features_on_selection_pushButton")
        self.verticalLayout.addWidget(self.features_on_selection_pushButton)
        self.line_2 = QtWidgets.QFrame(UnitOperateTools)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.selection_label = QtWidgets.QLabel(UnitOperateTools)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.selection_label.setFont(font)
        self.selection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.selection_label.setObjectName("selection_label")
        self.verticalLayout.addWidget(self.selection_label)
        self.unit_name_value_label = QtWidgets.QLabel(UnitOperateTools)
        self.unit_name_value_label.setObjectName("unit_name_value_label")
        self.verticalLayout.addWidget(self.unit_name_value_label)
        self.selection_spikes_horizontalLayout = QtWidgets.QHBoxLayout()
        self.selection_spikes_horizontalLayout.setObjectName("selection_spikes_horizontalLayout")
        self.spikes_label_1 = QtWidgets.QLabel(UnitOperateTools)
        self.spikes_label_1.setObjectName("spikes_label_1")
        self.selection_spikes_horizontalLayout.addWidget(self.spikes_label_1)
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.selection_spikes_horizontalLayout.addItem(spacerItem)
        self.spikes_value_label = QtWidgets.QLabel(UnitOperateTools)
        self.spikes_value_label.setObjectName("spikes_value_label")
        self.selection_spikes_horizontalLayout.addWidget(self.spikes_value_label)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.selection_spikes_horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.selection_spikes_horizontalLayout)
        self.selection_rate_horizontalLayout = QtWidgets.QHBoxLayout()
        self.selection_rate_horizontalLayout.setObjectName("selection_rate_horizontalLayout")
        self.rate_label_1 = QtWidgets.QLabel(UnitOperateTools)
        self.rate_label_1.setObjectName("rate_label_1")
        self.selection_rate_horizontalLayout.addWidget(self.rate_label_1)
        spacerItem2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.selection_rate_horizontalLayout.addItem(spacerItem2)
        self.rate_value_label = QtWidgets.QLabel(UnitOperateTools)
        self.rate_value_label.setObjectName("rate_value_label")
        self.selection_rate_horizontalLayout.addWidget(self.rate_value_label)
        self.rate_label_2 = QtWidgets.QLabel(UnitOperateTools)
        self.rate_label_2.setObjectName("rate_label_2")
        self.selection_rate_horizontalLayout.addWidget(self.rate_label_2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.selection_rate_horizontalLayout.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.selection_rate_horizontalLayout)
        self.selection_isi_horizontalLayout = QtWidgets.QHBoxLayout()
        self.selection_isi_horizontalLayout.setObjectName("selection_isi_horizontalLayout")
        self.isi_label_1 = QtWidgets.QLabel(UnitOperateTools)
        self.isi_label_1.setObjectName("isi_label_1")
        self.selection_isi_horizontalLayout.addWidget(self.isi_label_1)
        self.isi_thr_doubleSpinBox = QtWidgets.QDoubleSpinBox(UnitOperateTools)
        self.isi_thr_doubleSpinBox.setDecimals(1)
        self.isi_thr_doubleSpinBox.setProperty("value", 5.0)
        self.isi_thr_doubleSpinBox.setObjectName("isi_thr_doubleSpinBox")
        self.selection_isi_horizontalLayout.addWidget(self.isi_thr_doubleSpinBox)
        self.isi_label_2 = QtWidgets.QLabel(UnitOperateTools)
        self.isi_label_2.setObjectName("isi_label_2")
        self.selection_isi_horizontalLayout.addWidget(self.isi_label_2)
        spacerItem4 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.selection_isi_horizontalLayout.addItem(spacerItem4)
        self.under_isi_thr_value_label = QtWidgets.QLabel(UnitOperateTools)
        self.under_isi_thr_value_label.setObjectName("under_isi_thr_value_label")
        self.selection_isi_horizontalLayout.addWidget(self.under_isi_thr_value_label)
        self.isi_label_3 = QtWidgets.QLabel(UnitOperateTools)
        self.isi_label_3.setObjectName("isi_label_3")
        self.selection_isi_horizontalLayout.addWidget(self.isi_label_3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.selection_isi_horizontalLayout.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.selection_isi_horizontalLayout)

        self.retranslateUi(UnitOperateTools)
        self.feature2_comboBox.setCurrentIndex(1)
        self.feature3_comboBox.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(UnitOperateTools)

    def retranslateUi(self, UnitOperateTools):
        _translate = QtCore.QCoreApplication.translate
        UnitOperateTools.setWindowTitle(_translate("UnitOperateTools", "Form"))
        self.unit_actions_label.setText(_translate("UnitOperateTools", "Unit Actions"))
        self.is_multiunit_checkBox.setText(_translate("UnitOperateTools", "is multiunit"))
        self.add_unit_pushButton.setText(_translate("UnitOperateTools", "Add"))
        self.remove_unit_pushButton.setText(_translate("UnitOperateTools", "Remove"))
        self.merge_units_pushButton.setText(_translate("UnitOperateTools", "Merge"))
        self.wav_actions_label.setText(_translate("UnitOperateTools", "Waveform Actions"))
        self.add_wav_pushButton.setText(_translate("UnitOperateTools", "Add"))
        self.remove_wav_pushButton.setText(_translate("UnitOperateTools", "Remove"))
        self.features_label.setText(_translate("UnitOperateTools", "Features"))
        self.feature1_comboBox.setItemText(0, _translate("UnitOperateTools", "PCA1"))
        self.feature1_comboBox.setItemText(1, _translate("UnitOperateTools", "PCA2"))
        self.feature1_comboBox.setItemText(2, _translate("UnitOperateTools", "PCA3"))
        self.feature1_comboBox.setItemText(3, _translate("UnitOperateTools", "time"))
        self.feature1_comboBox.setItemText(4, _translate("UnitOperateTools", "amplitude"))
        self.feature1_comboBox.setItemText(5, _translate("UnitOperateTools", "slice"))
        self.feature2_comboBox.setCurrentText(_translate("UnitOperateTools", "PCA2"))
        self.feature2_comboBox.setItemText(0, _translate("UnitOperateTools", "PCA1"))
        self.feature2_comboBox.setItemText(1, _translate("UnitOperateTools", "PCA2"))
        self.feature2_comboBox.setItemText(2, _translate("UnitOperateTools", "PCA3"))
        self.feature2_comboBox.setItemText(3, _translate("UnitOperateTools", "time"))
        self.feature2_comboBox.setItemText(4, _translate("UnitOperateTools", "amplitude"))
        self.feature2_comboBox.setItemText(5, _translate("UnitOperateTools", "slice"))
        self.feature3_comboBox.setCurrentText(_translate("UnitOperateTools", "PCA3"))
        self.feature3_comboBox.setItemText(0, _translate("UnitOperateTools", "PCA1"))
        self.feature3_comboBox.setItemText(1, _translate("UnitOperateTools", "PCA2"))
        self.feature3_comboBox.setItemText(2, _translate("UnitOperateTools", "PCA3"))
        self.feature3_comboBox.setItemText(3, _translate("UnitOperateTools", "time"))
        self.feature3_comboBox.setItemText(4, _translate("UnitOperateTools", "amplitude"))
        self.feature3_comboBox.setItemText(5, _translate("UnitOperateTools", "slice"))
        self.features_on_selection_pushButton.setText(_translate("UnitOperateTools", "Features on selection"))
        self.selection_label.setText(_translate("UnitOperateTools", "Selection"))
        self.unit_name_value_label.setText(_translate("UnitOperateTools", "Unit_name"))
        self.spikes_label_1.setText(_translate("UnitOperateTools", "Spikes: "))
        self.spikes_value_label.setText(_translate("UnitOperateTools", "0"))
        self.rate_label_1.setText(_translate("UnitOperateTools", "Rate: "))
        self.rate_value_label.setText(_translate("UnitOperateTools", "0"))
        self.rate_label_2.setText(_translate("UnitOperateTools", "Hz"))
        self.isi_label_1.setText(_translate("UnitOperateTools", "ISI < "))
        self.isi_label_2.setText(_translate("UnitOperateTools", "ms: "))
        self.under_isi_thr_value_label.setText(_translate("UnitOperateTools", "0"))
        self.isi_label_3.setText(_translate("UnitOperateTools", "%"))
