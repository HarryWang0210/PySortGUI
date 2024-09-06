import logging

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QItemSelection, QItemSelectionModel, Qt
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QStyledItemDelegate, QVBoxLayout, QWidget)

from pysortgui.DataStructure.datav3 import (ContinuousData, DiscreteData,
                                            SpikeSorterData)
from pysortgui.UI.BackgroundChannelSetting_ui import \
    Ui_BackgroundChannelSetting
from pysortgui.Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class BackgroundChannelSetting(WidgetsInterface, Ui_BackgroundChannelSetting):
    # send dict
    signal_set_background_channel = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.window_title = "Background Channel Setting"
        self.setupUi(self)

        self.default_background_channel_setting = {
            'Show': False,
            'BackgroundChannel': 0,
            'Color': None,
            'ShowOnTop': False,
            'Reference': (False, 0),
            'Filter': (False, 250, 6000),
        }
        self.current_data_object: SpikeSorterData | None = None
        self.current_raw_object: ContinuousData | None = None
        self.current_filted_object: ContinuousData | None = None
        self.current_spike_object: DiscreteData | None = None
        self.current_event_object: DiscreteData | None = None

        self.setting = self.default_background_channel_setting.copy()

        self.current_showing_events: list = []  # event units that are selected to show
        self.setupConnections()
        self.setColor()
        self.bg_channel_checkBox.setChecked(self.setting['Show'])
        self.bg_channel_comboBox.setEnabled(self.setting['Show'])
        self.show_on_top_checkBox.setChecked(self.setting['ShowOnTop'])
        self.ref_groupBox.setChecked(self.setting['Reference'][0])

        self.filter_groupBox.setChecked(self.setting['Filter'][0])
        self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][1])
        self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][2])

    def setColor(self):
        if self.setting['Color'] is None:
            self.setting['Color'] = QColor(0, 255, 255)
        self.color_pushButton.setStyleSheet(
            f"background-color: {self.setting['Color'].name()}")

    def setupConnections(self):
        self.color_pushButton.clicked.connect(self.selectColor)
        self.apply_pushButton.clicked.connect(self.applyChanges)

    def selectColor(self):
        if self.setting['Color'] is None:
            color = QColorDialog.getColor(parent=self)
        else:
            color = QColorDialog.getColor(
                initial=self.setting['Color'], parent=self)

        if color.isValid():
            self.setting['Color'] = color
            self.setColor()

    def applyChanges(self):
        logger.debug('apply')
        bg_channel = self.bg_channel_comboBox.currentText()
        try:
            bg_channel = int(bg_channel)
        except (TypeError, ValueError):
            pass

        # if self.raw_reference_radioButton.isChecked():
        #     ref = 'raw'
        #     ref_channel = self.raw_reference_comboBox.currentText()
        # elif self.ref_reference_radioButton.isChecked():
        #     ref = 'ref'
        ref_channel = self.select_reference_comboBox.currentText()
        try:
            ref_channel = int(ref_channel)
        except (TypeError, ValueError):
            pass
        self.setting['Show'] = self.bg_channel_checkBox.isChecked()
        self.setting['BackgroundChannel'] = bg_channel
        # self.setting['Color']
        self.setting['ShowOnTop'] = self.show_on_top_checkBox.isChecked()
        self.setting['Reference'] = (self.ref_groupBox.isChecked(),
                                     ref_channel)
        self.setting['Filter'] = (self.filter_groupBox.isChecked(),
                                  self.filter_low_doubleSpinBox.value(),
                                  self.filter_high_doubleSpinBox.value())

        self.signal_set_background_channel.emit(self.setting)

    # ========== Slot ==========
    def data_file_name_changed(self, data):
        self.current_data_object = data
        all_channel_IDs = self.current_data_object.channel_IDs
        self.bg_channel_comboBox.addItems(map(str, all_channel_IDs))
        self.bg_channel_comboBox.setCurrentText(
            str(self.setting['BackgroundChannel']))
        self.select_reference_comboBox.addItems(map(str, all_channel_IDs))

# class SetBackgroundChannelDialog(Ui_SetBackgroundChannelDialog, QDialog):
#     def __init__(self, all_channel_IDs, default_setting, parent=None):
#         """_summary_

#         Args:
#             all_channel_IDs (_type_): _description_
#             default_setting (_type_): {
#                 'BackgroundChannel': 'No select',
#                 'Color': None,
#                 'Reference': (False, 0),
#                 'Filter': (False, 250, 6000),
#                 }
#             parent (_type_, optional): _description_. Defaults to None.
#         """
#         super().__init__(parent)
#         self.setupUi(self)
#         self.setWindowTitle("Set BackgroundChannel Dialog")
#         # self.setMinimumWidth(500)
#         # self.setMinimumHeight(500)
#         self.setting = default_setting
#         self.setColor()

#         self.bg_channel_checkBox.setChecked(self.setting['Show'])
#         self.bg_channel_comboBox.setEnabled(self.setting['Show'])
#         self.bg_channel_comboBox.addItems(map(str, all_channel_IDs))
#         self.bg_channel_comboBox.setCurrentText(
#             str(self.setting['BackgroundChannel']))

#         self.show_on_top_checkBox.setChecked(self.setting['ShowOnTop'])

#         self.ref_groupBox.setChecked(self.setting['Reference'][0])
#         self.select_reference_comboBox.addItems(map(str, all_channel_IDs))

#         # if self.ref_groupBox.isChecked():
#         #     if self.setting['Reference'][1] == 'raw':
#         #         self.raw_reference_radioButton.setChecked(True)
#         #         self.raw_reference_comboBox.setCurrentText(
#         #             str(self.setting['Reference'][2]))

#         #     elif self.setting['Reference'][1] == 'ref':
#         #         self.raw_reference_radioButton.setChecked(True)
#         #         self.raw_reference_comboBox.setCurrentText(
#         #             str(self.setting['Reference'][2]))

#         self.filter_groupBox.setChecked(self.setting['Filter'][0])
#         self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][1])
#         self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][2])

#         self.color_pushButton.clicked.connect(self.selectColor)

#     def setColor(self):
#         if self.setting['Color'] is None:
#             self.setting['Color'] = QColor(0, 255, 255)
#         self.color_pushButton.setStyleSheet(
#             f"background-color: {self.setting['Color'].name()}")

#     def selectColor(self):
#         if self.setting['Color'] is None:
#             color = QColorDialog.getColor(parent=self)
#         else:
#             color = QColorDialog.getColor(
#                 initial=self.setting['Color'], parent=self)

#         if color.isValid():
#             self.setting['Color'] = color
#             self.setColor()

#     def accept(self):
#         bg_channel = self.bg_channel_comboBox.currentText()
#         try:
#             bg_channel = int(bg_channel)
#         except TypeError:
#             pass

#         # if self.raw_reference_radioButton.isChecked():
#         #     ref = 'raw'
#         #     ref_channel = self.raw_reference_comboBox.currentText()
#         # elif self.ref_reference_radioButton.isChecked():
#         #     ref = 'ref'
#         ref_channel = self.select_reference_comboBox.currentText()
#         try:
#             ref_channel = int(ref_channel)
#         except TypeError:
#             pass
#         self.setting['Show'] = self.bg_channel_checkBox.isChecked()
#         self.setting['BackgroundChannel'] = bg_channel
#         # self.setting['Color']
#         self.setting['ShowOnTop'] = self.show_on_top_checkBox.isChecked()
#         self.setting['Reference'] = (self.ref_groupBox.isChecked(),
#                                      ref_channel)
#         self.setting['Filter'] = (self.filter_groupBox.isChecked(),
#                                   self.filter_low_doubleSpinBox.value(),
#                                   self.filter_high_doubleSpinBox.value())

#         super().accept()
