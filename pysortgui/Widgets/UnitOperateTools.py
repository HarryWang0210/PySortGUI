import logging

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QItemSelectionModel, Qt
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QStyledItemDelegate, QVBoxLayout, QWidget)

from pysortgui.DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from pysortgui.UI.UnitOperateToolsUIv3_ui import Ui_UnitOperateTools

logger = logging.getLogger(__name__)


class UnitOperateTools(QtWidgets.QWidget, Ui_UnitOperateTools):
    # signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    # signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_showing_spike_data_changed = QtCore.pyqtSignal(object)
    # send spike object to Channel Detail widget
    signal_updating_spike_data = QtCore.pyqtSignal(object)
    signal_showing_units_changed = QtCore.pyqtSignal(object)
    signal_manual_mode_state_changed = QtCore.pyqtSignal(bool)
    signal_features_changed = QtCore.pyqtSignal(list)
    # signal_set_feature_on_selection = QtCore.pyqtSignal(bool)
    signal_feature_on_selection_state_changed = QtCore.pyqtSignal(bool)
    signal_isi_threshold_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Unit Operate Tools"
        self.setupUi(self)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.current_data_object: SpikeSorterData | None = None
        self.current_spike_object: DiscreteData | None = None

        # self.spike_chan = {
        #     'ID': None,
        #     'Name': None,
        #     'Label': None
        # }
        # self.has_spikes = False
        # self.spikes = None
        self.locked_rows_list = []  # store the rows that have been locked
        self.selected_rows_list = []  # store the rows that have been selected
        self.color_palette_list = sns.color_palette('bright', 64)
        self.axis_features = ["PCA1", "PCA2", "PCA3", 'time']
        self.feature1_comboBox.addItems(self.axis_features)
        self.feature1_comboBox.setCurrentIndex(0)

        self.feature2_comboBox.addItems(self.axis_features)
        self.feature2_comboBox.setCurrentIndex(1)

        self.feature3_comboBox.addItems(self.axis_features)
        self.feature3_comboBox.setCurrentIndex(2)

        self.wav_actions_state = {
            self.add_wav_new_pushButton.objectName(): [self.add_wav_new_pushButton, False],
            self.add_wav_selected_pushButton.objectName(): [self.add_wav_selected_pushButton, False],
            self.remove_wav_pushButton.objectName(): [self.remove_wav_pushButton, False],
            self.invalidate_wav_pushButton.objectName(): [self.invalidate_wav_pushButton, False]}

        # self.current_wav_units = []  # all waveform units (N,), int
        self.current_showing_units = []  # the unit id that are showing
        self.isi_result = None
        # store the table data(contain only units)
        # index: 'ID'
        # column: 'Name', 'NumRecords', 'UnitType', 'row'
        self.df_table_data = None

        self.initDataModel()

        # set table color by unit id
        delegate = TableColorDelegate(self.color_palette_list)
        self.tableView.setItemDelegate(delegate)

        self.setupConnections()

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.tableView.verticalHeader().setVisible(False)  # hide index

        selection_model = self.tableView.selectionModel()
        selection_model.selectionChanged.connect(self.onSelectionChanged)

    def setDataModel(self):
        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])

        self.appendUnitRow(0, "All")
        self.appendUnitRow(1, "All_Sorted_Units")

        for i in self.df_table_data.index.to_list():
            self.appendUnitRow(
                self.df_table_data.loc[i, "row"], self.df_table_data.loc[i, "Name"])
        self.tableView.resizeColumnToContents(0)

    def appendUnitRow(self, row, unit_name):
        model = self.tableView.model()

        # 創建一個 CheckBox Widget
        checkbox = QCheckBox()
        checkbox.setProperty("row", row)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.checkboxStateChanged)

        # 將 CheckBox Widget 放入自定義的 Widget 容器中
        checkbox_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(checkbox)
        checkbox_widget.setLayout(layout)

        # 將自定義的 Widget 設定為表格的單元格
        model.appendRow([QStandardItem(),
                        QStandardItem(unit_name)])
        self.tableView.setIndexWidget(
            model.index(row, 0), checkbox_widget)

    def setupConnections(self):
        # Units Action
        self.merge_unit_pushButton.clicked.connect(self.mergeUnits)
        self.swap_unit_pushButton.clicked.connect(self.swapUnits)
        self.remove_unit_pushButton.clicked.connect(self.removeUnits)
        self.invalid_unit_pushButton.clicked.connect(self.invalidateUnits)

        self.add_wav_new_pushButton.clicked.connect(self.sendWaveformAction)
        self.add_wav_selected_pushButton.clicked.connect(
            self.sendWaveformAction)
        self.remove_wav_pushButton.clicked.connect(self.sendWaveformAction)
        self.invalidate_wav_pushButton.clicked.connect(self.sendWaveformAction)

        self.feature1_comboBox.currentTextChanged.connect(self.sendFeatures)
        self.feature2_comboBox.currentTextChanged.connect(self.sendFeatures)
        self.feature3_comboBox.currentTextChanged.connect(self.sendFeatures)

        self.features_on_selection_pushButton.clicked.connect(
            self.setFeatureOnSelection)

        self.isi_thr_doubleSpinBox.valueChanged.connect(
            self.computeUnderISIPercentage)

    # ========== Slot ==========
    def data_file_name_changed(self, data):
        self.data_object = data

        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])
        self.unit_ids_value_label.setText('')
        self.rate_value_label.setText('')
        self.spikes_value_label.setText('')
        self.under_isi_thr_value_label.setText('')

    def spike_data_changed(self, new_spike_object: DiscreteData | None, reset_selection: bool):
        logger.debug('spike_data_changed')

        if reset_selection:
            self.locked_rows_list = []
            self.selected_rows_list = []
            self.current_showing_units = []

        self.current_spike_object = new_spike_object

        self.sendFeatures()
        self.signal_showing_spike_data_changed.emit(self.current_spike_object)

        if self.current_spike_object is None:
            model = self.tableView.model()
            model.clear()
            model.setHorizontalHeaderLabels(["Locked", "UnitName"])
            self.unit_ids_value_label.setText('')
            self.rate_value_label.setText('')
            self.spikes_value_label.setText('')
            self.under_isi_thr_value_label.setText('')
        else:
            self.df_table_data = self.current_spike_object.unit_header
            # self.df_table_data[['ID', 'Name', 'NumRecords', 'UnitType']] = self.[[
            #     'ID', 'Name', 'NumRecords', 'UnitType']].copy()
            self.df_table_data.set_index('ID', inplace=True)
            self.df_table_data['row'] = np.arange(
                self.df_table_data.shape[0]) + 2
        # self.spike_chan['ID'] = int(current_chan_info["ID"])
        # self.spike_chan['Name'] = current_chan_info["Name"]
        # self.spike_chan['Label'] = current_chan_info["Label"]

        # spikes = self.data_object.getSpikes(
        #     self.spike_chan['ID'], self.spike_chan['Label'])
        # spikes = current_chan_info

        # if spikes["unitID"] is None:
        #     self.spikes = None

        # else:
        #     self.spikes = spikes
        #     self.has_spikes = True
        #     self.df_table_data = pd.DataFrame()
        #     self.df_table_data[['ID', 'Name', 'NumRecords', 'UnitType']] = self.spikes['unitInfo'][[
        #         'ID', 'Name', 'NumRecords', 'UnitType']].copy()
        #     self.df_table_data.set_index('ID', inplace=True)
        #     self.df_table_data['row'] = self.df_table_data.index + 2

            self.setDataModel()
            # self.current_wav_units = self.spikes["unitID"].copy()

            # selecting first row by default
            model = self.tableView.model()
            selection_model = self.tableView.selectionModel()
            # selection_model.select(model.index(0, 0),
            #                        QItemSelectionModel.Select)
            if reset_selection:
                selection_model.select(model.index(0, 0),
                                       QItemSelectionModel.Rows | QItemSelectionModel.Select)
            else:
                self.restoreSelection()

    def manual_waveforms(self, wav_index):
        if len(wav_index) == 0:
            return
        # slot of signal_manual_waveforms
        keys_list = np.array(list(self.wav_actions_state.keys()))
        state_list = np.array(list(self.wav_actions_state.values()))[:, 1]
        activate_action = keys_list[np.where(state_list)[0]]

        if activate_action == self.add_wav_new_pushButton.objectName():
            self.addAsNewUnit(wav_index)

        elif activate_action == self.add_wav_selected_pushButton.objectName():
            self.addToSelectedUnit(wav_index)

        elif activate_action == self.remove_wav_pushButton.objectName():
            self.removeWaveforms(wav_index)

        elif activate_action == self.invalidate_wav_pushButton.objectName():
            self.invalidateWaveforms(wav_index)
        else:
            logger.error('Unknowed waveform action type.')

    def checkboxStateChanged(self, state):
        checkbox = self.sender()
        row = checkbox.property("row")
        if state == Qt.Checked and row not in self.locked_rows_list:
            self.locked_rows_list.append(row)
        elif state == Qt.Unchecked and row in self.locked_rows_list:
            self.locked_rows_list.remove(row)
        logger.debug('checkboxStateChanged')
        self.sendShowingUnits()

    def onSelectionChanged(self, selected, deselected):
        selection_model = self.tableView.selectionModel()
        selected_indexes = selection_model.selectedRows()
        self.selected_rows_list = [index.row() for index in selected_indexes]
        logger.debug('onSelectionChanged')
        self.sendShowingUnits()

    def setFeatureOnSelection(self, checked):
        self.signal_feature_on_selection_state_changed.emit(checked)

    # ====================
    def sendShowingUnits(self):
        # if not self.has_spikes:
        #     spikes = {'current_wav_units': self.current_wav_units,
        #               'current_showing_units': self.current_showing_units}
        #     self.signal_showing_spikes_data_changed.emit(spikes)
        all_selected_rows = self.selected_rows_list + self.locked_rows_list

        # All
        if 0 in all_selected_rows:
            selected_ID = self.df_table_data.index.to_list()

        # All_Sorted_Units
        elif 1 in all_selected_rows:
            selected_ID = self.df_table_data[~self.df_table_data['UnitType'].isin(
                ['Unsorted', 'Invalid'])].index.to_list()

        else:
            selected_ID = self.df_table_data[self.df_table_data['row'].isin(
                all_selected_rows)].index.to_list()

        self.current_showing_units = selected_ID

        # spikes = {'current_wav_units': self.current_wav_units,
        #           'current_showing_units': self.current_showing_units}
        logger.info('Selected units: %s', self.current_showing_units)
        self.signal_showing_units_changed.emit(self.current_showing_units)
        self.unit_ids_value_label.setText(
            ', '.join(str(x) for x in self.current_showing_units))
        # logger.debug(
        #     self.df_table_data.loc[self.current_showing_units, 'NumRecords'].sum())
        self.spikes_value_label.setText(str(
            self.df_table_data.loc[self.current_showing_units, 'NumRecords'].sum()))
        self.rate_value_label.setText(str(round(self.current_spike_object.firingRate(
            self.current_showing_units), 2)))
        self.isi_result = self.current_spike_object.ISI(
            self.current_showing_units)
        self.computeUnderISIPercentage()

        # logger.debug(self.current_spike_object.firingRate(
        #     self.current_showing_units))
        # self.signal_showing_spikes_data_changed.emit(spikes)

    def updatingSpikeData(self):
        self.signal_updating_spike_data.emit(self.current_spike_object)
        self.signal_showing_spike_data_changed.emit(self.current_spike_object)

    def sendWaveformAction(self, checked):
        sender = self.sender().objectName()
        if checked:
            self.wav_actions_state[sender][1] = True
            self.exclusiveWaveformActions(sender)
            self.signal_manual_mode_state_changed.emit(True)

        else:
            self.wav_actions_state[sender][1] = False
            self.signal_manual_mode_state_changed.emit(False)

    def sendFeatures(self):
        features = [self.feature1_comboBox.currentText(),
                    self.feature2_comboBox.currentText(),
                    self.feature3_comboBox.currentText()]
        self.signal_features_changed.emit(features)

    def exclusiveWaveformActions(self, new_sender):
        for object_name in self.wav_actions_state.keys():
            if object_name != new_sender:
                if self.wav_actions_state[object_name][1] == True:
                    self.wav_actions_state[object_name][0].setChecked(False)
                    self.wav_actions_state[object_name][1] = False

    def computeUnderISIPercentage(self):
        time_unit = 0.001  # sec = 1ms
        thr = self.isi_thr_doubleSpinBox.value() * time_unit
        under_thr_mask = self.isi_result[0] < thr
        result = self.isi_result[1][under_thr_mask].sum() * 100
        self.under_isi_thr_value_label.setText(str(round(result, 1)))
        self.signal_isi_threshold_changed.emit(thr)
    # ==================== Unit Actions ====================

    def mergeUnits(self):
        logger.info('Merge Units')
        # TODO: contain unsorted unit, all unit merge warning
        merge_unit_IDs = self.current_showing_units
        if len(merge_unit_IDs) < 2:
            logger.error(
                'Merge Units action must select at least 2 units.')
            return

        target_unit_ID = merge_unit_IDs[0]
        wav_index = np.where(
            np.isin(self.current_spike_object.unit_IDs, merge_unit_IDs))[0]

        new_unit_IDs = self.moveWaveforms(wav_index, target_unit_ID)

        self.removeEmptyUnits()

        self.reorderUnitID(new_unit_IDs)

        self.updatingSpikeData()

        self.setDataModel()

        self.restoreSelection()

    def swapUnits(self):
        logger.info('Swap Units')
        swap_unit_IDs = self.current_showing_units
        if len(swap_unit_IDs) != 2:
            logger.error('Swap Units action can only select 2 units.')
            return

        unit1_ID, unit2_ID = swap_unit_IDs
        unit1_wav_index = np.where(
            np.isin(self.current_spike_object.unit_IDs, [unit1_ID]))[0]
        unit2_wav_index = np.where(
            np.isin(self.current_spike_object.unit_IDs, [unit2_ID]))[0]

        temp_unit_id = self.createNewUnit('Unit')

        new_unit_IDs = self.moveWaveforms(
            unit1_wav_index, temp_unit_id)  # 1 to temp
        new_unit_IDs = self.moveWaveforms(
            unit2_wav_index, unit1_ID, new_unit_IDs)  # 2 to 1
        new_unit_IDs = self.moveWaveforms(
            unit1_wav_index, unit2_ID, new_unit_IDs)  # temp to 2

        self.removeEmptyUnits()

        self.reorderUnitID(new_unit_IDs)

        self.updatingSpikeData()

        self.setDataModel()

        self.restoreSelection()

    def removeUnits(self):
        # TODO: all unit warning
        logger.info('Remove Units')
        remove_unit_IDs = self.current_showing_units
        if len(remove_unit_IDs) == 0:
            return
        wav_index = np.where(
            np.isin(self.current_spike_object.unit_IDs, remove_unit_IDs))[0]
        self.removeWaveforms(wav_index)

    def invalidateUnits(self):
        # TODO: all unit warning
        logger.info('Invalidate Units')
        invalid_unit_IDs = self.current_showing_units
        if len(invalid_unit_IDs) == 0:
            return
        wav_index = np.where(
            np.isin(self.current_spike_object.unit_IDs, invalid_unit_IDs))[0]
        self.invalidateWaveforms(wav_index)
    # ========================================

    # ==================== Waveform Actions ====================
    def addAsNewUnit(self, wav_index):
        new_unit_ID = self.createNewUnit('Unit')

        new_unit_IDs = self.moveWaveforms(wav_index, new_unit_ID)

        self.removeEmptyUnits()

        map_new_ID_dict = self.reorderUnitID(new_unit_IDs)

        self.updatingSpikeData()

        self.setDataModel()
        if len(self.selected_rows_list) < 1:
            self.selected_rows_list = [
                self.df_table_data.loc[map_new_ID_dict[new_unit_ID], 'row']]
        self.restoreSelection()

    def addToSelectedUnit(self, wav_index):
        unit_ID_list = self.current_showing_units
        accepted, target_unit_ID = self.askTargetUnit(unit_ID_list)
        if accepted:
            target_unit_type = self.df_table_data.loc[target_unit_ID, 'UnitType']

            if target_unit_type in ['Unsorted', 'Invalid']:
                warning_result = QMessageBox.warning(self,
                                                     "Warning",
                                                     f"The target unit is {target_unit_type} unit.\nDo you want to continue?",
                                                     QMessageBox.Yes | QMessageBox.No)
                if warning_result != QMessageBox.Yes:
                    return
        else:
            return

        new_unit_IDs = self.moveWaveforms(wav_index, target_unit_ID)

        self.removeEmptyUnits()

        map_new_ID_dict = self.reorderUnitID(new_unit_IDs)

        self.updatingSpikeData()

        self.setDataModel()
        if len(self.selected_rows_list) < 1:
            self.selected_rows_list = [
                self.df_table_data.loc[map_new_ID_dict[target_unit_ID], 'row']]
        self.restoreSelection()

    def removeWaveforms(self, wav_index):
        unsorted_unit_ID = self.df_table_data[self.df_table_data['UnitType'] == 'Unsorted'].index.tolist(
        )
        if len(unsorted_unit_ID) < 1:
            unsorted_unit_ID = self.createNewUnit('Unsorted')
            new_unit_IDs = self.moveWaveforms(wav_index, unsorted_unit_ID)

        elif len(unsorted_unit_ID) != 1:
            # TODO: Select one to move.
            print('TODO: Select one to move.')

        else:
            unsorted_unit_ID = unsorted_unit_ID[0]
            new_unit_IDs = self.moveWaveforms(wav_index, unsorted_unit_ID)

        self.removeEmptyUnits()

        map_new_ID_dict = self.reorderUnitID(new_unit_IDs)

        self.updatingSpikeData()

        self.setDataModel()
        if len(self.selected_rows_list) < 1:
            self.selected_rows_list = [
                self.df_table_data.loc[map_new_ID_dict[unsorted_unit_ID], 'row']]
        self.restoreSelection()

    def invalidateWaveforms(self, wav_index):
        invalid_unit_ID = self.df_table_data[self.df_table_data['UnitType'] == 'Invalid'].index.tolist(
        )
        if len(invalid_unit_ID) < 1:
            invalid_unit_ID = self.createNewUnit('Invalid')
            new_unit_IDs = self.moveWaveforms(wav_index, invalid_unit_ID)

        elif len(invalid_unit_ID) != 1:
            # TODO: Select one to move.
            print('TODO: Select one to move.')

        else:
            invalid_unit_ID = invalid_unit_ID[0]
            new_unit_IDs = self.moveWaveforms(wav_index, invalid_unit_ID)

        self.removeEmptyUnits()

        map_new_ID_dict = self.reorderUnitID(new_unit_IDs)

        self.updatingSpikeData()

        self.setDataModel()
        if len(self.selected_rows_list) < 1:
            self.selected_rows_list = [
                self.df_table_data.loc[map_new_ID_dict[invalid_unit_ID], 'row']]
        self.restoreSelection()
    # ========================================

    def askTargetUnit(self, unit_ID_list):
        """Create a dialog to ask the target unit.

        Args:
            unit_ID_list (list): a list of unit IDs.

        Returns:
            A tuple contain response (bool) and target unit ID (int or None).

            (True, 0)
            (False, None)
        """
        unit_name_list = self.df_table_data.loc[self.current_showing_units, 'Name']
        dialog = SelectTargetUnitDialog(unit_ID_list=unit_ID_list,
                                        unit_name_list=unit_name_list,
                                        color_palette=self.color_palette_list,
                                        parent=self)
        result = dialog.exec_()  # 显示对话框并获取结果

        if result == QDialog.Accepted:
            target_unit_name = dialog.comboBox.currentText()
            target_unit_ID = self.df_table_data[self.df_table_data['Name'] == target_unit_name].index.to_list()[
                0]
            return (True, target_unit_ID)
        else:
            return (False, None)

    def createNewUnit(self, unit_type):
        new_unit_ID = int(self.df_table_data.index.max() + 1)
        new_unit_row = int(self.df_table_data['row'].max() + 1)

        new_unit_name_suffix = f'_{unit_type}' if unit_type in [
            'Unsorted', 'Invalid'] else ''
        new_unit = pd.DataFrame({
            'Name': f'{self.current_spike_object.channel_name}_Unit_{new_unit_ID:02}{new_unit_name_suffix}',
            'NumRecords': 0,
            'UnitType': unit_type,
            'row': new_unit_row}, index=[new_unit_ID])
        # self.appendUnitRow(new_unit.loc[new_unit_ID, 'row'],
        #                    new_unit.loc[new_unit_ID, 'Name'])
        self.df_table_data = pd.concat([self.df_table_data, new_unit], axis=0)

        return new_unit_ID

    def removeEmptyUnits(self):
        excluded_units = self.df_table_data['UnitType'].isin(
            ['Unsorted', 'Invalid'])
        remove_units = self.df_table_data['NumRecords'] == 0

        self.df_table_data = self.df_table_data[
            (~remove_units) | excluded_units]

    def moveWaveforms(self, wav_index, target_unit, unit_IDs=None):
        if unit_IDs is None:
            new_unit_IDs = self.current_spike_object.unit_IDs
        else:
            new_unit_IDs = unit_IDs
        new_unit_IDs[wav_index] = target_unit

        # update NumRecords
        unique, counts = np.unique(new_unit_IDs, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.df_table_data['NumRecords'] = self.df_table_data.index.map(
            counts_dict)
        self.df_table_data['NumRecords'].fillna(0, inplace=True)
        self.df_table_data['NumRecords'] = self.df_table_data['NumRecords'].astype(
            int)
        return new_unit_IDs

    def reorderUnitID(self, unit_IDs):
        unsorted_units = self.df_table_data['UnitType'] == 'Unsorted'
        invalid_units = self.df_table_data['UnitType'] == 'Invalid'

        # """unsorted/invalid/sorted"""
        # new_df_table_data = pd.concat(
        #     [self.df_table_data[unsorted_units], self.df_table_data[invalid_units]])
        # new_df_table_data = pd.concat(
        #     [new_df_table_data, self.df_table_data[~(unsorted_units | invalid_units)]])

        """unsorted/sorted/invalid"""
        new_df_table_data = pd.concat(
            [self.df_table_data[unsorted_units], self.df_table_data[~(unsorted_units | invalid_units)]])
        new_df_table_data = pd.concat(
            [new_df_table_data, self.df_table_data[invalid_units]])

        old_ID = new_df_table_data.index.copy()
        old_row = new_df_table_data['row'].copy()
        new_df_table_data.reset_index(drop=True, inplace=True)

        new_df_table_data['row'] = new_df_table_data.index + 2

        map_new_row_dict = dict(zip(old_row, new_df_table_data['row']))

        # handle selection changed
        new_locked_rows_list = []
        for i in self.locked_rows_list:
            if i in [0, 1]:
                new_locked_rows_list.append(i)
            elif i in map_new_row_dict.keys():
                new_locked_rows_list.append(map_new_row_dict[i])

        new_selected_rows_list = []
        for i in self.selected_rows_list:
            if i in [0, 1]:
                new_selected_rows_list.append(i)
            elif i in map_new_row_dict.keys():
                new_selected_rows_list.append(map_new_row_dict[i])

        self.locked_rows_list = new_locked_rows_list
        self.selected_rows_list = new_selected_rows_list

        # handle ID and Name changed
        map_new_ID_dict = dict(zip(old_ID, new_df_table_data.index))

        def replace_func(match):
            index_value = map_new_ID_dict[int(match.group(1))]
            return f"_Unit_{index_value:02}"

        new_df_table_data['Name'] = new_df_table_data['Name'].str.replace(
            r'_Unit_(\d{2})', replace_func, regex=True)
        self.df_table_data = new_df_table_data
        new_unit_IDs = np.vectorize(
            map_new_ID_dict.get)(unit_IDs)

        new_unit_header = self.df_table_data.copy()
        new_unit_header['ID'] = self.df_table_data.index
        new_unit_header.drop(columns=['row'], inplace=True)
        self.current_spike_object = self.current_spike_object.setUnit(new_unit_IDs,
                                                                      new_unit_header=new_unit_header)
        return map_new_ID_dict

    def restoreSelection(self):
        model = self.tableView.model()
        for row in self.locked_rows_list:
            if row > model.rowCount() - 1:
                break

            checkbox = self.tableView.indexWidget(
                model.index(row, 0)).layout().itemAt(0).widget()
            checkbox.setChecked(True)

        selection_model = self.tableView.selectionModel()
        for row in self.selected_rows_list:
            if row > model.rowCount() - 1:
                selection_model.select(model.index(0, 0),
                                       QItemSelectionModel.Rows | QItemSelectionModel.Select)
            selection_model.select(model.index(row, 0),
                                   QItemSelectionModel.Rows | QItemSelectionModel.Select)
            # selection_model.select(model.index(row, 1),
            #                        QItemSelectionModel.Select)


class SelectTargetUnitDialog(QDialog):
    def __init__(self, unit_ID_list, unit_name_list, color_palette, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Unit Dialog")

        self.setMinimumWidth(100)
        self.setMinimumHeight(50)
        self.color_palette_list = color_palette

        self.setupUi()

        self.createComboBox(unit_name_list)
        delegate = ComboBoxColorDelegate(unit_ID_list, self.color_palette_list)
        self.comboBox.setItemDelegate(delegate)

    def setupUi(self):
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        label = QLabel('Please select the target unit.')
        layout.addWidget(label)

        self.comboBox = QComboBox()
        layout.addWidget(self.comboBox)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)  # 关闭对话框并返回 Accepted 结果
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)  # 关闭对话框并返回 Accepted 结果
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def createComboBox(self, unit_name_list):
        model = QStandardItemModel()
        self.comboBox.setModel(model)

        for unit in unit_name_list:
            model.appendRow(QStandardItem(str(unit)))


class TableColorDelegate(QStyledItemDelegate):
    def __init__(self, color_palette):
        super().__init__()
        self.color_palette_list = color_palette

    def initStyleOption(self, option, index):
        # FIXME: the color of table do not consist with the views
        super().initStyleOption(option, index)

        if index.row() < 2:
            option.backgroundBrush = QColor(255, 255, 255)
        else:
            option.backgroundBrush = QColor(
                *[int(c * 255) for c in self.color_palette_list[index.row() - 2]])


class ComboBoxColorDelegate(QStyledItemDelegate):
    def __init__(self, unit_ID_list, color_palette):
        super().__init__()
        self.unit_ID_list = unit_ID_list
        self.color_palette_list = color_palette

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)

        option.backgroundBrush = QColor(
            *[int(c * 255) for c in self.color_palette_list[self.unit_ID_list[index.row()]]])
