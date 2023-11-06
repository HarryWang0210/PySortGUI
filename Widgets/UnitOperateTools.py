from UI.UnitOperateToolsUIv3_ui import Ui_UnitOperateTools
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QColor
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QAbstractItemView, QCheckBox, QStyledItemDelegate, QDialog, QComboBox, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QItemSelectionModel
import numpy as np
import pandas as pd
import seaborn as sns
from DataStructure.data import SpikeSorterData


class UnitOperateTools(QtWidgets.QWidget, Ui_UnitOperateTools):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_activate_manual_mode = QtCore.pyqtSignal(bool)
    signal_showing_spikes_data_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Unit Operate Tools"
        self.setupUi(self)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.data_object = None
        self.spike_chan_name = ''
        self.visible = False
        self.spikes = None
        self.has_waveforms = False
        self.locked_rows_list = []
        self.selected_rows_list = []
        self.color_palette_list = sns.color_palette(None, 64)
        self.wav_actions_state = {
            self.add_wav_new_pushButton.objectName(): [self.add_wav_new_pushButton, False],
            self.add_wav_selected_pushButton.objectName(): [self.add_wav_selected_pushButton, False],
            self.remove_wav_pushButton.objectName(): [self.remove_wav_pushButton, False],
            self.invalidate_wav_pushButton.objectName(): [self.invalidate_wav_pushButton, False]}

        self.current_wav_units = []  # waveform units (N,), int
        self.current_showing_units = []
        # self.df_table_data = None
        self.initDataModel()

        delegate = TableColorDelegate(self.color_palette_list)
        self.tableView.setItemDelegate(delegate)
        self.setupConnections()

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.tableView.verticalHeader().setVisible(False)

        selection_model = self.tableView.selectionModel()
        selection_model.selectionChanged.connect(self.onSelectionChanged)

    def setDataModel(self):
        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])

        self.appendUnitRow(0, "All")
        self.appendUnitRow(1, "All_Sorted_Units")

        for i in range(self.df_table_data.shape[0]):
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

        self.feature1_comboBox
        self.feature2_comboBox
        self.feature3_comboBox

        self.features_on_selection_pushButton

        self.isi_thr_doubleSpinBox

    # ========== Slot ==========
    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False

    def spike_chan_changed(self, meta_data):
        spikes = self.data_object.getSpikes(
            meta_data["ID"], meta_data["Label"])
        self.spike_chan_name = meta_data["Name"]

        self.visible = True
        self.locked_rows_list = []
        self.selected_rows_list = []
        self.current_showing_units = []

        if spikes["unitInfo"] is None:
            self.spikes = None
            self.has_waveforms = False
            model = self.tableView.model()
            model.clear()
            self.current_wav_units = []

        else:
            self.spikes = spikes
            self.has_waveforms = True
            self.df_table_data = pd.DataFrame()
            self.df_table_data[['ID', 'Name', 'NumRecords', 'UnitType']] = self.spikes['unitInfo'][[
                'ID', 'Name', 'NumRecords', 'UnitType']].copy()
            self.df_table_data.set_index('ID', inplace=True)
            self.df_table_data['row'] = self.df_table_data.index + 2

            self.setDataModel()
            self.current_wav_units = self.spikes["unitID"]

            model = self.tableView.model()
            selection_model = self.tableView.selectionModel()
            selection_model.select(model.index(0, 0),
                                   QItemSelectionModel.Select)
            selection_model.select(model.index(0, 1),
                                   QItemSelectionModel.Select)
        self.sendSpikesData()

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
            print('Error: Unknowed waveform action type.')

    def checkboxStateChanged(self, state):
        checkbox = self.sender()
        row = checkbox.property("row")
        if state == Qt.Checked and row not in self.locked_rows_list:
            self.locked_rows_list.append(row)
        elif state == Qt.Unchecked and row in self.locked_rows_list:
            self.locked_rows_list.remove(row)
        self.sendSpikesData()

    def onSelectionChanged(self, selected, deselected):
        selection_model = self.tableView.selectionModel()
        selected_indexes = selection_model.selectedRows()
        self.selected_rows_list = [index.row() for index in selected_indexes]
        self.sendSpikesData()

    # ====================

    def sendSpikesData(self):
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

        spikes = {'current_wav_units': self.current_wav_units,
                  'current_showing_units': self.current_showing_units}
        self.signal_showing_spikes_data_changed.emit(spikes)

    def sendWaveformAction(self, checked):
        sender = self.sender().objectName()
        if checked:
            self.wav_actions_state[sender][1] = True
            self.exclusiveWaveformActions(sender)
            self.signal_activate_manual_mode.emit(True)

        else:
            self.wav_actions_state[sender][1] = False
            self.signal_activate_manual_mode.emit(False)

    def exclusiveWaveformActions(self, new_sender):
        for object_name in self.wav_actions_state.keys():
            if object_name != new_sender:
                if self.wav_actions_state[object_name][1] == True:
                    self.wav_actions_state[object_name][0].setChecked(False)
                    self.wav_actions_state[object_name][1] = False

    # ==================== Unit Actions ====================
    def mergeUnits(self):
        unit_IDs = self.current_showing_units
        if len(unit_IDs) < 2:
            print('Error: Merge Units action must have at least 2 units.')
            return

        target_unit_ID = unit_IDs[0]
        wav_index = np.where(np.isin(self.current_wav_units, unit_IDs))[0]

        self.moveWaveforms(wav_index, target_unit_ID)

        self.removeEmptyUnits()

        self.reorderUnitID()

        self.sendSpikesData()

        self.setDataModel()

        self.recoverySelection()

    def swapUnits(self):
        unit_IDs = self.current_showing_units
        if len(unit_IDs) != 2:
            print('Error: Can only swap 2 units.')
            return

        unit1_ID, unit2_ID = unit_IDs
        unit1_wav_index = np.where(
            np.isin(self.current_wav_units, [unit1_ID]))[0]
        unit2_wav_index = np.where(
            np.isin(self.current_wav_units, [unit2_ID]))[0]

        temp_unit_id = self.createNewUnit('Unit')

        self.moveWaveforms(unit1_wav_index, temp_unit_id)  # 1 to temp
        self.moveWaveforms(unit2_wav_index, unit1_ID)  # 2 to 1
        self.moveWaveforms(unit1_wav_index, unit2_ID)  # temp to 2

        self.removeEmptyUnits()

        self.reorderUnitID()

        self.sendSpikesData()

        self.setDataModel()

        self.recoverySelection()

    def removeUnits(self):
        unit_IDs = self.current_showing_units
        if len(unit_IDs) == 0:
            return
        wav_index = np.where(np.isin(self.current_wav_units, unit_IDs))[0]
        self.removeWaveforms(wav_index)

    def invalidateUnits(self):
        unit_IDs = self.current_showing_units
        if len(unit_IDs) == 0:
            return
        wav_index = np.where(np.isin(self.current_wav_units, unit_IDs))[0]
        self.invalidateWaveforms(wav_index)
    # ========================================

    # ==================== Waveform Actions ====================
    def addAsNewUnit(self, wav_index):
        new_unit_ID = self.createNewUnit('Unit')

        self.moveWaveforms(wav_index, new_unit_ID)

        self.removeEmptyUnits()

        self.reorderUnitID()

        self.sendSpikesData()

        self.setDataModel()

        self.recoverySelection()

    def addToSelectedUnit(self, wav_index):
        unit_ID_list = self.current_showing_units
        unit_name_list = self.df_table_data.loc[self.current_showing_units, 'Name']

        accepted, target_unit_name = self.askTargetUnit(
            unit_ID_list=unit_ID_list,
            unit_name_list=unit_name_list)
        if accepted:
            target_unit_mask = self.df_table_data['Name'] == target_unit_name
            target_unit_type = self.df_table_data['UnitType'][target_unit_mask].to_list()[
                0]

            if target_unit_type in ['Unsorted', 'Invalid']:
                warning_result = QMessageBox.warning(self,
                                                     "Warning",
                                                     f"The target unit is {target_unit_type} unit.\nDo you want to continue?",
                                                     QMessageBox.Ok | QMessageBox.Cancel)
                if warning_result != QMessageBox.Ok:
                    return

            target_unit_ID = self.df_table_data.index[target_unit_mask].tolist()[
                0]
        else:
            return

        self.moveWaveforms(wav_index, target_unit_ID)

        self.removeEmptyUnits()

        self.reorderUnitID()

        self.sendSpikesData()

        self.setDataModel()

        self.recoverySelection()

    def removeWaveforms(self, wav_index):
        unsorted_unit_ID = self.df_table_data[self.df_table_data['UnitType'] == 'Unsorted'].index.tolist(
        )
        if len(unsorted_unit_ID) < 1:
            unsorted_unit_ID = self.createNewUnit('Unsorted')
            self.moveWaveforms(wav_index, unsorted_unit_ID)

        elif len(unsorted_unit_ID) != 1:
            # TODO: Select one to move.
            print('TODO: Select one to move.')

        else:
            unsorted_unit_ID = unsorted_unit_ID[0]
            self.moveWaveforms(wav_index, unsorted_unit_ID)

        self.removeEmptyUnits()

        self.reorderUnitID()

        self.sendSpikesData()

        self.setDataModel()

        self.recoverySelection()

    def invalidateWaveforms(self, wav_index):
        invalid_unit_ID = self.df_table_data[self.df_table_data['UnitType'] == 'Invalid'].index.tolist(
        )
        if len(invalid_unit_ID) < 1:
            invalid_unit_ID = self.createNewUnit('Invalid')
            self.moveWaveforms(wav_index, invalid_unit_ID)

        elif len(invalid_unit_ID) != 1:
            # TODO: Select one to move.
            print('TODO: Select one to move.')

        else:
            invalid_unit_ID = invalid_unit_ID[0]
            self.moveWaveforms(wav_index, invalid_unit_ID)

        self.removeEmptyUnits()

        self.reorderUnitID()

        self.sendSpikesData()

        self.setDataModel()

        self.recoverySelection()
    # ========================================

    def askTargetUnit(self, unit_ID_list, unit_name_list):
        dialog = SelectTargetUnitDialog(unit_ID_list=unit_ID_list,
                                        unit_name_list=unit_name_list,
                                        color_palette=self.color_palette_list,
                                        parent=self)
        result = dialog.exec_()  # 显示对话框并获取结果

        if result == QDialog.Accepted:
            selected_option = dialog.comboBox.currentText()
            return (True, selected_option)
        else:
            return (False, None)

    def createNewUnit(self, unit_type):
        new_unit_ID = self.df_table_data.index.max() + 1
        new_unit_row = self.df_table_data['row'].max() + 1
        new_unit_name_suffix = f'_{unit_type}' if unit_type in [
            'Unsorted', 'Invalid'] else ''
        new_unit = pd.DataFrame({
            'Name': f'{self.spike_chan_name}_Unit_{new_unit_ID:02}{new_unit_name_suffix}',
            'NumRecords': 0,
            'UnitType': unit_type,
            'row': new_unit_row}, index=[new_unit_ID])
        self.appendUnitRow(
            new_unit.loc[new_unit_ID, 'row'], new_unit.loc[new_unit_ID, 'Name'])
        self.df_table_data = pd.concat([self.df_table_data, new_unit], axis=0)

        return new_unit_ID

    def removeEmptyUnits(self):
        excluded_units = self.df_table_data['UnitType'].isin(
            ['Unsorted', 'Invalid'])
        remove_units = self.df_table_data['NumRecords'] == 0

        self.df_table_data = self.df_table_data[
            (~remove_units) | excluded_units]

    def moveWaveforms(self, wav_index, target_unit):
        self.current_wav_units[wav_index] = target_unit

        # update NumRecords
        unique, counts = np.unique(self.current_wav_units, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.df_table_data['NumRecords'] = self.df_table_data.index.map(
            counts_dict)
        self.df_table_data['NumRecords'].fillna(0, inplace=True)
        self.df_table_data['NumRecords'] = self.df_table_data['NumRecords'].astype(
            int)

    def reorderUnitID(self):
        unsorted_units = self.df_table_data['UnitType'] == 'Unsorted'
        invalid_units = self.df_table_data['UnitType'] == 'Invalid'

        new_df_table_data = pd.concat(
            [self.df_table_data[unsorted_units], self.df_table_data[invalid_units]])
        new_df_table_data = pd.concat(
            [new_df_table_data, self.df_table_data[~(unsorted_units | invalid_units)]])
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
        self.current_wav_units = np.vectorize(
            map_new_ID_dict.get)(self.current_wav_units)

    def recoverySelection(self):
        model = self.tableView.model()
        for row in self.locked_rows_list:
            checkbox = self.tableView.indexWidget(
                model.index(row, 0)).layout().itemAt(0).widget()
            checkbox.setChecked(True)

        selection_model = self.tableView.selectionModel()
        for row in self.selected_rows_list:
            selection_model.select(model.index(row, 0),
                                   QItemSelectionModel.Select)
            selection_model.select(model.index(row, 1),
                                   QItemSelectionModel.Select)


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
