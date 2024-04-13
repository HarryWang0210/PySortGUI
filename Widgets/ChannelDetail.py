import logging

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                             QMainWindow, QUndoCommand, QUndoStack,
                             QVBoxLayout, QWidget)

from DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from UI.ChannelDetailv2_ui import Ui_ChannelDetail
from UI.ExtractWaveformSettings_ui import Ui_ExtractWaveformSettings

logger = logging.getLogger(__name__)


class ChannelDetail(QtWidgets.QWidget, Ui_ChannelDetail):
    signal_data_file_name_changed = QtCore.pyqtSignal(object)
    signal_continuous_data_changed = QtCore.pyqtSignal((object, object))
    signal_spike_data_changed = QtCore.pyqtSignal((object, bool))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.window_title = "Channel Detail"
        self.setupUi(self)
        self.current_data_object: SpikeSorterData | None = None
        self.current_raw_object: ContinuousData | None = None
        self.current_filted_object: ContinuousData | None = None
        self.current_spike_object: DiscreteData | None = None

        self.header_name = ['ID', 'Label', 'Name', 'NumUnits',
                            'NumRecords', 'LowCutOff', 'HighCutOff', 'Reference', 'Threshold', 'Type']
        self.raws_header = None
        self.spikes_header = None
        self.events_header = None
        self.undo_stack_dict = dict()
        self.current_undo_stack: QUndoStack = None
        self.initDataModel()
        self.setupConnections()

    def setupConnections(self):
        self.open_file_toolButton.released.connect(self.openFile)
        self.extract_wav_setting_toolButton.released.connect(
            self.setExtractWaveformParams)

        self.extract_wav_pushButton.clicked.connect(self.extractWaveforms)
        self.sort_channel_pushButton.clicked.connect(self.sortChannel)

        self.treeView.setSelectionMode(QAbstractItemView.SingleSelection)
        selection_model = self.treeView.selectionModel()
        selection_model.selectionChanged.connect(self.onSelectionChanged)

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(self.header_name)
        self.treeView.setModel(model)

        # for col in range(model.columnCount()):
        #     self.treeView.resizeColumnToContents(col)

    def setDataModel(self):
        model = self.treeView.model()
        model.clear()
        model.setHorizontalHeaderLabels(self.header_name)
        self.spikes_header = self.current_data_object.spikes_header
        self.raws_header = self.current_data_object.raws_header
        self.events_header = self.current_data_object.events_header
        if not self.spikes_header is None:
            spikes_header = self.spikes_header.copy()
            for nan_column in [x for x in self.header_name if x not in spikes_header.columns]:
                spikes_header[nan_column] = ''
            spikes_header['Reference'] = spikes_header['ReferenceID']
            spikes_header = spikes_header[self.header_name]

            self.spike_group_item = QStandardItem('Spikes')  # Top level, Group
            model.appendRow(self.spike_group_item)

            for chan_ID in spikes_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = spikes_header[spikes_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                self.spike_group_item.appendRow(first_items)

                for sub_row in sub_data.iloc[1:, 1:].values:
                    values = [QStandardItem("")] + \
                        [QStandardItem(str(col_value))
                         for col_value in sub_row]
                    chan_ID_item.appendRow(values)

        if not self.raws_header is None:
            raws_header = self.raws_header.copy()
            for nan_column in [x for x in self.header_name if x not in raws_header.columns]:
                raws_header[nan_column] = ''
            raws_header = raws_header[self.header_name]

            group_item = QStandardItem('Raws')  # Top level, Group
            model.appendRow(group_item)

            for chan_ID in raws_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = raws_header[raws_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                group_item.appendRow(first_items)

        if not self.events_header is None:
            events_header = self.events_header.copy()
            for nan_column in [x for x in self.header_name if x not in events_header.columns]:
                events_header[nan_column] = ''
            events_header = events_header[self.header_name]

            group_item = QStandardItem('Events')  # Top level, Group
            model.appendRow(group_item)

            for chan_ID in events_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = events_header[events_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                group_item.appendRow(first_items)

    def createRowItems(self, spike_object: DiscreteData) -> list[QStandardItem]:
        row_items = []
        for key in self.header_name:
            if key == 'Reference':
                key = 'ReferenceID'
            # if key == 'Label':
            #     label_item = QStandardItem(
            #         str(spike_object.header.get(key, '')))
            #     row_items.append(label_item)
            #     continue

            row_items.append(QStandardItem(
                str(spike_object.header.get(key, ''))))
        return row_items

    def rowItemsToMetadata(self, row_items: list[QStandardItem]) -> dict[str, str]:
        meta_data = [item.text() for item in row_items]
        meta_data = dict(zip(self.header_name, meta_data))
        return meta_data

    def dropSuffix(self, text: str, suffix: str = '*') -> str:
        return text[:-1] if text.endswith(suffix) else text

    def getSelectedRowItems(self) -> list:
        model = self.treeView.model()
        selection_model = self.treeView.selectionModel()
        selected_indexes = selection_model.selectedIndexes()

        items = [model.itemFromIndex(ind) for ind in selected_indexes]
        if items == []:  # No select
            return
        if items[0].parent() is None:  # Group
            return
        elif not items[0].parent().parent() is None:  # Label
            items[0] = items[0].parent()

        return items

    def getRowItemsFromChannel(self, channel: int) -> list:
        num_row = self.spike_group_item.rowCount()
        num_col = self.spike_group_item.columnCount()

        channel_items = [self.spike_group_item.child(
            row, 0) for row in range(num_row)]
        channel_IDs = [item.text()[:-1] if item.text().endswith('*')
                       else item.text() for item in channel_items]
        row = channel_IDs.index(str(channel))

        result = [[self.spike_group_item.child(row, col)
                   for col in range(num_col)]]
        logger.debug([item.text() for item in result[0]])
        if result[0][0].hasChildren():
            ID_item = result[0][0]
            for row in range(ID_item.rowCount()):
                temp = [ID_item.child(row, col)
                        for col in range(ID_item.columnCount())]
                temp[0] = ID_item
                result.append(temp)
                logger.debug([item.text() for item in result[-1]])
        return result

    # ========== Slot ==========
    def updating_spike_data(self, new_spike_object: DiscreteData | None):
        # import pandas as pd
        # logger.debug('Spike header')
        # logger.debug('\n' +
        #              '\n'.join([f'     {k}: {new_spike_object.header[k]}' for k in new_spike_object.header]))
        logger.debug(new_spike_object)

        if new_spike_object is None:
            return

        row_items = self.getSelectedRowItems()
        if row_items is None:
            return

        if new_spike_object is self.current_spike_object:
            return

        command = ChangeSpikeCommand(text="Manual unit",
                                     widget=self,
                                     raw_object=self.current_raw_object,
                                     old_filted_object=self.current_filted_object,
                                     new_filted_object=self.current_filted_object,
                                     old_spike_object=self.current_spike_object,
                                     new_spike_object=new_spike_object)
        self.current_spike_object = new_spike_object
        self.current_undo_stack.push(command)

    def onSelectionChanged(self, selected, deselected):
        row_items = self.getSelectedRowItems()
        if row_items is None:
            return
        meta_data = self.rowItemsToMetadata(row_items)

        self.current_raw_object = None
        self.current_filted_object = None
        self.current_spike_object = None
        chan_ID = int(self.dropSuffix(meta_data["ID"]))
        label = self.dropSuffix(meta_data["Label"])
        logger.info(f'Selected type: {meta_data["Type"]}')

        if meta_data['Type'] == 'Spikes':
            # raw
            self.current_data_object.loadRaw(channel=chan_ID)
            new_raw_object = self.current_data_object.getRaw(chan_ID)
            # spike
            new_spike_object = self.current_data_object.getSpike(channel=chan_ID,
                                                                 label=label)
            if new_spike_object == 'Removed':
                # This spike is removed but unsaved
                new_raw_object = None
                new_filted_object = None
                new_spike_object = None
            else:
                self.current_data_object.loadSpike(
                    channel=chan_ID, label=label)
                # Use spike to generate filted
                new_filted_object = self.current_data_object.subtractReference(
                    channel=chan_ID, reference=new_spike_object.reference)
                new_filted_object = new_filted_object.bandpassFilter(low=new_spike_object.low_cutoff,
                                                                     high=new_spike_object.high_cutoff)
                new_filted_object = new_filted_object.createCopy(
                    threshold=new_spike_object.threshold)
                # self.setSpikeSetting()
                self.setLabelCombox(labels=new_raw_object.spikes,
                                    current=label)

            if not self.current_undo_stack is None:
                self.current_undo_stack.setActive(False)

            if not chan_ID in self.undo_stack_dict:
                self.current_undo_stack = QUndoStack(
                    self.main_window.undo_group)
                chan_ID_dict = {label: self.current_undo_stack}
                self.undo_stack_dict[chan_ID] = chan_ID_dict

            elif not label in self.undo_stack_dict[chan_ID]:
                self.current_undo_stack = QUndoStack(
                    self.main_window.undo_group)
                self.undo_stack_dict[chan_ID][label] = self.current_undo_stack

            self.current_raw_object = new_raw_object
            self.current_filted_object = new_filted_object
            self.current_spike_object = new_spike_object
            # logger.debug(self.current_undo_stack)
            self.current_undo_stack = self.undo_stack_dict[chan_ID][label]
            self.current_undo_stack.setActive(True)
            logger.debug(
                f'Undostack {chan_ID} {label}: {self.current_undo_stack}')

        elif meta_data['Type'] == 'Raws':
            self.current_data_object.loadRaw(channel=chan_ID)
            self.current_raw_object = self.current_data_object.getRaw(chan_ID)
            self.setLabelCombox(labels=self.current_raw_object.spikes)

        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object, True)

    # ========== Actions ==========
    def openFile(self):
        """Open file manager and load selected file. """
        self.file_type_dict = {  # "openephy": "Open Ephys Format (*.continuous)",
            "pyephys": "pyephys format (*.h5)"}  # File types to load
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "./",
                                                                   ";;".join(self.file_type_dict.values()))  # start path
        if filename == "":
            return

        if isinstance(self.current_data_object, SpikeSorterData):
            if filename == self.current_data_object.filename:
                return

        self.current_data_object = SpikeSorterData(filename)
        self.signal_data_file_name_changed.emit(self.current_data_object)

        self.setDataModel()
        self.undo_stack_dict = dict()

        # Clear all undo stack
        for undo_stack in self.main_window.undo_group.stacks():
            self.main_window.undo_group.removeStack(undo_stack)

        self.sorting_label_comboBox.clear()
        self.file_name_lineEdit.setText(filename)

    def copySpike(self):
        row_items = self.getSelectedRowItems()
        meta_data = self.rowItemsToMetadata(row_items)
        row_type = meta_data['Type']

        if row_type != 'Spikes':
            return

        channel_ID = int(self.dropSuffix(meta_data['ID']))
        label = self.dropSuffix(meta_data['Label'])
        raw_object = self.current_data_object.getRaw(channel_ID)
        spike_object = self.current_raw_object.getSpike(label)
        new_spike_object = spike_object.createCopy()

        i = 1
        while True:
            new_label = f'label{i}'
            if new_label in raw_object.spikes:
                i += 1
            else:
                break

        new_spike_object.setLabel(new_label)
        raw_object.setSpike(new_spike_object, new_label)

        channel_item = row_items[0]

        new_row_items = self.createRowItems(new_spike_object)
        new_row_items[0] = QStandardItem('')
        label_item = new_row_items[1]

        channel_item.appendRow(new_row_items)

        selection_model = self.treeView.selectionModel()
        selection_model.select(
            label_item.index(), QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect)
        self.treeView.scrollTo(label_item.index())

        row_items = self.getSelectedRowItems()
        if row_items is None:
            return

        self.updataSpikeInfo(row_items, new_spike_object)
        self.setUnsavedChangeIndicator(row_items, new_spike_object)

    def deleteSpike(self):
        row_items = self.getSelectedRowItems()
        meta_data = self.rowItemsToMetadata(row_items)
        row_type = meta_data['Type']

        if row_type != 'Spikes':
            return

        command = DeleteSpikeCommand(text="Delete spike",
                                     widget=self,
                                     raw_object=self.current_raw_object,
                                     old_filted_object=self.current_filted_object,
                                     old_spike_object=self.current_spike_object)
        self.current_undo_stack.push(command)

        # if label_item.parent() is spike_group_item:
        #     # deleting node
        #     if ID_item.hasChildren():
        #         # has leaves
        #         channel_ID = int(self.dropSuffix(ID_item.text()))
        #         channel_row_items = self.getRowItemsFromChannel(channel_ID)
        #         next_row_items = channel_row_items[1]
        #         for i in range(1, len(row_items)):
        #             old = row_items[i]
        #             new = next_row_items[i]
        #             old.setText(new.text())
        #             old.setFont(new.font())
        #         row = next_row_items[1].row()
        #         ID_item.removeRow(row)
        #     else:
        #         # no leaves
        #         spike_group_item.removeRow(ID_item.row())
        # else:
        #     # deleting leaf
        #     ID_item.removeRow(label_item.row())

        # self.signal_data_file_name_changed.emit(self.current_data_object)

    def setExtractWaveformParams(self):
        if self.current_data_object is None:
            logger.warning('No SpikeSorterData.')
            return
        if self.current_raw_object is None:
            logger.warning('No selected channel.')
            return
        all_chan_ID = self.current_data_object.channel_IDs

        dialog = ExtractWaveformSettingsDialog(
            filted_object=self.current_filted_object,
            all_chan_ID=all_chan_ID,
            parent=self)

        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        if dialog.channel_ref_radioButton.isChecked():
            ref = int(dialog.channel_ref_comboBox.currentText())

        elif dialog.median_ref_radioButton.isChecked():
            ref = [int(i)
                   for i in dialog.show_channels_lineEdit.text().split(', ')]

        low = dialog.filter_low_doubleSpinBox.value()
        high = dialog.filter_high_doubleSpinBox.value()

        new_filted_object = self.preprocessing(ref, low, high)
        if dialog.const_thr_radioButton.isChecked():
            threshold = dialog.const_thr_doubleSpinBox.value()
        elif dialog.mad_thr_radioButton.isChecked():
            threshold = new_filted_object.estimated_sd * \
                dialog.mad_thr_doubleSpinBox.value()

        new_filted_object = new_filted_object.createCopy(threshold=threshold)
        new_spike_object = None

        # command = ChangeFilterCommand("Change filter",
        #                               self,
        #                               self.current_raw_object,
        #                               self.current_filted_object,
        #                               new_filted_object,
        #                               self.current_spike_object,
        #                               new_spike_object)

        self.current_filted_object = new_filted_object
        # self.current_spike_object = new_spike_object
        # self.current_undo_stack.push(command)

        # self.current_filted_object = self.current_filted_object.createCopy(
        #     threshold=threshold)
        # self.current_spike_object = None
        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(new_spike_object, True)

    def preprocessing(self, ref: list = [],
                      low: int | float = None,
                      high: int | float = None) -> ContinuousData:
        new_filted_object = self.current_raw_object.createCopy()
        # if len(ref) > 0:
        new_filted_object = self.current_data_object.subtractReference(channel=self.current_raw_object.channel_ID,
                                                                       reference=ref)
        if not low is None and not high is None:
            new_filted_object = new_filted_object.bandpassFilter(
                low, high)
        return new_filted_object

    def extractWaveforms(self):
        row_items = self.getSelectedRowItems()
        # model = self.treeView.model()
        # # logger.debug(model)

        # selection_model = self.treeView.selectionModel()
        # selected_indexes = selection_model.selectedIndexes()
        # # selected_rows_list = [index.row() for index in selected_indexes]
        # # logger.debug(selected_indexes)
        # items = [model.itemFromIndex(ind) for ind in selected_indexes]

        # if items[0].parent() == None:  # Group
        #     return
        # elif items[0].parent().parent() != None:  # Label
        #     items[0] = items[0].parent()

        meta_data = [item.text() for item in row_items]
        meta_data = dict(zip(self.header_name, meta_data))
        row_type = meta_data['Type']

        new_spike_object = self.current_filted_object.extractWaveforms(
            self.current_filted_object.threshold)

        if row_type == 'Spikes':
            label = meta_data['Label']
            label = label[:-1] if label.endswith('*') else label
            new_spike_object.setLabel(label)

            old_filted_object = self.current_data_object.subtractReference(
                channel=self.current_spike_object.channel_ID, reference=self.current_spike_object.reference)
            old_filted_object = old_filted_object.bandpassFilter(low=self.current_spike_object.low_cutoff,
                                                                 high=self.current_spike_object.high_cutoff)
            old_filted_object = old_filted_object.createCopy(
                threshold=self.current_spike_object.threshold)
            command = ChangeSpikeCommand(text="Extract waveform",
                                         widget=self,
                                         raw_object=self.current_raw_object,
                                         old_filted_object=old_filted_object,
                                         new_filted_object=self.current_filted_object,
                                         old_spike_object=self.current_spike_object,
                                         new_spike_object=new_spike_object)

            self.current_spike_object = new_spike_object
            self.current_undo_stack.push(command)

            self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                     self.current_filted_object)
            self.signal_spike_data_changed.emit(self.current_spike_object,
                                                True)

        elif row_type == 'Raws':
            chan_ID = meta_data['ID']
            chan_ID = int(chan_ID[:-1]) \
                if chan_ID.endswith('*') else int(chan_ID)

            # add new spike row
            if self.current_raw_object.spikes == []:
                label = 'default'
                new_spike_object.setLabel(label)
                self.current_raw_object.setSpike(new_spike_object, label)

                num_row = self.spike_group_item.rowCount()
                num_col = self.spike_group_item.columnCount()

                channel_items = [self.spike_group_item.child(row, 0)
                                 for row in range(num_row)]
                channel_IDs = [int(item.text()[:-1])
                               if item.text().endswith('*') else int(item.text())
                               for item in channel_items] + [chan_ID]
                channel_IDs.sort()
                new_row = channel_IDs.index(chan_ID)

                new_row_items = self.createRowItems(new_spike_object)
                label_item = new_row_items[1]
                self.spike_group_item.insertRow(new_row, new_row_items)

            else:
                i = 1
                while True:
                    label = f'label{i}'
                    if label in self.current_raw_object.spikes:
                        i += 1
                    else:
                        break

                new_spike_object.setLabel(label)
                self.current_raw_object.setSpike(new_spike_object, label)

                num_row = self.spike_group_item.rowCount()
                num_col = self.spike_group_item.columnCount()

                row_items_list = self.getRowItemsFromChannel(chan_ID)
                channel_item = row_items_list[0][0]

                new_row_items = self.createRowItems(new_spike_object)
                new_row_items[0] = QStandardItem('')
                label_item = new_row_items[1]

                channel_item.appendRow(new_row_items)

            selection_model = self.treeView.selectionModel()
            selection_model.select(
                label_item.index(), QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect)
            self.treeView.scrollTo(label_item.index())

            row_items = self.getSelectedRowItems()
            if row_items is None:
                return

            self.updataSpikeInfo(row_items, new_spike_object)
            self.setUnsavedChangeIndicator(row_items, new_spike_object)
            # if len(row_items_list) == 1:
            # select row
        # row_items = self.getSelectedRowItems()
        # if row_items is None:
        #     return
        # self.setUnsavedChangeIndicator(row_items, self.current_spike_object)
        # logger.debug(type(self.current_filted_object))

        # logger.debug(self.current_spike_object)

    def sortChannel(self):
        new_spike_object = self.current_spike_object.autosort()
        # self.signal_spike_data_changed.emit(self.current_spike_object, True)

        command = ChangeSpikeCommand(text="Autosort",
                                     widget=self,
                                     raw_object=self.current_raw_object,
                                     old_filted_object=self.current_filted_object,
                                     new_filted_object=self.current_filted_object,
                                     old_spike_object=self.current_spike_object,
                                     new_spike_object=new_spike_object)
        self.current_spike_object = new_spike_object
        self.current_undo_stack.push(command)

        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object, True)
        # row_items = self.getSelectedRowItems()
        # if row_items is None:
        #     return
        # self.setUnsavedChangeIndicator(row_items, self.current_spike_object)

    def setLabelCombox(self, labels: list | None = None, current: str | None = None):
        self.sorting_label_comboBox.clear()
        if labels is None:
            return
        self.sorting_label_comboBox.addItems(labels)
        if current is None:
            return
        self.sorting_label_comboBox.setCurrentText(current)

    # def setSpikeSetting(self):
    #     self.current_spike_setting['Reference'] = ('Single',
    #                                                [self.current_spike_object.reference])
    #     self.current_spike_setting['Filter'] = (self.current_spike_object.low_cutoff,
    #                                             self.current_spike_object.high_cutoff)
    #     self.current_spike_setting['Threshold'] = ('MAD',
    #                                                self.current_spike_object.threshold / self.current_filted_object.estimated_sd)

    def handleUndoRedo(self, action_type: str,
                       new_raw_object: ContinuousData,
                       new_filted_object: ContinuousData,
                       new_spike_object: DiscreteData | None):
        reset_selection = False
        row_items = self.getSelectedRowItems()
        if row_items is None:
            logger.warning('No selected row.')
            return

        if action_type == 'Delete spike':
            ID_item = row_items[0]
            label_item = row_items[1]

            label_item.setText(self.dropSuffix(label_item.text()))
            # font = label_item.font()
            # font.setStrikeOut(True)
            # label_item.setFont(font)
            for item in row_items[1:]:
                font = ID_item.font()
                font.setBold(False)
                item.setFont(font)
                item.setForeground(QColor('darkGray'))

            # Check ID
            if not ID_item.text().endswith('*'):
                ID_item.setText(ID_item.text() + '*')
            font = ID_item.font()
            font.setBold(True)
            ID_item.setFont(font)

            self.current_raw_object = None
            self.current_filted_object = None
            self.current_spike_object = None
            reset_selection = True

        else:
            if new_spike_object is None:
                logger.warning('No spike data.')
                return

            if action_type == 'Recovery spike':
                ID_item = row_items[0]
                label_item = row_items[1]

                # font = label_item.font()
                # font.setStrikeOut(False)
                # label_item.setFont(font)
                for item in row_items[1:]:
                    item.setForeground(QColor('black'))

            logger.debug('undo/redo')
            self.updataSpikeInfo(row_items, new_spike_object)
            self.setUnsavedChangeIndicator(row_items, new_spike_object)

            if (new_spike_object is self.current_spike_object) and \
                    (new_filted_object is self.current_filted_object):
                return

            self.current_raw_object = new_raw_object
            self.current_filted_object = new_filted_object
            self.current_spike_object = new_spike_object

        if action_type in ['Recovery spike', 'Extract waveform']:
            reset_selection = True
        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object,
                                            reset_selection)

    def saveChannel(self):
        row_items = self.getSelectedRowItems()
        if row_items is None:
            return
        ID_item = row_items[0]
        selecting_label = self.dropSuffix(row_items[1].text())
        new_selecting_item = None
        channel_row = ID_item.row()
        spike_group_item = ID_item.parent()

        channel_ID = int(self.dropSuffix(ID_item.text()))
        self.current_data_object.saveChannel(channel_ID)

        raw_object = self.current_data_object.getRaw(channel_ID)
        labels = sorted(raw_object.spikes)

        # Modify row items
        spike_group_item.removeRow(channel_row)
        if len(labels) > 0:
            spike_object = raw_object.getSpike(labels[0])
            first_row_items = self.createRowItems(spike_object)

            new_ID_item = first_row_items[0]
            for label in labels[1:]:
                spike_object = raw_object.getSpike(label)
                new_row_items = self.createRowItems(spike_object)
                new_row_items[0] = QStandardItem('')
                if new_row_items[1].text() == selecting_label:
                    new_selecting_item = new_row_items[1]
                new_ID_item.appendRow(new_row_items)
            spike_group_item.insertRow(channel_row, first_row_items)

            if new_selecting_item is None:
                new_selecting_item = first_row_items[1]

        # Undo stack
        channel_undo_stack_dict = self.undo_stack_dict[channel_ID]
        labels_in_undo_stack = list(channel_undo_stack_dict.keys())
        for label in labels_in_undo_stack:
            if not label in labels:
                undo_stack = channel_undo_stack_dict[label]
                self.main_window.undo_group.removeStack(undo_stack)
                del channel_undo_stack_dict[label]

        if new_selecting_item:
            selection_model = self.treeView.selectionModel()
            selection_model.select(
                new_selecting_item.index(), QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect)
            self.treeView.scrollTo(new_selecting_item.index())

        # row_items_list = self.getRowItemsFromChannel(channel_ID)
        # for row_items in row_items_list:
        #     label_item = row_items[1]
        #     label = self.dropSuffix(label_item.text())
        #     spike_object = raw_object.getSpike(label)
        #     self.setUnsavedChangeIndicator(row_items, spike_object)

        # if label_item.parent() is spike_group_item:
        #     # deleting node
        #     if ID_item.hasChildren():
        #         # has leaves
        #         channel_ID = int(self.dropSuffix(ID_item.text()))
        #         channel_row_items = self.getRowItemsFromChannel(channel_ID)
        #         next_row_items = channel_row_items[1]
        #         for i in range(1, len(row_items)):
        #             old = row_items[i]
        #             new = next_row_items[i]
        #             old.setText(new.text())
        #             old.setFont(new.font())
        #         row = next_row_items[1].row()
        #         ID_item.removeRow(row)
        #     else:
        #         # no leaves
        #         spike_group_item.removeRow(ID_item.row())
        # else:
        #     # deleting leaf
        #     ID_item.removeRow(label_item.row())

    def setUnsavedChangeIndicator(self, row_items: list, spike_object: DiscreteData):
        if spike_object is None:
            return
        ID_item = row_items[0]
        label_item = row_items[1]
        meta_items = row_items[2:]
        if spike_object._from_file:
            if label_item.text().endswith('*'):
                label_item.setText(label_item.text()[:-1])

            for item in row_items[1:]:
                font = item.font()
                font.setBold(False)
                item.setFont(font)

        else:
            if not ID_item.text().endswith('*'):
                ID_item.setText(ID_item.text() + '*')

            if not label_item.text().endswith('*'):
                label_item.setText(label_item.text() + '*')

            for item in row_items:
                font = item.font()
                font.setBold(True)
                item.setFont(font)

        # Check ID
        channel_ID = spike_object.channel_ID
        raw_object = self.current_data_object.getRaw(channel_ID)
        if raw_object.allSaved():
            if ID_item.text().endswith('*'):
                ID_item.setText(ID_item.text()[:-1])
                logger.debug(raw_object.allSaved())
            font = ID_item.font()
            font.setBold(False)
            ID_item.setFont(font)
        else:
            if not ID_item.text().endswith('*'):
                ID_item.setText(ID_item.text() + '*')
            font = ID_item.font()
            font.setBold(True)
            ID_item.setFont(font)

        # row_index = 1  # 第二行
        # col_index = 2  # 第三列
        # new_value = "New Value"

        # item = model.item(row_index, col_index)
        # logger.debug(item.text())
        # if item is not None:
        #     item.setData(new_value, QtCore.Qt.DisplayRole)

    def updataSpikeInfo(self, row_items, spike_object: DiscreteData):
        for index, key in enumerate(self.header_name):
            if key == 'Reference':
                key = 'ReferenceID'
            row_items[index].setText(str(spike_object.header.get(key, '')))


class DeleteSpikeCommand(QUndoCommand):
    def __init__(self, text, widget, raw_object: ContinuousData,
                 old_filted_object: ContinuousData,
                 old_spike_object: DiscreteData):
        super().__init__(text)
        self.widget = widget
        self.action_type = 'Delete spike'
        self.raw_object = raw_object
        self.old_filted_object = old_filted_object
        self.old_spike_object = old_spike_object

    def redo(self):
        self.raw_object.removeSpike(self.old_spike_object.label)
        self.widget.handleUndoRedo(
            'Delete spike', self.raw_object, None, None)
        logger.info(
            f"Redo: {self.text()}, "
            f"Spike: {self.old_spike_object.channel_ID} {self.old_spike_object.label}")
        logger.debug(
            f"Redo: {self.text()}, Data: {None}")

    def undo(self):
        self.raw_object.setSpike(
            self.old_spike_object, self.old_spike_object.label)
        self.widget.handleUndoRedo(
            'Recovery spike', self.raw_object, self.old_filted_object, self.old_spike_object)
        logger.info(
            f"Undo: {self.text()}, "
            f"Spike: {self.old_spike_object.channel_ID} {self.old_spike_object.label}")
        logger.debug(
            f"Undo: {self.text()}, Data: {self.old_spike_object}")

        # self.raw_object.setSpike(
        #     self.old_spike_object, self.old_spike_object.label)
        # logger.info(
        #     f"Undo: {self.text()}, Data: {self.old_spike_object}")


class ChangeSpikeCommand(QUndoCommand):
    def __init__(self, text, widget, raw_object: ContinuousData,
                 old_filted_object: ContinuousData, new_filted_object: ContinuousData,
                 old_spike_object: DiscreteData, new_spike_object: DiscreteData):
        super().__init__(text)
        self.widget = widget
        self.action_type = text
        self.raw_object = raw_object
        self.old_filted_object = old_filted_object
        self.new_filted_object = new_filted_object
        self.old_spike_object = old_spike_object
        self.new_spike_object = new_spike_object

    def redo(self):
        # 在这里执行操作，修改应用程序状态
        self.raw_object.setSpike(
            self.new_spike_object, self.new_spike_object.label)
        self.widget.handleUndoRedo(
            self.action_type, self.raw_object, self.new_filted_object, self.new_spike_object)

        logger.info(
            f"Redo: {self.text()} {self.new_spike_object}")

    def undo(self):
        # 撤销操作，回滚应用程序状态
        self.raw_object.setSpike(
            self.old_spike_object, self.old_spike_object.label)
        self.widget.handleUndoRedo(
            self.action_type, self.raw_object, self.old_filted_object, self.old_spike_object)

        logger.info(
            f"Undo: {self.text()} {self.old_spike_object}")


class ExtractWaveformSettingsDialog(Ui_ExtractWaveformSettings, QDialog):
    def __init__(self, filted_object: ContinuousData | None, all_chan_ID: list, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Extract Waveform Settings Dialog")
        self.default_spike_setting = {
            'Reference': [0],
            'Filter': (250, 6000),
            'Threshold': ('MAD', -3)
        }

        if filted_object is None:
            self.setting = self.default_spike_setting
            logger.debug(filted_object)
        else:
            self.setting = {
                'Reference': filted_object.reference,
                'Filter': (filted_object.low_cutoff, filted_object.high_cutoff),
                'Threshold': ('MAD', filted_object.threshold / filted_object.estimated_sd)
            }
        self.channel_ref_comboBox.clear()
        self.channel_ref_comboBox.addItems(map(str, all_chan_ID))
        self.initSetting()

    def initSetting(self):
        # if len(self.setting['Reference']) == 1:
        self.channel_ref_radioButton.setChecked(True)
        self.channel_ref_comboBox.setCurrentText(
            str(self.setting['Reference']))
        # elif len(self.setting['Reference']) > 1:
        #     logger.critical('Use median of channels: Not implemented error')

        self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][0])
        self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][1])

        if self.setting['Threshold'][0] == 'MAD':
            self.mad_thr_radioButton.setChecked(True)
            self.mad_thr_doubleSpinBox.setValue(self.setting['Threshold'][1])

        elif self.setting['Threshold'][0] == 'Const':
            self.const_thr_radioButton.setChecked(True)
            self.const_thr_doubleSpinBox.setValue(self.setting['Threshold'][1])
