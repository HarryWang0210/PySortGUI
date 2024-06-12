import logging
import os

import numpy as np
import seaborn as sns
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QItemSelectionModel, Qt
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel, QPalette
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QColorDialog, QDialog, QMainWindow, QMessageBox,
                             QStyledItemDelegate, QUndoCommand, QUndoStack,
                             QVBoxLayout, QWidget)

from pysortgui.DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from pysortgui.UI.ChannelDetailv2_ui import Ui_ChannelDetail
from pysortgui.UI.CreateReferenceDialog_ui import Ui_CreateReferenceDialog
from pysortgui.UI.ExtractWaveformSettings_ui import Ui_ExtractWaveformSettings
from pysortgui.UI.SelectEventsDialog_ui import Ui_SelectEventsDialog
from pysortgui.UI.SetBackgroundChannelDialog_ui import Ui_SetBackgroundChannelDialog
from pysortgui.Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class ChannelDetail(WidgetsInterface, Ui_ChannelDetail):
    signal_data_file_name_changed = QtCore.pyqtSignal(object)
    signal_continuous_data_changed = QtCore.pyqtSignal((object, object))
    signal_spike_data_changed = QtCore.pyqtSignal((object, bool))
    signal_event_data_changed = QtCore.pyqtSignal(object)
    signal_showing_events_changed = QtCore.pyqtSignal(list)
    signal_background_continuous_data_changed = QtCore.pyqtSignal(
        (object, object, bool))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.window_title = "Channel Detail"
        self.setupUi(self)
        self.default_root_folder = './'
        self.default_spike_setting = {
            'Reference': (True, 0),
            'Filter': (250, 6000),
            'ThresholdType': 'MAD',
            'MADFactor': -3,
            'ConstThreshold': 0
        }
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

        self.header_name = ['ID', 'Label', 'Name', 'NumUnits',
                            'NumRecords', 'LowCutOff', 'HighCutOff', 'Reference', 'Threshold', 'Type']
        self.raws_header = None
        self.spikes_header = None
        self.events_header = None
        self.undo_stack_dict: dict[tuple[int, str], QUndoStack] = dict()
        self.current_undo_stack: QUndoStack = None
        self.current_showing_events: list = []
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

        self.spike_group_item = QStandardItem('Spikes')  # Top level, Group
        self.raws_group_item = QStandardItem('Raws')  # Top level, Group
        self.events_group_item = QStandardItem('Events')  # Top level, Group

        model.appendRow(self.spike_group_item)
        if not self.spikes_header is None:
            spikes_header = self.spikes_header.copy()
            for nan_column in [x for x in self.header_name if x not in spikes_header.columns]:
                spikes_header[nan_column] = ''
            spikes_header['Reference'] = spikes_header['ReferenceID']
            spikes_header = spikes_header[self.header_name]

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

        model.appendRow(self.raws_group_item)
        if not self.raws_header is None:
            raws_header = self.raws_header.copy()
            for nan_column in [x for x in self.header_name if x not in raws_header.columns]:
                raws_header[nan_column] = ''
            raws_header = raws_header[self.header_name]

            for chan_ID in raws_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = raws_header[raws_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                self.raws_group_item.appendRow(first_items)

        model.appendRow(self.events_group_item)
        if not self.events_header is None:
            events_header = self.events_header.copy()
            for nan_column in [x for x in self.header_name if x not in events_header.columns]:
                events_header[nan_column] = ''
            events_header = events_header[self.header_name]

            for chan_ID in events_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = events_header[events_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                self.events_group_item.appendRow(first_items)

    def createRowItems(self, header: dict) -> list[QStandardItem]:
        row_items = []
        for key in self.header_name:
            if key == 'Reference':
                key = 'ReferenceID'

            row_items.append(QStandardItem(
                str(header.get(key, ''))))
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
        new_raw_object = None
        new_filted_object = None
        new_spike_object = None
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
                    channel_ID=chan_ID, reference_ID=new_spike_object.reference)
                new_filted_object = new_filted_object.bandpassFilter(low=new_spike_object.low_cutoff,
                                                                     high=new_spike_object.high_cutoff)
                new_filted_object = new_filted_object.createCopy(
                    threshold=new_spike_object.threshold)

            self.current_undo_stack = self.undo_stack_dict.get(
                (chan_ID, label))

            if self.current_undo_stack is None:
                self.current_undo_stack = QUndoStack(
                    self.main_window.undo_group)
                self.undo_stack_dict[(chan_ID,
                                      label)] = self.current_undo_stack

            self.current_undo_stack.setActive(True)

        elif meta_data['Type'] in ['Raws', 'Ref']:
            new_raw_object = self.current_data_object.getRaw(chan_ID)
            if new_raw_object == 'Removed':
                # This spike is removed but unsaved
                new_raw_object = None
                new_filted_object = None
                new_spike_object = None
            elif new_raw_object._from_file:
                self.current_data_object.loadRaw(channel=chan_ID)
                self.updataTreeView(row_items, new_raw_object)

        self.current_raw_object = new_raw_object
        self.current_filted_object = new_filted_object
        self.current_spike_object = new_spike_object

        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object, True)

        if self.current_event_object is None:
            # Events
            event_IDs = self.current_data_object.event_IDs
            new_event_object = None
            if event_IDs != []:
                new_event_object = self.current_data_object.getEvent(
                    event_IDs[0])
            if new_event_object is None:
                logger.warning('No events data.')
                return
            new_event_object._loadData()
            self.current_event_object = new_event_object
            self.current_showing_events = np.unique(
                self.current_event_object.unit_IDs).tolist()

        self.signal_event_data_changed.emit(self.current_event_object)
        self.signal_showing_events_changed.emit(self.current_showing_events)

    # ========== Actions ==========
    def openFile(self, filename: str = ''):
        """Open file manager and load selected file. """
        self.file_type_dict = {
            # "openephy": "Open Ephys Format (*.continuous)",
            "pyephys": "pyephys format (*.h5)"
        }  # File types to load
        if filename == "":
            filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Open file",
                                                                       self.default_root_folder,
                                                                       ";;".join(self.file_type_dict.values()))  # start path
        if filename == "":
            return
        if isinstance(self.current_data_object, SpikeSorterData):
            if filename == self.current_data_object.path:
                return
        self.current_data_object = SpikeSorterData(filename, 'pyephys')
        self.current_raw_object = None
        self.current_filted_object = None
        self.current_spike_object = None
        self.current_event_object = None
        self.default_root_folder = os.path.split(
            self.current_data_object.path)[0]
        self.signal_data_file_name_changed.emit(self.current_data_object)

        self.setDataModel()
        self.undo_stack_dict.clear()

        # Clear all undo stack
        for undo_stack in self.main_window.undo_group.stacks():
            self.main_window.undo_group.removeStack(undo_stack)

        # self.sorting_label_comboBox.clear()
        self.file_name_lineEdit.setText(filename)

    def openFolder(self):
        """Open file manager and load selected folder. """
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open folder", self.default_root_folder)
        # filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "./",
        #                                                            ";;".join(self.file_type_dict.values()))  # start path
        if folder_path == "":
            return

        if isinstance(self.current_data_object, SpikeSorterData):
            if folder_path == self.current_data_object.path:
                return

        self.current_data_object = SpikeSorterData(folder_path, 'openephys')
        self.current_raw_object = None
        self.current_filted_object = None
        self.current_spike_object = None
        self.current_event_object = None
        self.default_root_folder = os.path.split(
            self.current_data_object.path)[0]
        self.signal_data_file_name_changed.emit(self.current_data_object)

        self.setDataModel()
        self.undo_stack_dict.clear()

        # Clear all undo stack
        for undo_stack in self.main_window.undo_group.stacks():
            self.main_window.undo_group.removeStack(undo_stack)

        # self.sorting_label_comboBox.clear()
        self.file_name_lineEdit.setText(folder_path)

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

        new_row_items = self.createRowItems(new_spike_object.header)
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

        self.updataTreeView(row_items, new_spike_object)
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
            default_setting=self.default_spike_setting,
            parent=self)

        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        self.default_spike_setting = dialog.setting

        if dialog.setting['Reference'][0]:
            ref = dialog.setting['Reference'][1]
        else:
            ref = -1
        filter_values = dialog.setting['Filter']

        new_filted_object = self.preprocessing(ref, *filter_values)

        if dialog.setting['ThresholdType'] == 'MAD':
            threshold = new_filted_object.estimated_sd * \
                dialog.setting['MADFactor']
        elif dialog.setting['ThresholdType'] == 'Const':
            threshold = dialog.setting['ConstThreshold']

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

    def preprocessing(self, ref: int,
                      low: int | float,
                      high: int | float) -> ContinuousData:
        new_filted_object = self.current_data_object.subtractReference(channel_ID=self.current_raw_object.channel_ID,
                                                                       reference_ID=ref)
        new_filted_object = new_filted_object.bandpassFilter(low, high)
        return new_filted_object

    def extractWaveforms(self):
        row_items = self.getSelectedRowItems()
        meta_data = [item.text() for item in row_items]
        meta_data = dict(zip(self.header_name, meta_data))
        row_type = meta_data['Type']
        if self.current_filted_object is None:
            # use default extract wavform setting
            setting = self.default_spike_setting

            if setting['Reference'][0]:
                ref = setting['Reference'][1]
            else:
                ref = -1
            filter_values = setting['Filter']

            new_filted_object = self.preprocessing(ref, *filter_values)

            if setting['ThresholdType'] == 'MAD':
                threshold = new_filted_object.estimated_sd * \
                    setting['MADFactor']
            elif setting['ThresholdType'] == 'Const':
                threshold = setting['ConstThreshold']

            new_filted_object = new_filted_object.createCopy(
                threshold=threshold)
            new_spike_object = None

            self.current_filted_object = new_filted_object

        new_spike_object = self.current_filted_object.extractWaveforms(
            self.current_filted_object.threshold)

        if row_type == 'Spikes':
            label = meta_data['Label']
            label = label[:-1] if label.endswith('*') else label
            new_spike_object.setLabel(label)

            # compute old for undo
            old_filted_object = self.current_data_object.subtractReference(
                channel_ID=self.current_spike_object.channel_ID,
                reference_ID=self.current_spike_object.reference)
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

                new_row_items = self.createRowItems(new_spike_object.header)
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

                new_row_items = self.createRowItems(new_spike_object.header)
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

            self.updataTreeView(row_items, new_spike_object)
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

    # def setLabelCombox(self, labels: list | None = None, current: str | None = None):
    #     self.sorting_label_comboBox.clear()
    #     if labels is None:
    #         return
    #     self.sorting_label_comboBox.addItems(labels)
    #     if current is None:
    #         return
    #     self.sorting_label_comboBox.setCurrentText(current)

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
            self.updataTreeView(row_items, new_spike_object)
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
        channel_ID = int(self.dropSuffix(ID_item.text()))
        raw_object = self.current_data_object.getRaw(channel_ID)
        channel_row = ID_item.row()

        meta_data = self.rowItemsToMetadata(row_items)
        if meta_data['Type'] == 'Ref':
            self.current_data_object.saveReference(channel_ID)
            if raw_object == 'Removed':
                self.raws_group_item.removeRow(channel_row)
            else:
                # self.current_data_object.saveReference(channel_ID)
                # self.updataTreeView(row_items, raw_object)
                self.setUnsavedChangeIndicator(row_items, raw_object)
            return

        selecting_label = self.dropSuffix(row_items[1].text())
        new_selecting_item = None  # To locate the selecting row

        old_labels = raw_object.spikes
        self.current_data_object.saveChannel(channel_ID)
        new_labels = sorted(raw_object.spikes)

        # Modify row items
        self.spike_group_item.removeRow(channel_row)
        if len(new_labels) > 0:
            # has spike, insert new row items
            spike_object = raw_object.getSpike(new_labels[0])
            first_row_items = self.createRowItems(spike_object.header)

            new_ID_item = first_row_items[0]
            for label in new_labels[1:]:
                spike_object = raw_object.getSpike(label)
                new_row_items = self.createRowItems(spike_object.header)
                new_row_items[0] = QStandardItem('')
                if label == selecting_label:
                    new_selecting_item = new_row_items[1]
                new_ID_item.appendRow(new_row_items)
            self.spike_group_item.insertRow(channel_row, first_row_items)

            if new_selecting_item is None:
                new_selecting_item = first_row_items[1]

        # Undo stack
        deleted_labels = list(set(old_labels).difference(new_labels))
        for label in deleted_labels:
            undo_stack = self.undo_stack_dict.pop((channel_ID, label), None)
            if not undo_stack is None:
                self.main_window.undo_group.removeStack(undo_stack)

        # Recovery selection
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

    def createReference(self):
        if self.current_data_object is None:
            logger.warning('Not load data yet.')

            return
        header = self.current_data_object.raws_header
        dialog = CreateReferenceDialog(header, parent=self)
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        new_raw_object = self.current_data_object.createMedianReference(channel_ID_list=dialog.select_channel_IDs,
                                                                        new_channel_name=dialog.channel_name_lineEdit.text(),
                                                                        new_comment=dialog.comment_lineEdit.text())
        new_row_items = self.createRowItems(new_raw_object.header)
        self.raws_group_item.appendRow(new_row_items)
        self.setUnsavedChangeIndicator(new_row_items, new_raw_object)

        selection_model = self.treeView.selectionModel()
        selection_model.select(
            new_row_items[0].index(), QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect)
        self.treeView.scrollTo(new_row_items[0].index())

    def deleteReference(self):
        row_items = self.getSelectedRowItems()
        meta_data = self.rowItemsToMetadata(row_items)
        row_type = meta_data['Type']

        if row_type != 'Ref':
            logger.warning('Can only delete reference channel!')
            return

        channel_ID = self.current_raw_object.channel_ID
        self.current_data_object.removeReference(channel_ID)
        logger.info(f'Delete reference channel {channel_ID}.')

        ID_item = row_items[0]
        if not ID_item.text().endswith('*'):
            ID_item.setText(ID_item.text() + '*')
            font = ID_item.font()
            font.setBold(True)
            ID_item.setFont(font)

        for item in row_items[1:]:
            font = ID_item.font()
            font.setBold(False)
            item.setFont(font)
            item.setForeground(QColor('darkGray'))

        self.current_raw_object = None
        self.current_filted_object = None
        self.current_spike_object = None
        reset_selection = True
        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object,
                                            reset_selection)

    def export(self):
        self.file_type_dict = {
            # "openephy": "Open Ephys Format (*.continuous)",
            "pyephys": "pyephys format (*.h5)"
        }  # File types to load
        default_filename = os.path.splitext(
            self.current_data_object.path)[0] + '.h5'
        new_filename, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Export file", default_filename,
                                                                       ";;".join(self.file_type_dict.values()))  # start path
        if new_filename == "":
            return

        self.current_data_object.export(new_filename, 'pyephys')
        self.openFile(new_filename)

        # self.current_data_object = SpikeSorterData(new_filename, 'pyephys')
        # self.signal_data_file_name_changed.emit(self.current_data_object)

        # self.setDataModel()
        # self.undo_stack_dict.clear()

        # # Clear all undo stack
        # for undo_stack in self.main_window.undo_group.stacks():
        #     self.main_window.undo_group.removeStack(undo_stack)

        # self.sorting_label_comboBox.clear()
        # self.file_name_lineEdit.setText(new_filename)

    def selectEvents(self):
        if self.current_data_object is None:
            logger.warning('Not load data yet.')
            return

        if self.current_event_object is None:
            # Events
            event_IDs = self.current_data_object.event_IDs
            new_event_object = None
            if event_IDs != []:
                new_event_object = self.current_data_object.getEvent(
                    event_IDs[0])
            if new_event_object is None:
                logger.warning('No events data.')
                return
            new_event_object._loadData()
            self.current_event_object = new_event_object
            self.current_showing_events = np.unique(
                self.current_event_object.unit_IDs).tolist()

        unit_header = self.current_event_object.unit_header
        dialog = SelectEventsDialog(
            unit_header, self.current_showing_events, parent=self)
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        self.current_showing_events = dialog.select_event_IDs
        self.signal_showing_events_changed.emit(self.current_showing_events)

    def setBackgroundChannel(self):
        if self.current_data_object is None:
            mbox = QMessageBox(self)
            mbox.warning(self, 'Warning',
                         'Not load data yet.')
            return
        all_chan_ID = self.current_data_object.channel_IDs

        dialog = SetBackgroundChannelDialog(all_channel_IDs=all_chan_ID,
                                            default_setting=self.default_background_channel_setting,
                                            parent=self)
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        self.default_background_channel_setting = dialog.setting

        new_bg_object = None

        if not dialog.setting['Show']:
            self.signal_background_continuous_data_changed.emit(new_bg_object,
                                                                dialog.setting['Color'],
                                                                dialog.setting['ShowOnTop'])
            return

        new_bg_object = self.current_data_object.getRaw(
            dialog.setting['BackgroundChannel'], load_data=True)

        if dialog.setting['Reference'][0]:
            new_bg_object = self.current_data_object.subtractReference(
                channel_ID=dialog.setting['BackgroundChannel'],
                reference_ID=dialog.setting['Reference'][1])
        else:
            new_bg_object = self.current_data_object.getRaw(
                dialog.setting['BackgroundChannel'], load_data=True)

        if dialog.setting['Filter'][0]:
            new_bg_object = new_bg_object.bandpassFilter(low=dialog.setting['Filter'][1],
                                                         high=dialog.setting['Filter'][2])

        self.signal_background_continuous_data_changed.emit(new_bg_object,
                                                            dialog.setting['Color'],
                                                            dialog.setting['ShowOnTop'])

    def setUnsavedChangeIndicator(self, row_items: list, obj: ContinuousData | DiscreteData):
        if obj is None:
            return
        ID_item = row_items[0]
        label_item = row_items[1]
        meta_items = row_items[2:]

        if isinstance(obj, ContinuousData):
            if obj._from_file:
                if ID_item.text().endswith('*'):
                    ID_item.setText(ID_item.text()[:-1])
                for item in row_items:
                    font = item.font()
                    font.setBold(False)
                    item.setFont(font)
            else:
                if not ID_item.text().endswith('*'):
                    ID_item.setText(ID_item.text() + '*')
                for item in row_items:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
            return

        if obj._from_file:
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
        channel_ID = obj.channel_ID
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

    def updataTreeView(self, row_items, object_has_header: ContinuousData | DiscreteData):
        for index, key in enumerate(self.header_name):
            if key == 'Reference':
                key = 'ReferenceID'
            row_items[index].setText(
                str(object_has_header.header.get(key, '')))


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
    def __init__(self, filted_object: ContinuousData | None, all_chan_ID: list,
                 default_setting=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Extract Waveform Settings Dialog")
        # self.default_spike_setting = default_setting
        self.const_thr = 0
        self.mad_factor = -3

        if filted_object is None:
            self.setting = default_setting
        else:
            if filted_object.reference == -1:
                ref = (False, default_setting['Reference'][1])
            else:
                ref = (True, filted_object.reference)

            self.const_thr = filted_object.threshold
            try:
                self.mad_factor = self.const_thr / filted_object.estimated_sd
            except ZeroDivisionError:
                self.mad_factor = -3

            self.setting = {
                'Reference': ref,
                'Filter': (filted_object.low_cutoff, filted_object.high_cutoff),
                'ThresholdType': default_setting['ThresholdType'],
                'MADFactor': self.mad_factor,
                'ConstThreshold': self.const_thr
            }
        self.channel_ref_comboBox.clear()
        self.channel_ref_comboBox.addItems(map(str, all_chan_ID))
        self.initSetting()

    def initSetting(self):
        # if len(self.setting['Reference']) == 1:
        self.ref_groupBox.setChecked(self.setting['Reference'][0])
        self.channel_ref_comboBox.setCurrentText(
            str(self.setting['Reference'][1]))

        self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][0])
        self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][1])

        if self.setting['ThresholdType'] == 'MAD':
            self.mad_thr_radioButton.setChecked(True)
        elif self.setting['ThresholdType'] == 'Const':
            self.const_thr_radioButton.setChecked(True)

        self.mad_thr_doubleSpinBox.setValue(self.setting['MADFactor'])
        self.const_thr_doubleSpinBox.setValue(self.setting['ConstThreshold'])

    def accept(self):
        self.setting['Reference'] = (self.ref_groupBox.isChecked(),
                                     int(self.channel_ref_comboBox.currentText()))

        self.setting['Filter'] = (self.filter_low_doubleSpinBox.value(),
                                  self.filter_high_doubleSpinBox.value())

        if self.mad_thr_radioButton.isChecked():
            self.setting['ThresholdType'] = 'MAD'
        elif self.const_thr_radioButton.isChecked():
            self.setting['ThresholdType'] = 'Const'

        self.setting['MADFactor'] = self.mad_thr_doubleSpinBox.value()
        self.setting['ConstThreshold'] = self.const_thr_doubleSpinBox.value()

        super().accept()


class CreateReferenceDialog(Ui_CreateReferenceDialog, QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Create Reference Channel Dialog")
        self.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)

        self.select_channel_IDs: list = []
        self.channel_checkbox_list = []
        self.colnames = ['ID', 'Name', 'Type']
        self.methods = ['Median']

        self.data = data

        self.user_changed_channel_name = False
        self.user_changed_comment = False
        self.channel_name_lineEdit.textEdited.connect(
            self.setCustomChannelName)
        self.comment_lineEdit.textEdited.connect(self.setCustomComment)

        self.method_comboBox.clear()
        self.method_comboBox.addItems(self.methods)
        self.method_comboBox.setCurrentText(self.methods[0])

        self.select_all_checkBox.stateChanged.connect(
            self.allCheckboxStateChanged)
        self.initDataModel()
        self.setDataModel()

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels([''] + self.colnames)
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.tableView.verticalHeader().setVisible(False)  # hide index

        selection_model = self.tableView.selectionModel()
        selection_model.selectionChanged.connect(self.onSelectionChanged)

    def setDataModel(self):
        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels([''] + self.colnames)

        # self.addAllChannelRow(row=0)
        for row, record in enumerate(self.data.to_records()):
            self.appendChannelRow(record, row=row)

        self.tableView.resizeColumnToContents(0)

    def appendChannelRow(self, header_records, row):
        model = self.tableView.model()
        channel_ID = header_records['ID']

        # 創建一個 CheckBox Widget
        checkbox = QCheckBox()
        checkbox.setProperty("ID", int(channel_ID))
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.checkboxStateChanged)
        self.channel_checkbox_list.append(checkbox)

        # 將 CheckBox Widget 放入自定義的 Widget 容器中
        checkbox_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(checkbox)
        checkbox_widget.setLayout(layout)

        # 將自定義的 Widget 設定為表格的單元格
        model.appendRow([QStandardItem()] + [
                        QStandardItem(str(header_records[col])) for col in self.colnames])
        self.tableView.setIndexWidget(
            model.index(row, 0), checkbox_widget)

    def allCheckboxStateChanged(self, state):
        if state == Qt.Checked:
            # logger.debug('Checked')
            [channel_checkbox.setChecked(True)
             for channel_checkbox in self.channel_checkbox_list]
            # self.locked_rows_list.append(channel_ID)
        elif state == Qt.Unchecked:
            # logger.debug('Unchecked')
            [channel_checkbox.setChecked(False)
             for channel_checkbox in self.channel_checkbox_list]
            # self.locked_rows_list.remove(channel_ID)
        elif state == Qt.PartiallyChecked:
            # logger.debug('PartiallyChecked')
            self.select_all_checkBox.setCheckState(Qt.Checked)

    def checkboxStateChanged(self, state):
        checkbox = self.sender()
        channel_ID = checkbox.property("ID")
        if state == Qt.Checked:
            self.select_channel_IDs.append(channel_ID)
        elif state == Qt.Unchecked:
            self.select_channel_IDs.remove(channel_ID)

        all_checkbox_state = [channel_checkbox.isChecked()
                              for channel_checkbox in self.channel_checkbox_list]

        # set all checkbox state
        self.select_all_checkBox.blockSignals(True)

        if np.all(all_checkbox_state):
            self.select_all_checkBox.setCheckState(Qt.Checked)
        elif np.any(all_checkbox_state):
            self.select_all_checkBox.setCheckState(Qt.PartiallyChecked)
        else:
            self.select_all_checkBox.setCheckState(Qt.Unchecked)

        self.select_all_checkBox.blockSignals(False)
        self.setChannelNameAndComment()

    def onSelectionChanged(self, selected, deselected):
        # model = self.tableView.model()
        # selection_model = self.tableView.selectionModel()
        # selected_indexes = selection_model.selectedIndexes()

        checkbox = self.tableView.indexWidget(
            selected.indexes()[0]).layout().itemAt(0).widget()

        current_state = checkbox.checkState()
        if current_state == Qt.Unchecked:
            checkbox.setCheckState(Qt.Checked)
        elif current_state == Qt.Checked:
            checkbox.setCheckState(Qt.Unchecked)

        # logger.debug(item.layout().itemAt(0).widget())
        # logger.debug(item.child(0))

        # selection_model = self.tableView.selectionModel()
        # selected_indexes = selection_model.selectedRows()
        # self.selected_rows_list = [index.row() for index in selected_indexes]
        # logger.debug('onSelectionChanged')
        # self.sendShowingUnits()

    def setCustomChannelName(self, channel_name: str):
        self.user_changed_channel_name = True
        if self.channel_name_lineEdit.text() == '':
            self.user_changed_channel_name = False

    def setCustomComment(self, comment: str):
        self.user_changed_comment = True

    def setChannelNameAndComment(self):
        if self.select_channel_IDs == []:
            if not self.user_changed_channel_name:
                self.channel_name_lineEdit.setText('')

            if not self.user_changed_comment:
                self.comment_lineEdit.setText('')

            return

        method: str = self.method_comboBox.currentText()
        if self.select_all_checkBox.checkState() == Qt.Checked:
            channels = 'all'
        else:
            index = self.data['ID'].isin(list(set(self.select_channel_IDs)))
            channel_name_list: list = self.data['Name'][index].to_list()
            channels = ",".join(channel_name_list)

        if not self.user_changed_channel_name:
            self.channel_name_lineEdit.setText(
                f'{method}Ref({channels})')

        if not self.user_changed_comment:
            self.comment_lineEdit.setText(f'This channel is a {method.lower()} reference channel made from ' +
                                          f'{channels}')

    def accept(self):
        if len(self.select_channel_IDs) < 2:
            mbox = QMessageBox(self)
            mbox.information(
                self, 'Warning', 'Must select at least two channels.')
            return
        elif self.channel_name_lineEdit.text() == '':
            mbox = QMessageBox(self)
            mbox.information(self, 'Warning', 'Channel name can not be empty.')
            return
        super().accept()


class SelectEventsDialog(Ui_SelectEventsDialog, QDialog):
    def __init__(self, data, selected: list = [], parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Select Events Dialog")
        self.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)

        self._init_state = selected
        self.select_event_IDs: list = []
        self.unit_checkbox_list: list = []
        self.colnames = ['ID', 'Name', 'NumRecords']
        self.color_palette_list = sns.color_palette(
            'bright', 64)  # palette for events and spikes

        self.data = data

        self.select_all_checkBox.stateChanged.connect(
            self.allCheckboxStateChanged)
        self.initDataModel()
        self.setDataModel()

        delegate = EventsColorDelegate(self.data, self.color_palette_list)
        self.tableView.setItemDelegate(delegate)

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels([''] + self.colnames)
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.tableView.verticalHeader().setVisible(False)  # hide index

        selection_model = self.tableView.selectionModel()
        selection_model.selectionChanged.connect(self.onSelectionChanged)

    def setDataModel(self):
        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels([''] + self.colnames)

        for row, record in enumerate(self.data.to_records()):
            self.appendChannelRow(record, row=row)

        self.tableView.resizeColumnToContents(0)

    def appendChannelRow(self, header_records, row):
        model = self.tableView.model()
        event_ID = header_records['ID']

        # 創建一個 CheckBox Widget
        checkbox = QCheckBox()
        checkbox.setProperty("ID", int(event_ID))
        checkbox.stateChanged.connect(self.checkboxStateChanged)
        checkbox.setChecked(int(event_ID) in self._init_state)
        self.unit_checkbox_list.append(checkbox)

        # 將 CheckBox Widget 放入自定義的 Widget 容器中
        checkbox_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(checkbox)
        checkbox_widget.setLayout(layout)

        # 將自定義的 Widget 設定為表格的單元格
        model.appendRow([QStandardItem()] + [
                        QStandardItem(str(header_records[col])) for col in self.colnames])
        self.tableView.setIndexWidget(
            model.index(row, 0), checkbox_widget)

    def allCheckboxStateChanged(self, state):
        if state == Qt.Checked:
            # logger.debug('Checked')
            [channel_checkbox.setChecked(True)
             for channel_checkbox in self.unit_checkbox_list]
            # self.locked_rows_list.append(channel_ID)
        elif state == Qt.Unchecked:
            # logger.debug('Unchecked')
            [channel_checkbox.setChecked(False)
             for channel_checkbox in self.unit_checkbox_list]
            # self.locked_rows_list.remove(channel_ID)
        elif state == Qt.PartiallyChecked:
            # logger.debug('PartiallyChecked')
            self.select_all_checkBox.setCheckState(Qt.Checked)

    def checkboxStateChanged(self, state):
        checkbox = self.sender()
        channel_ID = checkbox.property("ID")
        if state == Qt.Checked:
            self.select_event_IDs.append(channel_ID)
        elif state == Qt.Unchecked:
            self.select_event_IDs.remove(channel_ID)

        all_checkbox_state = [channel_checkbox.isChecked()
                              for channel_checkbox in self.unit_checkbox_list]

        # set all checkbox state
        self.select_all_checkBox.blockSignals(True)

        if np.all(all_checkbox_state):
            self.select_all_checkBox.setCheckState(Qt.Checked)
        elif np.any(all_checkbox_state):
            self.select_all_checkBox.setCheckState(Qt.PartiallyChecked)
        else:
            self.select_all_checkBox.setCheckState(Qt.Unchecked)

        self.select_all_checkBox.blockSignals(False)

    def onSelectionChanged(self, selected, deselected):
        checkbox = self.tableView.indexWidget(
            selected.indexes()[0]).layout().itemAt(0).widget()

        current_state = checkbox.checkState()
        if current_state == Qt.Unchecked:
            checkbox.setCheckState(Qt.Checked)
        elif current_state == Qt.Checked:
            checkbox.setCheckState(Qt.Unchecked)

    def accept(self):
        # if len(self.select_event_IDs) < 2:
        #     mbox = QMessageBox(self)
        #     mbox.information(
        #         self, 'Warning', 'Must select at least two channels.')
        #     return
        super().accept()


class EventsColorDelegate(QStyledItemDelegate):
    def __init__(self, header, color_palette):
        super().__init__()
        self.color_palette_list = color_palette
        self.unit_color_map = dict(zip(header['ID'], np.arange(
            header.shape[0], dtype=int)))

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        color = self.color_palette_list[index.row()]
        color = (np.array(color) * 255).astype(int)

        option.backgroundBrush = QColor(*color)


class SetBackgroundChannelDialog(Ui_SetBackgroundChannelDialog, QDialog):
    def __init__(self, all_channel_IDs, default_setting, parent=None):
        """_summary_

        Args:
            all_channel_IDs (_type_): _description_
            default_setting (_type_): {
                'BackgroundChannel': 'No select',
                'Color': None,
                'Reference': (False, 0),
                'Filter': (False, 250, 6000),
                }
            parent (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Set BackgroundChannel Dialog")
        # self.setMinimumWidth(500)
        # self.setMinimumHeight(500)
        self.setting = default_setting
        self.setColor()

        self.bg_channel_checkBox.setChecked(self.setting['Show'])
        self.bg_channel_comboBox.setEnabled(self.setting['Show'])
        self.bg_channel_comboBox.addItems(map(str, all_channel_IDs))
        self.bg_channel_comboBox.setCurrentText(
            str(self.setting['BackgroundChannel']))

        self.show_on_top_checkBox.setChecked(self.setting['ShowOnTop'])

        self.ref_groupBox.setChecked(self.setting['Reference'][0])
        self.select_reference_comboBox.addItems(map(str, all_channel_IDs))

        # if self.ref_groupBox.isChecked():
        #     if self.setting['Reference'][1] == 'raw':
        #         self.raw_reference_radioButton.setChecked(True)
        #         self.raw_reference_comboBox.setCurrentText(
        #             str(self.setting['Reference'][2]))

        #     elif self.setting['Reference'][1] == 'ref':
        #         self.raw_reference_radioButton.setChecked(True)
        #         self.raw_reference_comboBox.setCurrentText(
        #             str(self.setting['Reference'][2]))

        self.filter_groupBox.setChecked(self.setting['Filter'][0])
        self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][1])
        self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][2])

        self.color_pushButton.clicked.connect(self.selectColor)

    def setColor(self):
        if self.setting['Color'] is None:
            self.setting['Color'] = QColor(0, 255, 255)
        self.color_pushButton.setStyleSheet(
            f"background-color: {self.setting['Color'].name()}")

    def selectColor(self):
        if self.setting['Color'] is None:
            color = QColorDialog.getColor(parent=self)
        else:
            color = QColorDialog.getColor(
                initial=self.setting['Color'], parent=self)

        if color.isValid():
            self.setting['Color'] = color
            self.setColor()

    def accept(self):
        bg_channel = self.bg_channel_comboBox.currentText()
        try:
            bg_channel = int(bg_channel)
        except TypeError:
            pass

        # if self.raw_reference_radioButton.isChecked():
        #     ref = 'raw'
        #     ref_channel = self.raw_reference_comboBox.currentText()
        # elif self.ref_reference_radioButton.isChecked():
        #     ref = 'ref'
        ref_channel = self.select_reference_comboBox.currentText()
        try:
            ref_channel = int(ref_channel)
        except TypeError:
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

        super().accept()
