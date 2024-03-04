from UI.ChannelDetailv2_ui import Ui_ChannelDetail
from UI.ExtractWaveformSettings_ui import Ui_ExtractWaveformSettings

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QWidget, QAbstractItemView, QDialog, QUndoStack, QUndoCommand)

from DataStructure.datav3 import SpikeSorterData, ContinuousData, DiscreteData
import numpy as np

import logging
logger = logging.getLogger(__name__)


class ChannelDetail(QtWidgets.QWidget, Ui_ChannelDetail):
    signal_data_file_name_changed = QtCore.pyqtSignal()
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
        self.signal_data_file_name_changed.emit()

        self.setDataModel()
        self.undo_stack_dict = dict()
        self.sorting_label_comboBox.clear()
        self.file_name_lineEdit.setText(filename)

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

            group_item = QStandardItem('Spikes')  # Top level, Group
            model.appendRow(group_item)

            for chan_ID in spikes_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = spikes_header[spikes_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                group_item.appendRow(first_items)

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
            ref = [int(dialog.channel_ref_comboBox.currentText())]

        elif dialog.median_ref_radioButton.isChecked():
            ref = [int(i)
                   for i in dialog.show_channels_lineEdit.text().split(', ')]

        low = dialog.filter_low_doubleSpinBox.value()
        high = dialog.filter_high_doubleSpinBox.value()

        new_filted_object = self.filtedData(ref, low, high)
        if dialog.const_thr_radioButton.isChecked():
            threshold = dialog.const_thr_doubleSpinBox.value()
        elif dialog.mad_thr_radioButton.isChecked():
            threshold = new_filted_object.estimated_sd * \
                dialog.mad_thr_doubleSpinBox.value()

        new_filted_object = new_filted_object.createCopy(threshold=threshold)
        new_spike_object = None

        command = ChangeFilterCommand("Change filter",
                                      self,
                                      self.current_raw_object,
                                      self.current_filted_object,
                                      new_filted_object,
                                      self.current_spike_object,
                                      new_spike_object)

        self.current_filted_object = new_filted_object
        self.current_spike_object = new_spike_object
        self.current_undo_stack.push(command)

        # self.current_filted_object = self.current_filted_object.createCopy(
        #     threshold=threshold)
        # self.current_spike_object = None
        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object, True)

    def filtedData(self, ref: list = [],
                   low: int | float = None,
                   high: int | float = None) -> ContinuousData:
        new_filted_object = self.current_raw_object.createCopy()
        if len(ref) > 0:
            new_filted_object = self.current_data_object.subtractReference(channel=self.current_raw_object.channel_ID,
                                                                           reference=ref)
        if not low is None and not high is None:
            new_filted_object = new_filted_object.bandpassFilter(
                low, high)
        return new_filted_object

    def extractWaveforms(self):
        new_spike_object = self.current_filted_object.extractWaveforms(
            self.current_filted_object.threshold)

        command = ChangeSpikeCommand("Extract waveform",
                                     self,
                                     self.current_raw_object,
                                     self.current_filted_object,
                                     self.current_spike_object,
                                     new_spike_object)
        self.current_spike_object = new_spike_object
        self.current_undo_stack.push(command)

        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object, True)
        # logger.debug(type(self.current_filted_object))

        # logger.debug(self.current_spike_object)

    def sortChannel(self):
        new_spike_object = self.current_spike_object.autosort()
        # self.signal_spike_data_changed.emit(self.current_spike_object, True)

        command = ChangeSpikeCommand("Autosort",
                                     self,
                                     self.current_raw_object,
                                     self.current_filted_object,
                                     self.current_spike_object,
                                     new_spike_object)
        self.current_spike_object = new_spike_object
        self.current_undo_stack.push(command)

        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)
        self.signal_spike_data_changed.emit(self.current_spike_object, True)

    def onSelectionChanged(self, selected, deselected):
        model = self.treeView.model()
        indexes = selected.indexes()
        items = [model.itemFromIndex(ind) for ind in indexes]

        if items[0].parent() == None:  # Group
            self.sorting_label_comboBox.clear()
            return
        elif items[0].parent().parent() != None:  # Label
            items[0] = items[0].parent()

        meta_data = [item.text() for item in items]
        meta_data = dict(zip(self.header_name, meta_data))

        self.current_raw_object = None
        self.current_filted_object = None
        self.current_spike_object = None
        chan_ID = int(meta_data["ID"])
        label = meta_data['Label']
        logger.info(f'Selected type: {meta_data["Type"]}')

        if meta_data['Type'] == 'Spikes':
            # raw
            self.current_data_object.loadRaw(channel=chan_ID)
            self.current_raw_object = self.current_data_object.getRaw(chan_ID)
            # spike
            self.current_data_object.loadSpike(
                channel=chan_ID, label=label)
            self.current_spike_object = self.current_data_object.getSpike(
                channel=chan_ID, label=label)
            logger.debug(f'spike object { self.current_spike_object}')
            # filted
            self.current_filted_object = self.current_data_object.subtractReference(
                channel=chan_ID, reference=[self.current_spike_object.reference])
            self.current_filted_object = self.current_filted_object.bandpassFilter(low=self.current_spike_object.low_cutoff,
                                                                                   high=self.current_spike_object.high_cutoff)
            self.current_filted_object = self.current_filted_object.createCopy(
                threshold=self.current_spike_object.threshold)
            # self.setSpikeSetting()
            self.setLabelCombox(labels=self.current_raw_object.spikes,
                                current=self.current_spike_object.label)

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
        if new_spike_object is self.current_spike_object and new_filted_object is self.current_filted_object:
            return

        self.current_raw_object = new_raw_object
        self.current_filted_object = new_filted_object
        self.current_spike_object = new_spike_object

        # if action_type == 'ManualUnit':
        # chan_ID = self.current_spike_object.channel_ID
        # label = self.current_spike_object.label
        # filted
        # self.current_filted_object = self.current_data_object.subtractReference(
        #     channel=chan_ID, reference=[self.current_spike_object.reference])
        # self.current_filted_object = self.current_filted_object.bandpassFilter(low=self.current_spike_object.low_cutoff,
        #                                                                        high=self.current_spike_object.high_cutoff)
        # self.current_filted_object = self.current_filted_object.createCopy(
        #     threshold=self.current_spike_object.threshold)
        # self.setSpikeSetting()
        # self.setLabelCombox(labels=self.current_raw_object.spikes,
        #                     current=label)

        self.signal_continuous_data_changed.emit(self.current_raw_object,
                                                 self.current_filted_object)

        self.signal_spike_data_changed.emit(
            self.current_spike_object, action_type in ['Change filter', 'Extract waveform'])

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        if new_spike_object is self.current_spike_object:
            return
        command = ChangeSpikeCommand("Manual unit",
                                     self,
                                     self.current_raw_object,
                                     self.current_filted_object,
                                     self.current_spike_object,
                                     new_spike_object)
        self.current_spike_object = new_spike_object
        self.current_undo_stack.push(command)


class ChangeFilterCommand(QUndoCommand):
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
        self.widget.handleUndoRedo(
            self.action_type, self.raw_object, self.new_filted_object, self.new_spike_object)
        # self.raw_object.setSpike(
        #     self.new_spike_object, self.new_spike_object.label)
        # logger.info(
        #     f"Redo: {self.text()}, Data: {self.new_spike_object}")

    def undo(self):
        # 撤销操作，回滚应用程序状态
        self.widget.handleUndoRedo(
            self.action_type, self.raw_object, self.old_filted_object, self.old_spike_object)
        # self.raw_object.setSpike(
        #     self.old_spike_object, self.old_spike_object.label)
        # logger.info(
        #     f"Undo: {self.text()}, Data: {self.old_spike_object}")


class ChangeSpikeCommand(QUndoCommand):
    def __init__(self, text, widget, raw_object: ContinuousData, filted_object: ContinuousData, old_spike_object: DiscreteData, new_spike_object: DiscreteData):
        super().__init__(text)
        self.widget = widget
        self.action_type = text
        self.raw_object = raw_object
        self.filted_object = filted_object
        self.old_spike_object = old_spike_object
        self.new_spike_object = new_spike_object

    def redo(self):
        # 在这里执行操作，修改应用程序状态
        self.widget.handleUndoRedo(
            self.action_type, self.raw_object, self.filted_object, self.new_spike_object)
        self.raw_object.setSpike(
            self.new_spike_object, self.new_spike_object.label)
        logger.info(
            f"Redo: {self.text()} {self.new_spike_object}")

    def undo(self):
        # 撤销操作，回滚应用程序状态
        self.widget.handleUndoRedo(
            self.action_type, self.raw_object, self.filted_object, self.old_spike_object)
        self.raw_object.setSpike(
            self.old_spike_object, self.old_spike_object.label)
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
        if len(self.setting['Reference']) == 1:
            self.channel_ref_radioButton.setChecked(True)
            self.channel_ref_comboBox.setCurrentText(
                str(self.setting['Reference'][0]))
        elif len(self.setting['Reference']) > 1:
            logger.critical('Use median of channels: Not implemented error')

        self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][0])
        self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][1])

        if self.setting['Threshold'][0] == 'MAD':
            self.mad_thr_radioButton.setChecked(True)
            self.mad_thr_doubleSpinBox.setValue(self.setting['Threshold'][1])

        elif self.setting['Threshold'][0] == 'Const':
            self.const_thr_radioButton.setChecked(True)
            self.const_thr_doubleSpinBox.setValue(self.setting['Threshold'][1])
