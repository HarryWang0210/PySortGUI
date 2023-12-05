from UI.ChannelDetailv2_ui import Ui_ChannelDetail
from UI.ExtractWaveformSettings_ui import Ui_ExtractWaveformSettings

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QAbstractItemView, QDialog

from DataStructure.datav2 import SpikeSorterData
import numpy as np

import logging
logger = logging.getLogger(__name__)


class ChannelDetail(QtWidgets.QWidget, Ui_ChannelDetail):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_filted_data_changed = QtCore.pyqtSignal(object)
    signal_extract_wav_changed = QtCore.pyqtSignal(object)
    signal_sorting_result_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Channel Detail"
        self.setupUi(self)
        self.data_object = None
        self.current_chan_info = None

        self.filted_data = None
        self.default_spike_setting = {
            'Reference': ('Single', [0]),
            'Filter': (250, 6000),
            'Threshold': ('MAD', -3)
        }
        self.current_spike_setting = {
            'Reference': ('Single', [0]),
            'Filter': (250, 6000),
            'Threshold': ('MAD', -3)
        }

        self.spikes = {
            "unitInfo": None,
            "unitID": None,
            "timestamps": None,
            "waveforms": None
        }

        self.header = ['ID', 'Label', 'Name', 'NumUnits',
                       'NumRecords', 'LowCutOff', 'HighCutOff', 'Reference', 'Threshold', 'Type']
        self.raws_header = None
        self.spikes_header = None
        self.events_header = None

        self.initDataModel()
        self.setupConnections()

    def setupConnections(self):
        self.open_file_toolButton.released.connect(self.openFile)
        self.extract_wav_setting_toolButton.released.connect(
            self.setExtractWaveformParams)

        self.extract_wav_pushButton.clicked.connect(self.extract_wav)
        self.sort_channel_pushButton.clicked.connect(self.sort_channel)

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

        if isinstance(self.data_object, SpikeSorterData):
            if filename == self.data_object.filename:
                return

        self.data_object = SpikeSorterData(filename)
        self.signal_data_file_name_changed.emit(self.data_object)
        self.raws_header = self.data_object.raws_header
        self.spikes_header = self.data_object.spikes_header
        self.events_header = self.data_object.events_header

        self.setDataModel()
        self.sorting_label_comboBox.clear()
        self.file_name_lineEdit.setText(filename)

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(self.header)
        self.treeView.setModel(model)

        # for col in range(model.columnCount()):
        #     self.treeView.resizeColumnToContents(col)

    def setDataModel(self):
        model = self.treeView.model()
        model.clear()
        model.setHorizontalHeaderLabels(self.header)

        if not isinstance(self.spikes_header, type(None)):
            spikes_header = self.spikes_header.copy()
            for nan_column in [x for x in self.header if x not in spikes_header.columns]:
                spikes_header[nan_column] = ''
            spikes_header['Reference'] = spikes_header['ReferenceID']
            spikes_header = spikes_header[self.header]

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

        if not isinstance(self.raws_header, type(None)):
            raws_header = self.raws_header.copy()
            for nan_column in [x for x in self.header if x not in raws_header.columns]:
                raws_header[nan_column] = ''
            raws_header = raws_header[self.header]

            group_item = QStandardItem('Raws')  # Top level, Group
            model.appendRow(group_item)

            for chan_ID in raws_header['ID'].unique():
                chan_ID_item = QStandardItem(str(chan_ID))
                sub_data = raws_header[raws_header['ID'] == chan_ID]

                first_items = [chan_ID_item] + \
                    [QStandardItem(str(col_value))
                     for col_value in sub_data.iloc[0, 1:]]
                group_item.appendRow(first_items)

        if not isinstance(self.events_header, type(None)):
            events_header = self.events_header.copy()
            for nan_column in [x for x in self.header if x not in events_header.columns]:
                events_header[nan_column] = ''
            events_header = events_header[self.header]

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
        if self.data_object == None:
            logger.warn('No SpikeSorterData.')
            return
        if isinstance(self.current_chan_info, type(None)):
            logger.warn('No selected channel.')
            return
        all_chan_ID = self.raws_header['ID'].unique().tolist()

        if self.current_chan_info['Type'] == 'Spikes':
            dialog = ExtractWaveformSettingsDialog(
                setting=self.current_spike_setting,
                all_chan_ID=all_chan_ID,
                parent=self)

        elif self.current_chan_info['Type'] == 'Raws':
            dialog = ExtractWaveformSettingsDialog(
                setting=self.default_spike_setting,
                all_chan_ID=all_chan_ID,
                parent=self)

        else:
            logger.warn('No support type.')
            return

        result = dialog.exec_()
        if result == QDialog.Accepted:
            if dialog.channel_ref_radioButton.isChecked():
                self.current_spike_setting['Reference'] = ('Single',
                                                           [int(dialog.channel_ref_comboBox.currentText())])
            elif dialog.median_ref_radioButton.isChecked():
                self.current_spike_setting['Reference'] = ('Median',
                                                           dialog.show_channels_lineEdit.text().split(', '))
            self.current_spike_setting['Filter'] = (dialog.filter_low_doubleSpinBox.value(),
                                                    dialog.filter_high_doubleSpinBox.value())

            if dialog.const_thr_radioButton.isChecked():
                self.current_spike_setting['Threshold'] = ('Const',
                                                           dialog.const_thr_doubleSpinBox.value())
            elif dialog.mad_thr_radioButton.isChecked():
                self.current_spike_setting['Threshold'] = ('MAD',
                                                           dialog.mad_thr_doubleSpinBox.value())

            self.current_chan_info['ReferenceID'] = self.current_spike_setting['Reference'][1][0]
            self.current_chan_info['LowCutOff'] = self.current_spike_setting['Filter'][0]
            self.current_chan_info['HighCutOff'] = self.current_spike_setting['Filter'][1]

            self.getFiltedData()
            self.current_chan_info['Threshold'] = self.current_spike_setting['Threshold'][1] * \
                self.data_object.estimatedSD(self.filted_data)

            self.signal_spike_chan_changed.emit(self.current_chan_info)
            self.signal_filted_data_changed.emit(self.filted_data)
        else:
            return

    def extract_wav(self):
        result, timestamps = self.data_object.test_extract_waveforms(self.filted_data,
                                                                     self.current_chan_info['ID'],
                                                                     self.current_chan_info['Threshold'])

        waveforms = result['Waveforms']

        self.spikes['timestamps'] = timestamps
        self.spikes['waveforms'] = waveforms

        self.signal_extract_wav_changed.emit({
            'timestamps': timestamps,
            'waveforms': waveforms
        })

    def sort_channel(self):
        unitID = self.data_object.test_auto_sort(self.current_chan_info['ID'],
                                                 waveforms=self.spikes['waveforms'],
                                                 timestamps=self.spikes['timestamps'])

        self.spikes['unitID'] = unitID

        self.signal_sorting_result_changed.emit(unitID)
    # def generateDataModel(self):
    #     model = QStandardItemModel()
    #     model.setHorizontalHeaderLabels(self.header)

    #     raws_header = self.data_object.raws_header
    #     spikes_header = self.data_object.spikes_header
    #     events_header = self.data_object.events_header

    #     if not isinstance(spikes_header, type(None)):
    #         for nan_column in [x for x in self.header if x not in spikes_header.columns]:
    #             spikes_header[nan_column] = ''
    #         spikes_header['Reference'] = spikes_header['ReferenceID']
    #         spikes_header = spikes_header[self.header]

    #         group_item = QStandardItem('Spikes')  # Top level, Group
    #         model.appendRow(group_item)

    #         for chan_ID in spikes_header['ID'].unique():
    #             chan_ID_item = QStandardItem(str(chan_ID))
    #             sub_data = spikes_header[spikes_header['ID'] == chan_ID]

    #             first_items = [chan_ID_item] + \
    #                 [QStandardItem(str(col_value))
    #                  for col_value in sub_data.iloc[0, 1:]]
    #             group_item.appendRow(first_items)

    #             for sub_row in sub_data.iloc[1:, 1:].values:
    #                 values = [QStandardItem("")] + \
    #                     [QStandardItem(str(col_value))
    #                      for col_value in sub_row]
    #                 chan_ID_item.appendRow(values)

    #     if not isinstance(raws_header, type(None)):
    #         for nan_column in [x for x in self.header if x not in raws_header.columns]:
    #             raws_header[nan_column] = ''
    #         raws_header = raws_header[self.header]

    #         group_item = QStandardItem('Raws')  # Top level, Group
    #         model.appendRow(group_item)

    #         for chan_ID in raws_header['ID'].unique():
    #             chan_ID_item = QStandardItem(str(chan_ID))
    #             sub_data = raws_header[raws_header['ID'] == chan_ID]

    #             first_items = [chan_ID_item] + \
    #                 [QStandardItem(str(col_value))
    #                  for col_value in sub_data.iloc[0, 1:]]
    #             group_item.appendRow(first_items)

    #     if not isinstance(events_header, type(None)):
    #         for nan_column in [x for x in self.header if x not in events_header.columns]:
    #             events_header[nan_column] = ''
    #         events_header = events_header[self.header]

    #         group_item = QStandardItem('Events')  # Top level, Group
    #         model.appendRow(group_item)

    #         for chan_ID in events_header['ID'].unique():
    #             chan_ID_item = QStandardItem(str(chan_ID))
    #             sub_data = events_header[events_header['ID'] == chan_ID]

    #             first_items = [chan_ID_item] + \
    #                 [QStandardItem(str(col_value))
    #                  for col_value in sub_data.iloc[0, 1:]]
    #             group_item.appendRow(first_items)

    #     self.treeView.setModel(model)

    #     for col in range(model.columnCount()):
    #         self.treeView.resizeColumnToContents(col)

    #     selection_model = self.treeView.selectionModel()
    #     selection_model.selectionChanged.connect(self.onSelectionChanged)

    # def initSpikeInfo(self):
    #     # init label
    #     self.sorting_label_comboBox.clear()
    #     # # init ref
    #     # if init_id:
    #     #     ID_list = df.index.get_level_values('ID').unique()
    #     #     self.ref_comboBox.clear()
    #     #     self.ref_comboBox.addItems(ID_list.map(str))
    #     # self.ref_checkBox.setChecked(False)
    #     # # init filter
    #     # self.filter_checkBox.setChecked(False)

    def onSelectionChanged(self, selected, deselected):
        model = self.treeView.model()
        indexes = selected.indexes()
        items = [model.itemFromIndex(ind) for ind in indexes]

        if items[0].parent() == None:  # Group
            self.sorting_label_comboBox.clear()
            self.current_chan_info = None
            return
        elif items[0].parent().parent() != None:  # Label
            items[0] = items[0].parent()

        meta_data = [item.text() for item in items]
        meta_data = dict(zip(self.header, meta_data))

        chan_ID = int(meta_data["ID"])
        if meta_data['Type'] == 'Spikes':
            spikes_header = self.spikes_header.copy()
            selected_df = spikes_header[(spikes_header['ID'] == chan_ID) & (
                spikes_header['Label'] == meta_data["Label"])].iloc[0, :]

        elif meta_data['Type'] == 'Raws':
            raws_header = self.raws_header.copy()
            selected_df = raws_header[raws_header['ID'] == chan_ID].iloc[0, :]

        elif meta_data['Type'] == 'Events':
            events_header = self.events_header.copy()
            selected_df = events_header[events_header['ID']
                                        == chan_ID].iloc[0, :]

        self.current_chan_info = selected_df.to_dict()
        if self.current_chan_info['Type'] == 'Spikes':
            self.getFiltedData()
            self.getSpikeSetting()
        else:
            self.filted_data = None

        self.updateLabel()

        # self.signal_spike_chan_changed.emit(meta_data)
        logger.info(f'Selected type: {self.current_chan_info["Type"]}')

        # # self.updateReference(meta_data)
        # # self.updateFilter(meta_data)
        self.signal_spike_chan_changed.emit(self.current_chan_info)
        self.signal_filted_data_changed.emit(self.filted_data)
    # def getGroup(self, df):
    #     df = df.copy()
    #     df["Group"] = df["NumUnits"].isnull()
    #     df["Group"] = df["Group"].map({True: "Raws", False: "Spikes"})
    #     df.set_index('Group', append=True, inplace=True)
    #     return df

    def updateLabel(self):
        self.sorting_label_comboBox.clear()
        if self.current_chan_info['Type'] == 'Spikes':
            spikes_header = self.spikes_header.copy()

            labels = spikes_header[spikes_header['ID']
                                   == self.current_chan_info["ID"]]['Label']
            self.sorting_label_comboBox.addItems(labels)
            self.sorting_label_comboBox.setCurrentText(
                self.current_chan_info["Label"])

    def getFiltedData(self):
        ref_data = self.data_object.getRaw(
            self.current_chan_info['ReferenceID'])
        self.filted_data = self.data_object.spikeFilter(chan_ID=self.current_chan_info['ID'],
                                                        ref=ref_data,
                                                        low=self.current_chan_info['LowCutOff'],
                                                        high=self.current_chan_info['HighCutOff'])

    def getSpikeSetting(self):
        self.current_spike_setting['Reference'] = ('Single',
                                                   [self.current_chan_info['ReferenceID']])
        self.current_spike_setting['Filter'] = (self.current_chan_info['LowCutOff'],
                                                self.current_chan_info['HighCutOff'])
        self.current_spike_setting['Threshold'] = ('MAD',
                                                   self.current_chan_info['Threshold'] / self.data_object.estimatedSD(self.filted_data))

    # def updateReference(self, meta_data):
    #     ref = meta_data["ReferenceID"]
    #     if ref == "nan":
    #         self.ref_checkBox.setChecked(False)
    #     else:
    #         self.ref_checkBox.setChecked(True)
    #         self.ref_comboBox.setCurrentText(ref)

    # def updateFilter(self, meta_data):
    #     filter_band = [meta_data["LowCutOff"], meta_data["HighCutOff"]]
    #     filter_band = [int(float(i)) if i !=
    #                    "nan" else 0 for i in filter_band]
    #     if filter_band[1] <= 0:
    #         self.filter_checkBox.setChecked(False)
    #         self.filter_low_spinBox.setValue(0)
    #         self.filter_high_spinBox.setValue(0)
    #     else:
    #         self.filter_checkBox.setChecked(True)
    #         self.filter_low_spinBox.setValue(filter_band[0])
    #         self.filter_high_spinBox.setValue(filter_band[1])


class ExtractWaveformSettingsDialog(Ui_ExtractWaveformSettings, QDialog):
    def __init__(self, setting, all_chan_ID, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Extract Waveform Settings Dialog")
        self.setting = setting

        # self.channel_ref_comboBox.clear()
        self.channel_ref_comboBox.addItems(map(str, all_chan_ID))
        self.initSetting()

    def initSetting(self):
        if self.setting['Reference'][0] == 'Single':
            self.channel_ref_radioButton.setChecked(True)
            self.channel_ref_comboBox.setCurrentText(
                str(self.setting['Reference'][1][0]))
        elif self.setting['Reference'][0] == 'Median':
            logger.critical('No implement error')

        self.filter_low_doubleSpinBox.setValue(self.setting['Filter'][0])
        self.filter_high_doubleSpinBox.setValue(self.setting['Filter'][1])

        if self.setting['Threshold'][0] == 'MAD':
            self.mad_thr_radioButton.setChecked(True)
            self.mad_thr_doubleSpinBox.setValue(self.setting['Threshold'][1])

        elif self.setting['Threshold'][0] == 'Const':
            self.const_thr_radioButton.setChecked(True)
            self.const_thr_doubleSpinBox.setValue(self.setting['Threshold'][1])
