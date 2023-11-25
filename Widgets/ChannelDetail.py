from UI.ChannelDetailv2_ui import Ui_ChannelDetail
from UI.ExtractWaveformSettings_ui import Ui_ExtractWaveformSettings

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QAbstractItemView, QDialog

from DataStructure.data import SpikeSorterData
import numpy as np

import logging
logger = logging.getLogger(__name__)


class ChannelDetail(QtWidgets.QWidget, Ui_ChannelDetail):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Channel Detail"
        self.setupUi(self)
        self.data_object = None
        self.spike_chan = {
            'ID': None,
            'Name': None,
            'Label': None
        }
        self.extract_spike_setting = {
            'Reference': False,
            'ReferenceSetting': ('Single', []),
            'Filter': False,
            'FilterSetting': (250, 6000),
            'Treshold': False,
            'TresholdSetting': ('Const', 0)
        }

        self.treeView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setupConnections()

    def setupConnections(self):
        self.open_file_toolButton.released.connect(self.openFile)
        self.extract_wav_setting_toolButton.released.connect(
            self.setExtractWaveformParams)

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

        self.chan_info = self.data_object.chan_info
        self.generateDataModel(self.chan_info)
        self.initSpikeInfo(self.chan_info, init_id=True)
        self.file_name_lineEdit.setText(filename)

    def setExtractWaveformParams(self):
        if self.data_object == None:
            logger.warn('No SpikeSorterData.')
            return
        if self.spike_chan['ID'] == None:
            logger.warn('No selected channel.')
            return
        dialog = ExtractWaveformSettingsDialog(
            data_object=self.data_object, spike_chan=self.spike_chan, parent=self)
        result = dialog.exec_()  # 显示对话框并获取结果
        if result == QDialog.Accepted:
            return
        else:
            return

    def generateDataModel(self, df):
        model = QStandardItemModel()
        df = self.getGroup(df)

        if 'Spikes' in df.index.get_level_values('Group'):
            group_item = QStandardItem(str('Spikes'))  # Top level, Group
            model.appendRow(group_item)

            Ch_data = df.xs('Spikes', level='Group')
            model.setHorizontalHeaderLabels(
                list(Ch_data.index.names) + list(Ch_data.columns))

            for ChanlID in Ch_data.index.get_level_values('ID').unique():
                ChanlID_item = QStandardItem(str(ChanlID))  # Top level, ID
                sub_data = Ch_data.xs(ChanlID, level='ID')
                first_items = [ChanlID_item, QStandardItem(str(sub_data.iloc[0, :].name))] + [
                    QStandardItem(str(col_value)) for col_value in sub_data.iloc[0, :]]
                group_item.appendRow(first_items)

                for label, sub_row in sub_data.iloc[1:, :].iterrows():
                    values = [QStandardItem(""),  QStandardItem(
                        str(label))] + [QStandardItem(str(col_value)) for col_value in sub_row]
                    ChanlID_item.appendRow(values)

        if 'Raws' in df.index.get_level_values('Group'):
            group_item = QStandardItem(str('Raws'))  # Top level, Group
            model.appendRow(group_item)

            Ch_data = df.xs('Raws', level='Group')
            model.setHorizontalHeaderLabels(
                list(Ch_data.index.names) + list(Ch_data.columns))

            for ChanlID in Ch_data.index.get_level_values('ID').unique():
                ChanlID_item = QStandardItem(str(ChanlID))  # Top level, ID
                sub_data = Ch_data.xs(ChanlID, level='ID')
                first_items = [ChanlID_item, QStandardItem(str(sub_data.iloc[0, :].name))] + [
                    QStandardItem(str(col_value)) for col_value in sub_data.iloc[0, :]]
                group_item.appendRow(first_items)

                for label, sub_row in sub_data.iloc[1:, :].iterrows():
                    values = [QStandardItem(""),  QStandardItem(
                        str(label))] + [QStandardItem(str(col_value)) for col_value in sub_row]
                    ChanlID_item.appendRow(values)

        self.treeView.setModel(model)

        for col in range(model.columnCount()):
            self.treeView.resizeColumnToContents(col)

        selection_model = self.treeView.selectionModel()
        selection_model.selectionChanged.connect(self.on_selection_changed)

    def initSpikeInfo(self, df=None, init_id=False):
        # init label
        self.sorting_label_comboBox.clear()
        # # init ref
        # if init_id:
        #     ID_list = df.index.get_level_values('ID').unique()
        #     self.ref_comboBox.clear()
        #     self.ref_comboBox.addItems(ID_list.map(str))
        # self.ref_checkBox.setChecked(False)
        # # init filter
        # self.filter_checkBox.setChecked(False)

    def on_selection_changed(self, selected, deselected):
        model = self.treeView.model()

        columns = [model.horizontalHeaderItem(
            ind).text() for ind in range(model.columnCount())]
        indexes = selected.indexes()
        items = [model.itemFromIndex(ind) for ind in indexes]

        if items[0].parent() == None:  # Group
            self.initSpikeInfo(init_id=False)
            return
        elif items[0].parent().parent() != None:  # Label
            items[0] = items[0].parent()

        meta_data = [item.text() for item in items]
        meta_data = dict(zip(columns, meta_data))

        self.updateLabel(meta_data)
        self.spike_chan['ID'] = int(meta_data["ID"])
        self.spike_chan['Name'] = meta_data["Name"]
        self.spike_chan['Label'] = meta_data["Label"]
        # self.updateReference(meta_data)
        # self.updateFilter(meta_data)
        self.signal_spike_chan_changed.emit(meta_data)

    def getGroup(self, df):
        df = df.copy()
        df["Group"] = df["NumUnits"].isnull()
        df["Group"] = df["Group"].map({True: "Raws", False: "Spikes"})
        df.set_index('Group', append=True, inplace=True)
        return df

    def updateLabel(self, meta_data):
        labels = self.chan_info.xs(
            meta_data["ID"], level='ID').index.tolist()
        self.sorting_label_comboBox.clear()
        self.sorting_label_comboBox.addItems(labels)
        self.sorting_label_comboBox.setCurrentText(meta_data["Label"])

    def updateReference(self, meta_data):
        ref = meta_data["ReferenceID"]
        if ref == "nan":
            self.ref_checkBox.setChecked(False)
        else:
            self.ref_checkBox.setChecked(True)
            self.ref_comboBox.setCurrentText(ref)

    def updateFilter(self, meta_data):
        filter_band = [meta_data["LowCutOff"], meta_data["HighCutOff"]]
        filter_band = [int(float(i)) if i !=
                       "nan" else 0 for i in filter_band]
        if filter_band[1] <= 0:
            self.filter_checkBox.setChecked(False)
            self.filter_low_spinBox.setValue(0)
            self.filter_high_spinBox.setValue(0)
        else:
            self.filter_checkBox.setChecked(True)
            self.filter_low_spinBox.setValue(filter_band[0])
            self.filter_high_spinBox.setValue(filter_band[1])


class ExtractWaveformSettingsDialog(Ui_ExtractWaveformSettings, QDialog):
    def __init__(self, data_object, spike_chan, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Extract Waveform Settings Dialog")
        self.data_object = data_object
        self.spike_chan = spike_chan

        ID_list = self.data_object.chan_info.index.get_level_values(
            'ID').unique()
        self.channel_ref_comboBox.clear()
        self.channel_ref_comboBox.addItems(ID_list.map(str))
        self.initSpikeInfo()

    def initSpikeInfo(self):
        index = (str(self.spike_chan['ID']), self.spike_chan['Label'])
        df = self.data_object.chan_info.xs(
            index,
            level=['ID', 'Label'])
        logger.debug(df.loc[index, 'ReferenceID'])
        if not df.loc[index, 'ReferenceID'].isdigit():
            self.ref_groupBox.setChecked(False)
            logger.debug('No reference id.')
        else:
            self.ref_groupBox.setChecked(True)
            self.channel_ref_radioButton.setChecked(True)
            self.channel_ref_comboBox.setCurrentText(
                df.loc[index, 'ReferenceID'])

        if df.loc[index, 'HighCutOff'] <= 0:
            self.filter_groupBox.setChecked(False)
            logger.info('No filter.')
        else:
            self.filter_groupBox.setChecked(True)
            filter_band = df.loc[index, ['LowCutOff', 'HighCutOff']].tolist()
            filter_band = [float(i) if i !=
                           "nan" else 0 for i in filter_band]
            self.filter_low_doubleSpinBox.setValue(filter_band[0])
            self.filter_high_doubleSpinBox.setValue(filter_band[1])

        if df.loc[index, 'Threshold'] == 'nan':
            self.thr_groupBox.setChecked(False)
            logger.info('No threshold.')
        else:
            self.thr_groupBox.setChecked(True)
            self.const_thr_radioButton.setChecked(True)
            self.const_thr_doubleSpinBox.setValue(
                float(df.loc[index, 'Threshold']))

        logger.debug(df)
        # self.data_object.chan_info

    # def
    # def createComboBox(self, unit_name_list):
    #     model = QStandardItemModel()
    #     self.comboBox.setModel(model)

    #     for unit in unit_name_list:
    #         model.appendRow(QStandardItem(str(unit)))
