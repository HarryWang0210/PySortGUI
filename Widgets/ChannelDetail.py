from UI.ChannelDetail_ui import Ui_ChannelDetail
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QAbstractItemView

from DataStructure.data import SpikeSorterData
import numpy as np


class ChannelDetail(QtWidgets.QWidget, Ui_ChannelDetail):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Channel Detail"
        self.setupUi(self)
        self.data = None

        self.treeView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setup_connections()

    def setup_connections(self):
        self.open_file_toolButton.released.connect(self.open_file)

    def open_file(self):
        """Open file manager and load selected file. """
        self.file_type_dict = {  # "openephy": "Open Ephys Format (*.continuous)",
            "pyephys": "pyephys format (*.h5)"}  # File types to load
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "./",
                                                                   ";;".join(self.file_type_dict.values()))  # start path
        if filename == "":
            return

        if isinstance(self.data, SpikeSorterData):
            if filename == self.data.filename:
                return

        self.data = SpikeSorterData(filename)
        self.signal_data_file_name_changed.emit(self.data)

        self.chan_info = self.data.chan_info
        self.generate_data_model(self.chan_info)
        self.init_spike_info(self.chan_info, init_id=True)
        self.file_name_lineEdit.setText(filename)

    def generate_data_model(self, df):
        model = QStandardItemModel()
        df = self.get_group(df)

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

    def init_spike_info(self, df=None, init_id=False):
        # init label
        self.sorting_label_comboBox.clear()
        # init ref
        if init_id:
            ID_list = df.index.get_level_values('ID').unique()
            self.ref_comboBox.clear()
            self.ref_comboBox.addItems(ID_list.map(str))
        self.ref_checkBox.setChecked(False)
        # init filter
        self.filter_checkBox.setChecked(False)

    def on_selection_changed(self, selected, deselected):
        model = self.treeView.model()

        columns = [model.horizontalHeaderItem(
            ind).text() for ind in range(model.columnCount())]
        indexes = selected.indexes()
        items = [model.itemFromIndex(ind) for ind in indexes]

        if items[0].parent() == None:  # Group
            self.init_spike_info(init_id=False)
            return
        elif items[0].parent().parent() != None:  # Label
            items[0] = items[0].parent()

        meta_data = [item.text() for item in items]
        meta_data = dict(zip(columns, meta_data))

        self.update_label(meta_data)
        self.update_ref(meta_data)
        self.update_filter(meta_data)
        self.signal_spike_chan_changed.emit(meta_data)

    def get_group(self, df):
        df = df.copy()
        df["Group"] = df["NumUnits"].isnull()
        df["Group"] = df["Group"].map({True: "Raws", False: "Spikes"})
        df.set_index('Group', append=True, inplace=True)
        return df

    def update_label(self, meta_data):
        labels = self.chan_info.xs(
            meta_data["ID"], level='ID').index.tolist()
        self.sorting_label_comboBox.clear()
        self.sorting_label_comboBox.addItems(labels)
        self.sorting_label_comboBox.setCurrentText(meta_data["Label"])

    def update_ref(self, meta_data):
        ref = meta_data["ReferenceID"]
        if ref == "nan":
            self.ref_checkBox.setChecked(False)
        else:
            self.ref_checkBox.setChecked(True)
            self.ref_comboBox.setCurrentText(ref)

    def update_filter(self, meta_data):
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
