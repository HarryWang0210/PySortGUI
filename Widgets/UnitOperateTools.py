from UI.UnitOperateToolsUIv3_ui import Ui_UnitOperateTools
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QColor
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QAbstractItemView, QCheckBox, QStyledItemDelegate
from PyQt5.QtCore import Qt
import numpy as np
import seaborn as sns
from DataStructure.data import SpikeSorterData


class UnitOperateTools(QtWidgets.QWidget, Ui_UnitOperateTools):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_selected_units_changed = QtCore.pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Unit Operate Tools"
        self.setupUi(self)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.data_object = None
        self.visible = False
        self.spikes = None
        self.has_waveforms = False
        self.locked_rows_list = []
        self.selected_rows_list = []
        self.color_palette_list = sns.color_palette(None, 64)

        self.initDataModel()

        delegate = ColorDelegate(self.color_palette_list)
        self.tableView.setItemDelegate(delegate)

    def initDataModel(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.tableView.verticalHeader().setVisible(False)

        selection_model = self.tableView.selectionModel()
        selection_model.selectionChanged.connect(self.on_selection_changed)

    def setDataModel(self, df):
        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])

        self.appendUnitRow(0, "All")
        self.appendUnitRow(1, "All_Sorted_Units")

        for row in range(df.shape[0]):
            self.appendUnitRow(row + 2, bytes.decode(df.loc[row, "Name"]))
        self.tableView.resizeColumnToContents(0)

    def appendUnitRow(self, row, unit_name):
        model = self.tableView.model()

        # 創建一個 CheckBox Widget
        checkbox = QCheckBox()
        checkbox.setProperty("row", row)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.checkbox_state_changed)

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

    def checkbox_state_changed(self, state):
        checkbox = self.sender()
        row = checkbox.property("row")
        if state == Qt.Checked:
            self.locked_rows_list.append(row)
        elif state == Qt.Unchecked:
            self.locked_rows_list.remove(row)
        self.sendSelectedID()

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        # self.update_plot()

    def spike_chan_changed(self, meta_data):
        spikes = self.data_object.getSpikes(
            meta_data["ID"], meta_data["Label"])
        if spikes["unitInfo"] is None:
            self.spikes = None
            self.has_waveforms = False
            model = self.tableView.model()
            model.clear()
        else:
            self.spikes = spikes
            self.has_waveforms = True
            self.setDataModel(self.spikes["unitInfo"])

        self.visible = True
        self.locked_rows_list = []
        self.selected_rows_list = []

    def on_selection_changed(self, selected, deselected):
        selection_model = self.tableView.selectionModel()
        selected_indexes = selection_model.selectedRows()
        self.selected_rows_list = [index.row() for index in selected_indexes]
        self.sendSelectedID()

    def sendSelectedID(self):
        all_selected_rows = self.selected_rows_list + self.locked_rows_list

        selected_ID = [row - 2 for row in all_selected_rows]

        model = self.tableView.model()

        # All_Sorted_Units
        if 1 in all_selected_rows:
            selected_ID += [row for row in range(1, model.rowCount() - 2)]

        # All
        if 0 in all_selected_rows:
            selected_ID += [row for row in range(model.rowCount() - 2)]

        self.signal_selected_units_changed.emit(set(selected_ID))


class ColorDelegate(QStyledItemDelegate):
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
