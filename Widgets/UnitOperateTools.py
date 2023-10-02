from UI.UnitOperateToolsUIv2_ui import Ui_UnitOperateTools
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
    signal_selected_units_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Unit Operate Tools"
        self.setupUi(self)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.color_palette_list = sns.color_palette(None, 64)
        self.init_data_model()

        delegate = ColorDelegate(self.color_palette_list)
        self.tableView.setItemDelegate(delegate)

    def init_data_model(self):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])
        self.tableView.setModel(model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.tableView.verticalHeader().setVisible(False)

        selection_model = self.tableView.selectionModel()
        selection_model.selectionChanged.connect(self.on_selection_changed)

    def set_data_model(self, df):
        model = self.tableView.model()
        model.clear()
        model.setHorizontalHeaderLabels(["Locked", "UnitName"])

        self.append_unit_row(0, "All")
        self.append_unit_row(1, "All_Sorted_Units")

        for row in range(df.shape[0]):
            self.append_unit_row(row + 2, bytes.decode(df.loc[row, "Name"]))
        self.tableView.resizeColumnToContents(0)

    def append_unit_row(self, row, unit_name):
        model = self.tableView.model()

        # 創建一個 CheckBox Widget
        checkbox = QCheckBox()
        checkbox.setChecked(False)

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

    def data_file_name_changed(self, data):
        self.data = data
        self.visible = False
        # self.update_plot()

    def spike_chan_changed(self, meta_data):
        spikes = self.data.get_spikes(meta_data["ID"], meta_data["Label"])
        if spikes["units_info"] is None:
            # self.has_spikes = False
            self.spikes = None
            self.has_waveforms = False
        else:
            # self.has_spikes = True
            self.spikes = spikes
            self.set_data_model(self.spikes["units_info"])

            self.has_waveform = True

        # self.unit_color = self.get_color()
        self.visible = True

    def on_selection_changed(self, selected, deselected):
        selection_model = self.tableView.selectionModel()
        selected_indexes = selection_model.selectedRows()
        selected_rows = [index.row() for index in selected_indexes]

        self.signal_selected_units_changed.emit
        print(selected_rows)


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
