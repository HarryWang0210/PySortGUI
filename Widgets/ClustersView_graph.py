from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import seaborn as sns
import time
from DataStructure.data import SpikeSorterData
from Widgets.WidgetsInterface import WidgetsInterface


class ClustersView(gl.GLViewWidget, WidgetsInterface):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_selected_units_changed = QtCore.pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Clusters View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.visible = False
        self.num_waveforms = 0
        self.waveforms_visible = []
        self.color_palette_list = sns.color_palette(None, 64)

        self.init_plotItem()

    def init_plotItem(self):
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackgroundColor(background_color)

        self.setCameraPosition(distance=2,  elevation=45, azimuth=45)

        # 添加XYZ轴
        axis_len = 1
        axis_pos = np.array([[0, 0, 0], [axis_len, 0, 0],
                             [0, 0, 0], [0, axis_len, 0],
                             [0, 0, 0], [0, 0, axis_len]])
        axis_color = np.array([[1, 0, 0, 1], [1, 0, 0, 1],
                               [0, 1, 0, 1], [0, 1, 0, 1],
                               [0, 0, 1, 1], [0, 0, 1, 1]])
        axis = gl.GLLinePlotItem(
            pos=axis_pos, color=axis_color,  width=2, mode='lines')
        self.addItem(axis)

        axis_text = ["PCA1", "PCA2", "PCA3"]
        label_x = gl.GLTextItem(text=axis_text[0], color=(255, 0, 0, 255))
        label_y = gl.GLTextItem(text=axis_text[1], color=(0, 255, 0, 255))
        label_z = gl.GLTextItem(text=axis_text[2], color=(0, 0, 255, 255))
        self.addItem(label_x)
        self.addItem(label_y)
        self.addItem(label_z)
        label_x.setData(pos=(axis_len * 1.1, 0, 0))
        label_y.setData(pos=(0, axis_len * 1.1, 0))
        label_z.setData(pos=(0, 0, axis_len * 1.1))

        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque')
        self.addItem(self.scatter)

    def data_file_name_changed(self, data):
        self.data = data
        self.visible = False
        self.update_plot()

    def spike_chan_changed(self, meta_data):
        self.compute_pca(meta_data["ID"], meta_data["Label"])
        self.visible = True
        self.waveforms_visible = [True] * self.num_waveforms
        self.update_plot()

    def selected_units_changed(self, selected_rows):
        self.waveforms_visible = np.isin(
            self.spikes["unitID"], list(selected_rows))
        self.update_plot()

    def compute_pca(self, chan_ID, label):
        spikes = self.data.get_spikes(chan_ID, label)
        if spikes["unitInfo"] is None:
            # self.has_spikes = False
            self.spikes = None
            self.has_waveforms = False
        else:
            # self.has_spikes = True
            self.spikes = spikes
            self.has_waveform = True
        self.num_waveforms = self.spikes["waveforms"].shape[0]
        self.pca = self.data.wavforms_pca(chan_ID, label)
        self.point_color = self.get_color()

    def update_plot(self):
        if self.visible:
            self.scatter.setData(pos=self.pca[self.waveforms_visible],
                                 size=3,
                                 color=self.point_color[self.waveforms_visible])
        self.scatter.setVisible(self.visible)

    def get_color(self):
        n = self.num_waveforms
        color = np.zeros((n, 3))

        for i in range(n):
            color[i, :] = self.color_palette_list[int(
                self.spikes["unitID"][i])]
        color = np.hstack((color, np.ones((n, 1))))
        return color
