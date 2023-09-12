from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import seaborn as sns

from DataStructure.data import SpikeSorterData
from Widgets.GLWidget import GLWidget
from Widgets.WidgetsInterface import WidgetsInterface


class WaveformsView(pg.PlotWidget, WidgetsInterface):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.useOpenGL(True)
        self.show_thr = False
        self.has_thr = False
        self.show_waveforms = False
        self.has_waveforms = False
        self.visible = False
        self.data = None
        self.thr = 0.0
        self.waveforms = None
        self.color_palette = sns.color_palette(None, 64)
        self.data_scale = 1.0

        self.init_plotItem()

    def data_file_name_changed(self, data):
        self.data = data
        self.visible = False
        self.update()

    def spike_chan_changed(self, meta_data):
        self.get_thr(meta_data["Threshold"])
        self.get_spikes(
            self.data.get_spikes(int(meta_data["ID"]), meta_data["Label"]))
        self.visible = True
        self.update()

    def get_thr(self, thr):
        try:
            self.thr = float(thr)
            self.has_thr = True
            self.show_thr = True
        except:
            self.thr = 0.0
            self.has_thr = False
            self.show_thr = False

    def get_spikes(self, spikes):
        if spikes["units_info"] is None:
            self.has_spikes = False
            self.spikes = None
            self.data_scale = 1.0
            self.has_waveforms, self.show_waveforms = False, False
        else:
            self.has_spikes = True
            self.spikes = spikes
            self.data_scale = np.max(np.abs(spikes["waveforms"]))
            self.has_waveforms, self.show_waveforms = True, True

    def init_plotItem(self):
        background_color = QColor(
            *[int(c * 255) for c in (0.35, 0.35, 0.35)])
        self.setBackground(background_color)
        self.hideButtons()
        x_axis = self.getAxis('bottom')
        y_axis = self.getAxis('left')
        x_axis.setPen(None)
        x_axis.setStyle(showValues=False)
        y_axis.setPen(None)
        y_axis.setStyle(showValues=False)

        self.waveforms_item_dict = []
        self.pen_color_dict = dict()

        self.thr = 0.0
        self.thr_item = pg.InfiniteLine(
            pos=self.thr, angle=0, pen="g")
        self.thr_item.setVisible(self.show_thr)
        self.addItem(self.thr_item)

    def update(self):
        if self.has_thr and self.show_thr and self.visible:
            self.draw_thr()
        self.thr_item.setVisible(self.show_thr and self.visible)

        if self.has_waveforms and self.show_waveforms and self.visible:
            self.draw_waveforms()
        [waveforms_item.setVisible(self.show_waveforms and self.visible)
         for waveforms_item in self.waveforms_item_dict]

    def draw_thr(self):
        self.thr_item.setValue(self.thr)

    def draw_waveforms(self):
        if len(self.waveforms_item_dict) != 0:
            [self.removeItem(item) for item in self.waveforms_item_dict]

        for i in range(len(self.spikes["units_id"])):
            x = np.arange(len(self.spikes["waveforms"][i]))
            y = self.spikes["waveforms"][i]

            color_id = self.spikes["units_id"][i]
            if int(color_id) not in self.pen_color_dict.keys():
                self.pen_color_dict[int(color_id)] = pg.mkPen(
                    color=[c * 255 for c in self.color_palette[int(color_id)]])
            self.waveforms_item_dict.append(pg.PlotDataItem(
                x=x, y=y, pen=self.pen_color_dict[int(color_id)]))

        [self.addItem(item) for item in self.waveforms_item_dict]

        # for i in range(len(self.spikes["units_id"])):
        #     y = self.spikes["waveforms"][i]
        #     x = np.arange(len(y))
        #     connect = np.ones(len(y) - 1)
        #     connect = np.append(connect, 0).astype(np.int32)

        #     units_id = int(self.spikes["units_id"][i])
        #     if units_id not in self.waveforms_item_dict.keys():
        #         pen = pg.mkPen(color=[int(c * 255)
        #                        for c in self.color_palette[units_id]])
        #         self.waveforms_item_dict[units_id] = pg.plot(pen=pen)

        #     self.waveforms_item_dict.append(pg.PlotDataItem(
        #         x=x, y=y, pen=self.pen_color_dict[units_id]))

        # [self.addItem(item) for item in self.waveforms_item_dict]
# class WaveformsViewGL(GLWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumWidth(100)
#         self.setMinimumHeight(100)
