from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import seaborn as sns
import time
from DataStructure.data import SpikeSorterData
from Widgets.WidgetsInterface import WidgetsInterface


class WaveformsView(pg.PlotWidget, WidgetsInterface):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_selected_units_changed = QtCore.pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.data_object = None  # SpikeSorterData object
        self.data_scale = 1.0
        self.spikes = None
        self.has_spikes = False
        self.thr = 0.0
        self.has_thr = False
        self.color_palette_list = sns.color_palette(None, 64)
        self.visible = False  # overall visible\

        self.num_unit = 1
        self.current_wav_units = []
        self.current_wavs_mask = []
        self.current_wav_colors = []

        self.current_showing_units = []
        self.current_showing_data = []

        self.manual_mode = False

        self.initPlotItem()

    def initPlotItem(self):
        """
        Initialize plotWidget and plotItems.
        """
        self.plot_item = self.getPlotItem()
        self.plot_item.clear()
        self.plot_item.setMenuEnabled(False)
        # setup background
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackground(background_color)

        # hide auto range button
        self.plot_item.hideButtons()

        # remove x, y axis
        self.plot_item.hideAxis('bottom')
        self.plot_item.hideAxis('left')

        self.waveforms_item_list = []

        self.thr_item = pg.InfiniteLine(pos=self.thr, angle=0, pen="w")
        self.thr_item.setVisible(False)
        self.addItem(self.thr_item)

        self.manual_curve_item = pg.PlotCurveItem(
            pen=pg.mkPen('r', width=2), clickable=False)
        self.manual_curve_item.setZValue(1)
        self.manual_curve_item.setVisible(False)
        self.addItem(self.manual_curve_item)

        self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.waveforms_item_list = []
        self.updatePlot()

    def spike_chan_changed(self, meta_data):
        self.getThreshold(meta_data["Threshold"])
        self.getSpikes(meta_data["ID"], meta_data["Label"])
        self.waveforms_item_list = []

    # def selected_units_changed(self, selected_rows):
    #     self.visible = [False] * self.num_unit
    #     for i in selected_rows:
    #         self.visible[i] = True
    #     self.redraw = False
    #     self.updatePlot()

    def showing_spikes_data_changed(self, spikes_data):
        self.current_wav_units = spikes_data['current_wav_units']
        self.current_wavs_mask = np.isin(spikes_data['current_wav_units'],
                                         spikes_data['current_showing_units'])
        self.current_showing_units = spikes_data['current_showing_units']
        self.num_unit = len(np.unique(self.current_wav_units))

        self.current_wav_colors = self.getColor(self.current_wav_units)
        self.setCurrentShowingData()
        self.updatePlot()

    def activate_manual_mode(self, state):
        self.manual_mode = state

    def getThreshold(self, thr):
        self.thr = float(thr)
        if np.isnan(self.thr):
            self.has_thr = False
        else:
            self.has_thr = True

    def getSpikes(self, chan_ID, label):
        spikes = self.data_object.getSpikes(chan_ID, label)
        if spikes["unitInfo"] is None:
            self.has_spikes = False
            self.spikes = None
            self.visible = False

            self.data_scale = 1.0
        else:
            self.has_spikes = True
            self.spikes = spikes
            self.visible = True

            self.data_scale = np.max(np.abs(spikes["waveforms"]))

    def getColor(self, unit_data):
        n = len(unit_data)
        color = np.zeros((n, 3))

        for i in range(n):
            color[i, :] = self.color_palette_list[int(unit_data[i])]
        color = color * 255
        return color.astype(np.int32)

    def setCurrentShowingData(self):
        self.current_showing_data = self.spikes['waveforms'][self.current_wavs_mask]

    def updatePlot(self):
        if self.visible and self.has_spikes:
            self.drawWaveforms(self.current_showing_data,
                               self.current_showing_units)
            if self.has_thr:
                self.drawThreshold()

        for waveforms_item in self.waveforms_item_list:
            waveforms_item.setVisible(self.visible and self.has_spikes)

        self.thr_item.setVisible(self.visible and self.has_thr)

    def drawThreshold(self):
        self.thr_item.setValue(self.thr)

    def drawWaveforms(self, waveforms, unit_ID):
        self.removeWaveformItems()
        # create elements
        xlen = waveforms.shape[1]
        x_element = np.arange(xlen)
        connect_element = np.append(np.ones(xlen - 1), 0).astype(np.int32)

        # setup range
        self.plot_item.getViewBox().setXRange(
            x_element[0], x_element[-1], padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

        for ID in unit_ID:
            ID_mask = self.current_wav_units[self.current_wavs_mask] == ID
            data_filtered = waveforms[ID_mask]
            n = data_filtered.shape[0]

            if n == 0:
                continue

            x = np.tile(x_element, n)
            y = np.ravel(data_filtered)
            connect = np.tile(connect_element, n)

            color = self.current_wav_colors[self.current_wav_units == ID][0, :]
            pen = pg.mkPen(
                color=color)

            self.waveforms_item_list.append(
                self.plot(x=x, y=y, connect=connect, pen=pen))

    def removeWaveformItems(self):
        for waveforms_item in self.waveforms_item_list:
            self.removeItem(waveforms_item)
        self.waveforms_item_list = []

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        pass

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        self.manual_curve_item.setVisible(True)

        pos = event.scenePos()
        mouse_view = self.getViewBox().mapSceneToView(pos)
        x = mouse_view.x()
        y = mouse_view.y()

        self.manual_curve_item.setData([x, x], [y, y])

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
        if self.manual_mode:
            pos = event.scenePos()
            mouse_view = self.getViewBox().mapSceneToView(pos)
            x = mouse_view.x()
            y = mouse_view.y()

            x_data, y_data = self.manual_curve_item.getData()

            x_data = np.append(x_data, x)
            y_data = np.append(y_data, y)

            self.manual_curve_item.setData(x_data, y_data)

    def graphMouseReleaseEvent(self, event):
        """Overwrite PlotItem.scene().mouseReleaseEvent."""
        self.manual_curve_item.setVisible(False)
