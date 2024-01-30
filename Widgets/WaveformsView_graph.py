from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import seaborn as sns
import time
from Widgets.WidgetsInterface import WidgetsInterface
from DataStructure.datav3 import SpikeSorterData, ContinuousData, DiscreteData

import logging
logger = logging.getLogger(__name__)


class WaveformsView(pg.PlotWidget, WidgetsInterface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.data_object = None  # SpikeSorterData object
        self.data_scale = 1.0
        # self.spikes = None
        # self.has_spikes = False
        self.thr = 0.0
        self.has_thr = False
        self.color_palette_list = sns.color_palette('bright', 64)
        self.visible = False  # overall visible

        # self.num_unit = 1
        # self.current_wav_units = []
        self.current_wavs_mask = []
        # self.current_wav_colors = []

        self.current_showing_units = []
        # self.current_showing_data = []

        self.current_spike_object = None
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

        self.select_point_item = pg.PlotCurveItem(
            pen=pg.mkPen('w', width=2), clickable=False)
        self.select_point_item.setZValue(1)
        self.select_point_item.setVisible(False)
        self.addItem(self.select_point_item)

        self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object

        # self.has_spikes = True
        self.visible = True
        if self.current_spike_object is None:
            # self.has_spikes = False
            self.visible = False
            self.updatePlot()
            return
        self.data_scale = np.max(np.abs(self.current_spike_object.waveforms))

    # def spike_chan_changed(self, current_chan_info):
    #     self.getThreshold(current_chan_info["Threshold"])
    #     # self.getSpikes(current_chan_info["ID"], current_chan_info["Label"])
    #     self.getSpikes(current_chan_info)

    # def extract_wav_changed(self, wav_dict):
    #     self.has_spikes = True
    #     self.spikes['timestamps'] = wav_dict['timestamps']
    #     self.spikes['waveforms'] = wav_dict['waveforms']
    #     self.spikes['unitInfo'] = None
    #     self.spikes['unitID'] = np.zeros(len(self.spikes['timestamps']))
    #     self.current_wav_units = self.spikes['unitID']
    #     logger.debug('extract_wav_changed')
    #     self.updatePlot()

    # def sorting_result_changed(self, unitID):
    #     self.has_spikes = True
    #     self.spikes['unitInfo'] = None
    #     self.spikes['unitID'] = unitID
    #     self.current_wav_units = self.spikes['unitID']

    #     logger.debug('sorting_result_changed')
    #     self.updatePlot()

    # def selected_units_changed(self, selected_rows):
    #     self.visible = [False] * self.num_unit
    #     for i in selected_rows:
    #         self.visible[i] = True
    #     self.redraw = False
    #     self.updatePlot()

    def showing_units_changed(self, showing_unit_IDs):
        self.current_showing_units = showing_unit_IDs
        self.current_wavs_mask = np.isin(self.current_spike_object.unit_IDs,
                                         self.current_showing_units)
        self.updatePlot()

    # def showing_spikes_data_changed(self, spikes_data):
    #     if self.has_spikes:
    #         self.current_wav_units = spikes_data['current_wav_units']
    #         self.current_wavs_mask = np.isin(spikes_data['current_wav_units'],
    #                                          spikes_data['current_showing_units'])
    #         self.current_showing_units = spikes_data['current_showing_units']
    #         # self.num_unit = len(np.unique(self.current_wav_units))

    #         self.current_wav_colors = self.getColor(self.current_wav_units)
    #         self.setCurrentShowingData()
    #     self.updatePlot()

    def activate_manual_mode(self, state):
        self.manual_mode = state

    def select_point(self, data):
        selected, wav_index = data
        if selected:
            y = self.current_spike_object.waveforms[wav_index, :]
            x = np.arange(len(y))
            self.select_point_item.setData(x, y)

        self.select_point_item.setVisible(selected)

    # def getThreshold(self, thr):
    #     self.thr = float(thr)
    #     if np.isnan(self.thr):
    #         self.has_thr = False
    #     else:
    #         self.has_thr = True

    # def getSpikes(self, current_chan_info):
    #     # def getSpikes(self, chan_ID, label):
    #     spikes = current_chan_info

    #     # spikes = self.data_object.getSpikes(chan_ID, label)
    #     if spikes["unitID"] is None:
    #         self.has_spikes = False
    #         self.spikes = None
    #         self.visible = False

    #         self.data_scale = 1.0
    #     else:
    #         self.has_spikes = True
    #         self.spikes = spikes
    #         self.visible = True

    #         self.data_scale = np.max(np.abs(spikes["waveforms"]))
    #     logger.debug(self.visible)

    # def getColor(self, unit_data):
    #     n = len(unit_data)
    #     color = np.zeros((n, 3))

    #     for i in range(n):
    #         color[i, :] = self.color_palette_list[int(unit_data[i])]
    #     color = color * 255
    #     return color.astype(np.int32)

    # def setCurrentShowingData(self):
    #     if self.current_spike_object is None:
    #         return
    #     self.current_showing_data = self.spikes['waveforms'][self.current_wavs_mask]

    def updatePlot(self):
        if self.visible and not self.current_spike_object is None:
            self.removeWaveformItems()

            self.drawWaveforms()
            # if self.has_thr:
            self.drawThreshold()

        for waveforms_item in self.waveforms_item_list:
            waveforms_item.setVisible(
                self.visible and not self.current_spike_object is None)

        self.thr_item.setVisible(self.visible)

    def drawThreshold(self):
        self.thr_item.setValue(self.current_spike_object.threshold)

    def drawWaveforms(self):
        # create elements
        waveforms = self.current_spike_object.waveforms[self.current_wavs_mask]
        unit_IDs = self.current_spike_object.unit_IDs[self.current_wavs_mask]
        xlen = waveforms.shape[1]
        x_element = np.arange(xlen)
        connect_element = np.append(np.ones(xlen - 1), 0).astype(np.int32)

        # setup range
        self.plot_item.getViewBox().setXRange(
            x_element[0], x_element[-1], padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

        for ID in self.current_showing_units:
            data_filtered = waveforms[unit_IDs == ID]
            n = data_filtered.shape[0]

            if n == 0:
                continue

            x = np.tile(x_element, n)
            y = np.ravel(data_filtered)
            connect = np.tile(connect_element, n)

            color = self.color_palette_list[int(ID)]
            color = (np.array(color) * 255).astype(int)
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
