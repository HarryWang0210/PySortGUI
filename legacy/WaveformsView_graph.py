import logging

import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QGuiApplication
from PyQt5.QtWidgets import QApplication

from pysortgui.DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from pysortgui.Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class WaveformsView(pg.PlotWidget, WidgetsInterface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.GLOBAL_WAVS_LIMIT = 50000

        self.redraw_wavs = False

        self.data_object = None  # SpikeSorterData object
        # self.data_scale = 1.0
        # self.spikes = None
        # self.has_spikes = False
        self.thr = 0.0
        # self.has_thr = False
        self.color_palette_list = sns.color_palette('bright', 64)
        self.plot_visible = False
        self.widget_visible = False

        self._x_boundary: tuple[int, int] = (0, 0)
        self._x_range: tuple[int, int] = (0, 1)
        # self._y_boundary: tuple[int, int] = (0, 0)
        self._y_range: tuple[int, int] = (-1000, 1000)

        self.current_wavs_mask = []
        self.current_showing_units = []

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

    def widgetVisibilityChanged(self, visible: bool):
        self.widget_visible = visible
        self.updatePlot()

    def data_file_name_changed(self, data):
        self.data_object = data
        self.plot_visible = False
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object

        # self.has_spikes = True
        self.redraw_wavs = True
        self.plot_visible = True
        if self.current_spike_object is None:
            # self.has_spikes = False
            self.plot_visible = False
            self.updatePlot()
            return
        data_scale = np.max(np.abs(self.current_spike_object.waveforms)) / 2
        self._y_range = (-data_scale, data_scale)

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
    #     self.plot_visible = [False] * self.num_unit
    #     for i in selected_rows:
    #         self.plot_visible[i] = True
    #     self.redraw = False
    #     self.updatePlot()

    def showing_units_changed(self, showing_unit_IDs):
        self.current_showing_units = showing_unit_IDs
        self.current_wavs_mask = np.isin(self.current_spike_object.unit_IDs,
                                         self.current_showing_units)
        self.redraw_wavs = True
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
        import time
        start = time.perf_counter()
        current_showing_data = self.current_spike_object._waveforms[self.current_wavs_mask]
        logger.info(f'get current data{time.perf_counter() - start}')
        selected, wav_index = data
        if selected:
            start = time.perf_counter()
            y = current_showing_data[wav_index, :]
            x = np.arange(len(y))
            self.select_point_item.setData(x, y)
            logger.info(f'plot data{time.perf_counter() - start}')

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
    #         self.plot_visible = False

    #         self.data_scale = 1.0
    #     else:
    #         self.has_spikes = True
    #         self.spikes = spikes
    #         self.plot_visible = True

    #         self.data_scale = np.max(np.abs(spikes["waveforms"]))
    #     logger.debug(self.plot_visible)

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
        visible = self.plot_visible and self.widget_visible
        if visible and not self.current_spike_object is None:
            if self.redraw_wavs:
                self.removeWaveformItems()
                self.drawWaveforms()
                self.redraw_wavs = False
            # if self.has_thr:
            self.drawThreshold()

        # self.plot_item.getViewBox().setXRange(*self._x_range, padding=0)
        self.plot_item.getViewBox().setYRange(*self._y_range, padding=0)

        for waveforms_item in self.waveforms_item_list:
            waveforms_item.setVisible(
                visible and not self.current_spike_object is None)

        self.thr_item.setVisible(visible)

    def drawThreshold(self):
        self.thr_item.setValue(self.current_spike_object.threshold)

    def drawWaveforms(self):
        import time

        start = time.perf_counter()
        # create elements
        waveforms = self.current_spike_object._waveforms[self.current_wavs_mask]
        unit_IDs = self.current_spike_object._unit_IDs[self.current_wavs_mask]
        xlen = waveforms.shape[1]
        x_element = np.arange(xlen)
        connect_element = np.append(np.ones(xlen - 1), 0).astype(np.int32)
        logger.info(f'create elements {time.perf_counter() - start}')

        # setup range
        self.plot_item.getViewBox().setXRange(
            x_element[0], x_element[-1], padding=0)
        # self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

        unit_color_map = dict(zip(self.current_spike_object.unit_header['ID'], np.arange(
            self.current_spike_object.unit_header.shape[0], dtype=int)))

        start = time.perf_counter()
        # logger.debug(unit_color_map)
        ds_index = self.downsamplingWaveforms(waveforms, unit_IDs)

        waveforms = waveforms[ds_index]
        unit_IDs = unit_IDs[ds_index]
        logger.info(f'downsamplingWaveforms {time.perf_counter() - start}')

        # if len(unit_IDs) > self.GLOBAL_WAVS_LIMIT:
        start = time.perf_counter()
        for ID in self.current_showing_units:
            data_filtered = waveforms[unit_IDs == ID]
            n = data_filtered.shape[0]

            if n == 0:
                continue

            x = np.tile(x_element, n)
            y = np.ravel(data_filtered)
            connect = np.tile(connect_element, n)

            color = self.color_palette_list[unit_color_map[int(ID)]]
            color = (np.array(color) * 255).astype(int)

            self.waveforms_item_list.append(
                self.plot(x=x, y=y, connect=connect, pen=pg.mkPen(color=color)))
        logger.info(f'plot {time.perf_counter() - start}')

    def removeWaveformItems(self):
        for waveforms_item in self.waveforms_item_list:
            self.removeItem(waveforms_item)
        self.waveforms_item_list = []

    def downsamplingWaveforms(self, waveforms, unit_IDs):
        length = waveforms.shape[0]
        if length > self.GLOBAL_WAVS_LIMIT:
            msg = "reducing number of waveforms to plot"
            logger.info(msg)
            max_set = []
            min_set = []
            # waveforms = self.data_spike_chan.Waveforms
            for unit in np.unique(unit_IDs):
                unit_mask = unit_IDs == unit
                waveforms_unit = waveforms[unit_mask, :]
                max_set_u = np.argmax(waveforms_unit, axis=0)
                min_set_u = np.argmin(waveforms_unit, axis=0)
                inv = np.where(unit_mask)[0]
                max_set.append(inv[max_set_u])
                min_set.append(inv[min_set_u])

            max_set = np.concatenate(max_set)
            min_set = np.concatenate(min_set)

            rand_set = np.random.permutation(length)[:self.GLOBAL_WAVS_LIMIT]

            ds_index = np.unique(np.hstack((max_set, rand_set, min_set)))

            return ds_index

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        modifiers = QGuiApplication.keyboardModifiers()
        if (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
            """scale y axis."""
            delta = int(event.delta() / 120)
            # current_range = self.plot_item.getViewBox().state['viewRange']
            data_scale = int(self._y_range[1] / (1 + delta / 10))
            self._y_range = (-data_scale, data_scale)

            # self.redraw_data = True
            # self.redraw_bg = True
            # self.redraw_events = True
            # self.redraw_spikes = True
            self.updatePlot()

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
