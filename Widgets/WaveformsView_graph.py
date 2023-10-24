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

        self.has_waveforms = False
        self.has_thr = False
        self.visible = [False]
        self.num_unit = 1
        self.redraw = True
        self.data = None  # SpikeSorterData object
        self.plot_waveforms = {"waveforms": None,
                               "unitID": None}
        self.thr = 0.0
        self.color_palette_list = np.array(sns.color_palette(None, 64))
        self.data_scale = 1.0
        self.draw_mode = False
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
        self.addItem(self.manual_curve_item)

        self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def data_file_name_changed(self, data):
        self.data = data
        self.visible = [False]
        self.initPlotItem()
        self.waveforms_item_list = []
        self.redraw = True
        self.updatePlot()

    def spike_chan_changed(self, meta_data):
        self.getThreshold(meta_data["Threshold"])
        self.getSpikes(meta_data["ID"], meta_data["Label"])
        self.visible = [True] * self.num_unit
        self.initPlotItem()
        self.waveforms_item_list = []
        self.redraw = True
        self.updatePlot()

    def selected_units_changed(self, selected_rows):
        self.visible = [False] * self.num_unit
        for i in selected_rows:
            self.visible[i] = True
        self.redraw = False
        self.updatePlot()

    def getThreshold(self, thr):
        self.thr = float(thr)
        if np.isnan(self.thr):
            self.has_thr = False
        else:
            self.has_thr = True

    def getSpikes(self, chan_ID, label):
        spikes = self.data.getSpikes(chan_ID, label)
        if spikes["unitInfo"] is None:
            self.has_spikes = False
            self.spikes = None
            self.plot_waveforms["waveforms"] = None
            self.plot_waveforms["unitID"] = None
            self.data_scale = 1.0
            self.has_waveforms, self.show_waveforms = False, False
        else:
            self.has_spikes = True
            self.spikes = spikes
            self.num_unit = spikes["unitInfo"].shape[0]
            self.plot_waveforms["waveforms"] = spikes["waveforms"]
            self.plot_waveforms["unitID"] = spikes["unitID"]
            self.data_scale = np.max(np.abs(spikes["waveforms"]))
            self.has_waveforms, self.show_waveforms = True, True

    def updatePlot(self):
        if self.has_waveforms and np.any(self.visible):
            if self.redraw:
                self.drawWaveforms(self.plot_waveforms["waveforms"],
                                   self.plot_waveforms["unitID"])
                self.redraw = False
            if self.has_thr:
                self.drawThreshold()

        for unitID in range(len(self.waveforms_item_list)):
            waveforms_item = self.waveforms_item_list[unitID]
            waveforms_item.setVisible(
                self.has_waveforms and self.visible[unitID])

        self.thr_item.setVisible(self.has_thr and np.any(self.visible))

    def drawThreshold(self):
        self.thr_item.setValue(self.thr)

    def drawWaveforms(self, waveforms, unitID):
        # create elements
        xlen = waveforms.shape[1]
        x_element = np.arange(xlen)
        connect_element = np.append(np.ones(xlen - 1), 0).astype(np.int32)

        # setup range
        self.plot_item.getViewBox().setXRange(
            x_element[0], x_element[-1], padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

        for ID in range(self.num_unit):
            pen = pg.mkPen(
                color=(self.color_palette_list[ID] * 255).astype(np.int32))
            self.waveforms_item_list.append(self.plot(pen=pen))

            data_filtered = waveforms[unitID == ID]
            n = data_filtered.shape[0]

            if n == 0:
                continue

            x = np.tile(x_element, n)
            y = np.ravel(data_filtered)
            # y = data_filtered.flatten()
            connect = np.tile(connect_element, n)
            self.waveforms_item_list[ID].setData(x=x, y=y, connect=connect)

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        pass

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        self.draw_mode = True
        self.redraw = True

        self.manual_curve_item.setVisible(self.draw_mode)

        pos = event.scenePos()
        mouse_view = self.getViewBox().mapSceneToView(pos)
        x = mouse_view.x()
        y = mouse_view.y()

        self.manual_curve_item.setData([x, x], [y, y])

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
        self.redraw = True
        if self.draw_mode:
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
        self.draw_mode = False
        self.redraw = False
        self.manual_curve_item.setVisible(self.draw_mode)
