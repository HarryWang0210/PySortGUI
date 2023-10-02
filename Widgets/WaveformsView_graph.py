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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.has_waveforms = False
        self.has_thr = False
        self.visible = False

        self.data = None  # SpikeSorterData object
        self.waveforms = None
        self.thr = 0.0
        self.color_palette_list = np.array(sns.color_palette(None, 64))
        self.data_scale = 1.0
        self.draw_mode = False
        self.init_plotItem()

    def init_plotItem(self):
        """
        Initialize plotWidget and plotItems.
        """
        self.plot_item = self.getPlotItem()
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

        self.waveforms_item_dict = dict()

        self.thr_item = pg.InfiniteLine(pos=self.thr, angle=0, pen="w")
        self.thr_item.setVisible(self.visible)
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
        self.visible = False
        [self.removeItem(waveforms_item)
         for waveforms_item in self.waveforms_item_dict.values()]
        self.waveforms_item_dict = dict()
        self.update_plot()

    def spike_chan_changed(self, meta_data):
        self.get_thr(meta_data["Threshold"])
        self.get_spikes(meta_data["ID"], meta_data["Label"])
        self.visible = True
        [self.removeItem(waveforms_item)
         for waveforms_item in self.waveforms_item_dict.values()]
        self.waveforms_item_dict = dict()
        self.update_plot()

    def get_thr(self, thr):
        self.thr = float(thr)
        if np.isnan(self.thr):
            self.has_thr = False
        else:
            self.has_thr = True

    def get_spikes(self, chan_ID, label):
        spikes = self.data.get_spikes(chan_ID, label)
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

    def update_plot(self):
        if self.has_waveforms and self.visible:
            self.draw_waveforms()
            if self.has_thr:
                self.draw_thr()

        [waveforms_item.setVisible(self.has_waveforms and self.visible)
         for waveforms_item in self.waveforms_item_dict.values()]
        self.thr_item.setVisible(self.has_thr and self.visible)

    def draw_thr(self):
        self.thr_item.setValue(self.thr)

    def draw_waveforms(self):
        # create elements
        xlen = self.spikes["waveforms"].shape[1]
        x_element = np.arange(xlen)
        connect_element = np.append(np.ones(xlen - 1), 0).astype(np.int32)

        # setup range
        self.plot_item.getViewBox().setXRange(
            x_element[0], x_element[-1], padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

        unique_unit = np.unique(self.spikes["units_id"])
        for units_id in unique_unit:
            pen = pg.mkPen(
                color=(self.color_palette_list[int(units_id)] * 255).astype(np.int32))
            self.waveforms_item_dict[units_id] = self.plot(pen=pen)

            data_filtered = self.spikes["waveforms"][self.spikes["units_id"] == units_id]
            n = data_filtered.shape[0]
            x = np.tile(x_element, n)
            y = np.ravel(data_filtered)
            # y = data_filtered.flatten()
            connect = np.tile(connect_element, n)
            self.waveforms_item_dict[units_id].setData(
                x=x, y=y, connect=connect)

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        pass

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        self.draw_mode = True

        self.manual_curve_item.setVisible(self.draw_mode)

        pos = event.scenePos()
        mouse_view = self.getViewBox().mapSceneToView(pos)
        x = mouse_view.x()
        y = mouse_view.y()

        self.manual_curve_item.setData([x, x], [y, y])

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
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
        print(self.manual_curve_item.getData())
        self.manual_curve_item.setVisible(self.draw_mode)
        # self.removeItem(self.manual_curve_item)
        # self.manual_curve_item.setData(np.array([]), np.array([]))

        # break
        # for i in range(len(self.spikes["units_id"])):
        #     # 瓶頸在這!! loop太多次，想辦法用matrix method
        #     y = self.spikes["waveforms"][i]
        #     x = np.arange(len(y))
        #     connect = np.ones(len(y) - 1)
        #     connect = np.append(connect, 0).astype(np.int32)

        #     units_id = int(self.spikes["units_id"][i])

        #     data_dict[units_id]["x"] = np.concatenate(
        #         (data_dict[units_id]["x"], x))
        #     data_dict[units_id]["y"] = np.concatenate(
        #         (data_dict[units_id]["y"], y))
        #     data_dict[units_id]["connect"] = np.concatenate(
        #         (data_dict[units_id]["connect"], connect))

# class WaveformsViewGL(GLWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumWidth(100)
#         self.setMinimumHeight(100)
