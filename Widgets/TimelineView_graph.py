from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QGuiApplication

import pyqtgraph as pg
import numpy as np
import seaborn as sns
from UI.TimelineView_ui import Ui_TimelineView
from DataStructure.data import SpikeSorterData


class TimelineView(QtWidgets.QWidget, Ui_TimelineView):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Timeline View"
        self.setupUi(self)
        self.graphWidget = TimelineViewGL(self)
        self.openglLayout.addWidget(self.graphWidget)
        self.data = None
        self.setup_connections()

    def setup_connections(self):
        self.thr_pushButton.clicked.connect(self.show_thr)
        self.events_pushButton.clicked.connect(self.show_events)
        self.spikes_pushButton.clicked.connect(self.show_spikes)
        # self.raw_pushButton.clicked.connect(self.show_raw)

    def data_file_name_changed(self, data):
        self.data = data
        self.graphWidget.visible = False
        self.graphWidget.update_plot()

    def spike_chan_changed(self, meta_data):
        self.graphWidget.get_raw(self.data.get_raw(int(meta_data["ID"])))
        self.graphWidget.get_thr(meta_data["Threshold"])
        self.graphWidget.get_spikes(
            self.data.get_spikes(int(meta_data["ID"]), meta_data["Label"]))
        self.graphWidget.init_param()
        self.graphWidget.visible = True

        self.graphWidget.update_plot()

    def show_thr(self, checked):
        self.graphWidget.show_thr = checked
        self.graphWidget.update_plot()

    def show_events(self, checked):
        self.graphWidget.show_events = checked
        self.graphWidget.update_plot()

    def show_spikes(self, checked):
        self.graphWidget.show_spikes = checked
        self.graphWidget.update_plot()


class TimelineViewGL(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.MIN_DATA_SHOW = 100
        self.MAX_DATA_SHOW = 30000

        self.visible = False
        self.show_thr = False
        self.has_thr = False
        self.show_events = False
        self.has_events = False
        self.show_spikes = False
        self.has_spikes = False

        self.raw = None

        self.offset = 0
        self.data_scale = 1000
        self.num_data_show = 1000  # initial number of data points show in window
        self.color_palette = sns.color_palette(None, 64)

        self.init_plotItem()

    def get_raw(self, raw):
        self.raw = raw
        self.raw_len = len(raw)
        x = np.arange(self.raw_len)
        y = raw
        self.raw_item.setData(x=x, y=y)

    def get_thr(self, thr):
        try:
            self.thr = float(thr)
            self.has_thr = True
        except:
            self.thr = 0.0
            self.has_thr = False

    def get_events(self, events):
        self.events = events
        self.has_events = True

    def get_spikes(self, spikes):
        if spikes["unitInfo"] is None:
            self.has_spikes = False
            self.spikes = None
        else:
            self.has_spikes = True
            self.spikes = spikes

    def init_param(self):
        self.data_scale = np.max(np.abs(self.raw))
        self.num_data_show = 1000  # initial number of data points show in window

    def init_plotItem(self):
        self.plot_item = self.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.setClipToView(True)

        # setup background
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackground(background_color)
        # hide auto range button
        self.plot_item.hideButtons()

        # remove x, y axis
        self.plot_item.hideAxis('bottom')
        self.plot_item.hideAxis('left')

        self.raw_item = pg.PlotDataItem(pen='w')
        self.raw_item.setVisible(self.visible)
        self.addItem(self.raw_item)

        self.thr = 0.0
        self.thr_item = pg.InfiniteLine(
            pos=self.thr, angle=0, pen="g")
        self.thr_item.setVisible(self.show_thr)
        self.addItem(self.thr_item)

        self.spikes_item_list = []

        self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def update_plot(self):
        if self.visible:
            self.draw_raw()
        self.raw_item.setVisible(self.visible)

        if self.has_thr and self.show_thr and self.visible:
            self.draw_thr()
        self.thr_item.setVisible(self.show_thr and self.visible)

        if self.has_spikes and self.show_spikes and self.visible:
            self.draw_spikes()
        [spikes_item.setVisible(self.show_spikes and self.visible)
         for spikes_item in self.spikes_item_list]

    def draw_raw(self):
        # self.raw_item.clear()
        lastpoint = self.offset + int(self.num_data_show)
        # x = np.arange(int(self.num_data_show))
        # y = self.raw[:int(self.num_data_show)]
        self.plot_item.getViewBox().setXRange(self.offset, lastpoint, padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)
        # self.raw_item.setData(x=x, y=y)

    def draw_thr(self):
        self.thr_item.setValue(self.thr)

    def draw_events(self):
        """TODO"""
        pass

    def draw_spikes(self):
        if len(self.spikes_item_list) != 0:
            [self.removeItem(item) for item in self.spikes_item_list]
        self.spikes_item_list = self.ts_to_lines(self.spikes["timestamps"],
                                                 self.spikes["unitID"], "spikes")

        [self.addItem(item) for item in self.spikes_item_list]

    def ts_to_lines(self, ts, color_ids, data_type):
        item_list = []
        lastpoint = self.offset + int(self.num_data_show)
        # ts_mask = np.all([ts >= self.offset, ts < lastpoint], axis=0)
        self.pan_color_dict = dict()

        y_element = np.array([-self.data_scale, self.thr])
        connect_element = np.array([1, 0])
        unique_unit = np.unique(color_ids)

        if data_type == "spikes":
            for color_id in unique_unit:
                pen = pg.mkPen(
                    color=[int(c * 255) for c in self.color_palette[int(color_id)]])
                data_filtered = ts[color_ids == color_id]
                n = data_filtered.shape[0]
                x = np.repeat(data_filtered, 2)
                y = np.tile(y_element, n)
                connect = np.tile(connect_element, n)
                item_list.append(pg.PlotCurveItem(
                    x=x, y=y, pen=pen, connect=connect))

        return item_list

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        modifiers = QGuiApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ShiftModifier:
            """scale x axis."""
            delta = int(event.delta() / 120)
            current_range = self.plot_item.getViewBox().state['viewRange']
            new_num_data_show = int(self.num_data_show / (1 + delta / 10))
            self.num_data_show = np.min(
                (np.max((new_num_data_show, self.MIN_DATA_SHOW)),
                 self.MAX_DATA_SHOW))
            new_range = [current_range[0][0],
                         current_range[0][0] + self.num_data_show]
            # check boundary
            if new_range[0] < 0:
                new_range = [0, self.num_data_show]
            if new_range[1] > self.raw_len:
                new_range = [self.raw_len - self.num_data_show, self.raw_len]

            self.plot_item.getViewBox().setXRange(*new_range, padding=0)

        elif (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
            """scale y axis."""
            delta = int(event.delta() / 120)
            current_range = self.plot_item.getViewBox().state['viewRange']
            self.data_scale = int(self.data_scale / (1 + delta / 10))
            new_range = [-self.data_scale, self.data_scale]

            self.plot_item.getViewBox().setYRange(*new_range, padding=0)

        else:
            """scroll the range."""
            delta = int(event.delta() / 120)
            current_range = self.plot_item.getViewBox().state['viewRange']
            new_range = [current_range[0][0] - int(delta * self.num_data_show / 10),
                         current_range[0][1] - int(delta * self.num_data_show / 10)]
            # check boundary
            if new_range[0] < 0:
                new_range = [0, self.num_data_show]
            if new_range[1] > self.raw_len:
                new_range = [self.raw_len - self.num_data_show, self.raw_len]

            self.plot_item.getViewBox().setXRange(*new_range, padding=0)

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        pass

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
        pass

    def graphMouseReleaseEvent(self, event):
        """Overwrite PlotItem.scene().mouseReleaseEvent."""
        pass
