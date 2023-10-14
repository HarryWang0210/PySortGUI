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
    signal_selected_units_changed = QtCore.pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Timeline View"
        self.setupUi(self)
        self.graphWidget = TimelineViewGraph(self)
        self.plotLayout.addWidget(self.graphWidget)
        self.setup_connections()

    def setup_connections(self):
        self.thr_pushButton.clicked.connect(self.graphWidget.show_thr)
        self.events_pushButton.clicked.connect(self.graphWidget.show_events)
        self.spikes_pushButton.clicked.connect(self.graphWidget.show_spikes)
        self.raw_pushButton.clicked.connect(self.graphWidget.show_raw)

        self.signal_data_file_name_changed.connect(
            self.graphWidget.data_file_name_changed)
        self.signal_spike_chan_changed.connect(
            self.graphWidget.spike_chan_changed)
        self.signal_selected_units_changed.connect(
            self.graphWidget.selected_units_changed)

    def data_file_name_changed(self, data):
        self.signal_data_file_name_changed.emit(data)

    def spike_chan_changed(self, meta_data):
        self.signal_spike_chan_changed.emit(meta_data)

    def selected_units_changed(self, selected_rows):
        self.signal_selected_units_changed.emit(selected_rows)


class TimelineViewGraph(pg.PlotWidget):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_selected_units_changed = QtCore.pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.MIN_DATA_SHOW = 100
        self.MAX_DATA_SHOW = 30000

        self.data_object = None
        self.visible = False  # overall visible
        self.color_palette = sns.color_palette(
            None, 64)  # palette for events and spikes

        # threshold relative variables
        self.thr = None
        self.has_thr = False
        self.thr_visible = False
        self.redraw_thr = True

        # events relative variables
        self.events = None
        self.has_events = False
        self.events_visible = False
        self.num_event_units = 0
        self.event_units_visible = []  # list of all event units
        self.redraw_events = True

        # spikes relative variables
        self.spikes = None
        self.has_spikes = False
        self.spikes_visible = False
        self.num_spike_units = 0
        self.spike_units_visible = []  # list of all spike units
        self.redraw_spikes = True

        # raw relative variables
        self.raw_visible = False
        self.raw = None
        self.raw_len = 0
        self.redraw_raw = True
        self.data_scale = 1000  # maximun height of data
        self.num_data_show = 1000  # initial number of data points show in window

        self.init_plotItem()

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
        self.raw_item.setVisible(False)
        self.addItem(self.raw_item)

        self.thr = 0.0
        self.thr_item = pg.InfiniteLine(pos=self.thr, angle=0, pen="g")
        self.thr_item.setVisible(False)
        self.addItem(self.thr_item)

        self.spikes_item_list = []

        self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.update_plot()

    def spike_chan_changed(self, meta_data):
        self.get_raw(self.data_object.get_raw(int(meta_data["ID"])))
        self.get_thr(meta_data["Threshold"])
        self.get_spikes(
            self.data_object.get_spikes(int(meta_data["ID"]), meta_data["Label"]))
        self.visible = True

        self.update_plot()

    def selected_units_changed(self, selected_rows):
        self.spike_units_visible = [False] * self.num_spike_units
        for i in selected_rows:
            self.spike_units_visible[i] = True
        self.redraw_spikes = False
        self.update_plot()

    def get_raw(self, raw):
        self.raw = raw
        self.raw_len = len(raw)
        self.data_scale = np.max(np.abs(self.raw)) / 2
        self.num_data_show = 1000  # initial number of data points show in window
        self.redraw_raw = True

    def get_thr(self, thr):
        try:
            self.thr = float(thr)
            self.has_thr = True
        except:
            self.thr = 0.0
            self.has_thr = False
        self.redraw_thr = True

    def get_events(self, events):
        # TODO: get_events
        self.events = events
        self.has_events = True
        self.redraw_events = True

    def get_spikes(self, spikes):
        if spikes["unitInfo"] is None:
            self.has_spikes = False
            self.spikes = None
            self.num_spike_units = 0

        else:
            self.has_spikes = True
            self.spikes = spikes
            self.num_spike_units = spikes["unitInfo"].shape[0]
        self.spike_units_visible = [True] * self.num_spike_units
        self.redraw_spikes = True

    def show_thr(self, show):
        """Control from TimelineView."""
        self.thr_visible = show
        self.redraw_thr = False
        self.update_plot()

    def show_events(self, show):
        """Control from TimelineView."""
        self.events_visible = show
        self.redraw_events = False
        self.update_plot()

    def show_spikes(self, show):
        """Control from TimelineView."""
        self.spikes_visible = show
        self.redraw_spikes = False
        self.update_plot()

    def show_raw(self, show):
        """Control from TimelineView."""
        self.raw_visible = show

    def update_plot(self):
        # FIXME: 每次按button都會使顯示範圍重置
        if self.visible:
            if self.redraw_raw:
                self.draw_raw()
                self.redraw_raw = False

            if self.has_thr and self.redraw_thr:
                self.draw_thr()
                self.redraw_thr = False

            if self.has_spikes and self.redraw_spikes:
                self.draw_spikes()
                self.redraw_spikes = False

        self.raw_item.setVisible(self.visible)
        self.thr_item.setVisible(self.visible and
                                 self.has_thr and
                                 self.thr_visible)

        for unit_ID in range(len(self.spikes_item_list)):
            spikes_item = self.spikes_item_list[unit_ID]
            spikes_item.setVisible(self.visible and
                                   self.spikes_visible and
                                   self.spike_units_visible[unit_ID])

    def draw_raw(self):
        self.raw_item.setData(self.raw)
        self.plot_item.getViewBox().setXRange(0, self.num_data_show, padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

    def draw_thr(self):
        self.thr_item.setValue(self.thr)

    def draw_events(self):
        """TODO"""
        pass

    def draw_spikes(self):
        if len(self.spikes_item_list) != 0:
            [self.removeItem(item) for item in self.spikes_item_list]
        self.spikes_item_list = self.ts_to_lines(self.spikes["timestamps"],
                                                 self.spikes["unitID"],
                                                 self.num_spike_units,
                                                 "spikes")

        [self.addItem(item) for item in self.spikes_item_list]

    def ts_to_lines(self, ts, color_ID, num_color, data_type):
        item_list = []
        if data_type == "spikes":
            y_element = np.array([-self.data_scale, self.thr])
        elif data_type == "events":
            y_element = np.array([self.data_scale, self.thr])
        else:
            return

        for ID in range(num_color):
            pen = pg.mkPen(
                color=[int(c * 255) for c in self.color_palette[ID]])
            data_filtered = ts[color_ID == ID]
            n = data_filtered.shape[0]
            x = np.repeat(data_filtered, 2)
            y = np.tile(y_element, n)
            item_list.append(pg.PlotCurveItem(
                x=x, y=y, pen=pen, connect="pairs"))

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
