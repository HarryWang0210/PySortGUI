from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QGuiApplication

import pyqtgraph as pg
import numpy as np
import seaborn as sns
from UI.TimelineView_ui import Ui_TimelineView

import logging
logger = logging.getLogger(__name__)


class TimelineView(QtWidgets.QWidget, Ui_TimelineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Timeline View"
        self.setupUi(self)
        self.graphWidget = TimelineViewGraph(self)
        self.plotLayout.addWidget(self.graphWidget)
        self.setupConnections()

    def setupConnections(self):
        self.thr_pushButton.clicked.connect(self.graphWidget.showThreshold)
        self.events_pushButton.clicked.connect(self.graphWidget.showEvents)
        self.spikes_pushButton.clicked.connect(self.graphWidget.showSpikes)
        self.raw_pushButton.toggled.connect(self.graphWidget.showRaw)

    def data_file_name_changed(self, data):
        self.graphWidget.data_file_name_changed(data)

    def spike_chan_changed(self, current_chan_info):
        self.raw_pushButton.setChecked(current_chan_info['Type'] == 'Raws')
        self.graphWidget.spike_chan_changed(current_chan_info)

    def filted_data_changed(self, filted_data):
        self.graphWidget.filted_data_changed(filted_data)

    def showing_spikes_data_changed(self, spikes_data):
        self.graphWidget.showing_spikes_data_changed(spikes_data)


class TimelineViewGraph(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.MIN_DATA_SHOW = 100
        self.MAX_DATA_SHOW = 30000

        self.data_object = None
        self.current_chan_info = None
        self.visible = False  # overall visible
        self.color_palette_list = sns.color_palette(
            None, 64)  # palette for events and spikes

        # threshold relative variables
        self.thr = 0.0
        self.has_thr = False
        self.thr_visible = False

        # events relative variables
        self.events = None
        self.has_events = False
        self.events_visible = False
        self.num_event_units = 0
        self.event_units_visible = []  # list of all event units

        # spikes relative variables
        self.spikes = None
        self.has_spikes = False
        self.spikes_visible = False
        self.num_spike_units = 0
        self.spike_units_visible = []  # list of all spike units

        # raw relative variables
        self.raw_data = None
        self.filted_data = None
        self.has_filted_data = False

        self.raw_data_visible = False

        self.data_len = 0
        self.data_scale = 1000  # maximun height of data
        self.num_data_show = 1000  # initial number of data points show in window

        self.current_wav_colors = []  # (units, 3)

        self.initPlotItem()

    def initPlotItem(self):
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

        self.data_item = pg.PlotDataItem(pen='w')
        self.data_item.setVisible(False)
        self.addItem(self.data_item)

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
        self.updatePlot()

    def spike_chan_changed(self, current_chan_info):
        self.current_chan_info = current_chan_info

        if self.current_chan_info['Type'] == 'Spikes':
            self.visible = True
            self.has_thr = True
            self.has_spikes = True

            self.getRaw(self.data_object.getRaw(self.current_chan_info['ID']))
            self.thr = self.current_chan_info["Threshold"]

            self.spikes = self.data_object.getSpikes(self.current_chan_info['ID'],
                                                     self.current_chan_info['Label'])
            self.num_spike_units = self.spikes["unitInfo"].shape[0]

        elif self.current_chan_info['Type'] == 'Raws':
            self.visible = True
            self.has_thr = False
            self.has_spikes = False

            self.getRaw(self.data_object.getRaw(self.current_chan_info['ID']))
            self.thr = 0.0
            self.spikes = None
            self.num_spike_units = 0

        elif self.current_chan_info['Type'] == 'Events':
            self.visible = False
            logger.critical('Not implement error.')

        self.spike_units_visible = [True] * self.num_spike_units

    def filted_data_changed(self, filted_data):
        self.filted_data = filted_data
        if isinstance(self.filted_data, np.ndarray):
            self.has_filted_data = True
            self.data_scale = np.max(np.abs(self.filted_data)) / 2

        else:
            self.has_filted_data = False
        logger.debug('filted_data_changed')
        self.updatePlot()

    def showing_spikes_data_changed(self, spikes_data):
        self.current_wav_units = spikes_data['current_wav_units']
        self.current_showing_units = spikes_data['current_showing_units']
        self.current_wavs_mask = np.isin(spikes_data['current_wav_units'],
                                         spikes_data['current_showing_units'])
        self.num_unit = len(np.unique(self.current_wav_units))
        logger.debug('showing_spikes_data_changed')
        # self.current_wav_colors = self.getColor(self.current_wav_units)
        self.updatePlot()

    def getRaw(self, raw):
        self.raw_data = raw
        self.data_len = len(raw)
        self.data_scale = np.max(np.abs(self.raw_data)) / 2
        self.num_data_show = 1000  # initial number of data points show in window

    def getEvents(self, events):
        # TODO: getEvents
        self.events = events
        self.has_events = True

    # def getSpikes(self, spikes):
    #     if spikes["unitInfo"] is None:
    #         self.visible = False

    #         self.has_spikes = False
    #         self.spikes = None
    #         self.num_spike_units = 0

    #     else:
    #         self.visible = True

    #         self.has_spikes = True
    #         self.spikes = spikes
    #         self.num_spike_units = spikes["unitInfo"].shape[0]
    #     self.spike_units_visible = [True] * self.num_spike_units

    # def getColor(self, unit_data):
    #     """_summary_

    #     Args:
    #         unit_data (list): list of all unit ID (int).

    #     Returns:
    #         list: color palette list.
    #     """
    #     n = len(unit_data)
    #     color = np.zeros((n, 3))

    #     for i in range(n):
    #         color[i, :] = self.color_palette_list[int(unit_data[i])]
    #     color = color * 255
    #     return color.astype(np.int32)

    def showThreshold(self, show):
        """Control from TimelineView."""
        self.thr_visible = show
        self.updatePlot()

    def showEvents(self, show):
        """Control from TimelineView."""
        self.events_visible = show
        self.updatePlot()

    def showSpikes(self, show):
        """Control from TimelineView."""
        self.spikes_visible = show
        self.updatePlot()

    def showRaw(self, show):
        """Control from TimelineView."""
        self.raw_data_visible = show
        logger.debug(f'showRaw: {show}')
        self.updatePlot()

    def updatePlot(self):
        if self.visible:
            if not self.raw_data_visible and self.has_filted_data:
                self.drawData(self.filted_data)
            else:
                self.drawData(self.raw_data)

            if self.has_thr:
                self.drawThreshold()

            if self.has_spikes:
                self.drawSpikes()

        self.data_item.setVisible(self.visible)
        self.thr_item.setVisible(self.visible and
                                 self.has_thr and
                                 self.thr_visible)

        for item in self.spikes_item_list:
            item.setVisible(self.visible and
                            self.has_spikes and
                            self.spikes_visible)

    def drawData(self, data):
        self.data_item.setData(data)
        self.plot_item.getViewBox().setXRange(0, self.num_data_show, padding=0)
        self.plot_item.getViewBox().setYRange(-self.data_scale, self.data_scale, padding=0)

    def drawThreshold(self):
        self.thr_item.setValue(self.thr)

    def drawEvents(self):
        """TODO"""
        pass

    def drawSpikes(self):
        self.removeSpikeItems()

        self.spikes_item_list = self.tsToLines(self.spikes["timestamps"],
                                               unit_ID=self.current_showing_units,
                                               all_unit_ID=self.current_wav_units,
                                               data_type="spikes")

        [self.addItem(item) for item in self.spikes_item_list]

    def removeSpikeItems(self):
        for item in self.spikes_item_list:
            self.removeItem(item)
        self.spikes_item_list = []

    def tsToLines(self, ts, unit_ID, all_unit_ID, data_type):
        # FIXME: y軸縮小時上下界不會跟著改變
        item_list = []
        if data_type == "spikes":
            y_element = np.array([-self.data_scale, self.thr])
        elif data_type == "events":
            y_element = np.array([self.data_scale, self.thr])
        else:
            print('Unknown type of timestamps.')
            return

        for ID in unit_ID:
            data_filtered = ts[all_unit_ID == ID].copy()

            color = self.color_palette_list[ID]
            color = (np.array(color) * 255).astype(int)

            n = data_filtered.shape[0]
            x = np.repeat(data_filtered, 2)
            y = np.tile(y_element, n)
            item_list.append(pg.PlotCurveItem(
                x=x, y=y, pen=pg.mkPen(color=color), connect="pairs"))

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
            if new_range[1] > self.data_len:
                new_range = [self.data_len -
                             self.num_data_show, self.data_len]

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
            if new_range[1] > self.data_len:
                new_range = [self.data_len -
                             self.num_data_show, self.data_len]

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
