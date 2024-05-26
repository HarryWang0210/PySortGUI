import logging

import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QGuiApplication
from PyQt5.QtWidgets import QApplication

from DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from UI.TimelineView_ui import Ui_TimelineView

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

    def continuous_data_changed(self, new_raw_object, new_filted_object):
        # print(new_raw_object, new_filted_object)
        self.raw_pushButton.setChecked(True)
        self.graphWidget.continuous_data_changed(
            new_raw_object, new_filted_object)
        self.raw_pushButton.setChecked(new_filted_object is None)

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.graphWidget.showing_spike_data_changed(new_spike_object)

    def showing_units_changed(self, showing_unit_IDs):
        self.graphWidget.showing_units_changed(showing_unit_IDs)

    def event_data_changed(self, new_event_object: DiscreteData | None):
        self.graphWidget.event_data_changed(new_event_object)

    def showing_events_changed(self, showing_event_IDs):
        self.graphWidget.showing_events_changed(showing_event_IDs)
    # def spike_chan_changed(self, current_chan_info):
    #     self.raw_pushButton.setChecked(current_chan_info['Type'] == 'Raws')
    #     self.graphWidget.spike_chan_changed(current_chan_info)

    # def filted_data_changed(self, filted_data):
    #     self.graphWidget.filted_data_changed(filted_data)

    # def showing_spikes_data_changed(self, spikes_data):
    #     self.graphWidget.showing_spikes_data_changed(spikes_data)

    # def extract_wav_changed(self, wav_dict):
    #     self.graphWidget.extract_wav_changed(wav_dict)

    # def sorting_result_changed(self, unitID):
    #     self.graphWidget.sorting_result_changed(unitID)


class TimelineViewGraph(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.MIN_DATA_SHOW = 100
        self.MAX_DATA_SHOW = 300000
        self.DEFAULT_DATA_SHOW = 30000

        self.data_object = None
        self.current_raw_object: ContinuousData | None = None
        self.current_filted_object: ContinuousData | None = None
        self.current_spike_object: DiscreteData | None = None
        self.current_event_object: DiscreteData | None = None
        # self.current_chan_info = None
        self.visible = False  # overall visible
        self.color_palette_list = sns.color_palette(
            'bright', 64)  # palette for events and spikes

        # threshold relative variables
        self.thr = 0.0
        self.has_thr = False
        self.show_thr = False

        # events relative variables
        self.events = None
        self.has_events = False
        self.show_events = False
        self.num_event_units = 0
        self.event_units_visible = []  # list of all event units

        # spikes relative variables
        # self.spikes = None
        self.has_spikes = False
        self.show_spikes = False
        # self.num_spike_units = 0
        # self.spike_units_visible = []  # list of all spike units

        # raw relative variables
        # self.raw_data = None
        # self.filted_data = None
        # self.has_filted_data = False

        self.show_raw = True

        # self.data_len = 0
        # self.data_scale = 1000  # maximun height of data
        # initial number of data points show in window
        self.num_data_show = self.DEFAULT_DATA_SHOW
        self._x_boundary: tuple[int, int] = (0, 0)
        self._x_range: tuple[int, int] = (0, self.num_data_show)
        # self._y_boundary: tuple[int, int] = (0, 0)
        self._y_range: tuple[int, int] = (-1000, 1000)

        # self.current_wav_colors = []  # (units, 3)
        # self.current_wav_units = []
        self.current_showing_units = []
        self.current_showing_events: list = []

        self.initPlotItem()

    def initPlotItem(self):
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

        self.data_item = pg.PlotDataItem(pen='w')
        self.data_item.setVisible(False)
        self.addItem(self.data_item)

        self.thr_item = pg.InfiniteLine(pos=self.thr, angle=0, pen="g")
        self.thr_item.setVisible(False)
        self.addItem(self.thr_item)

        self.spikes_item_list = []
        self.events_item_list = []

        self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.updatePlot()

    def continuous_data_changed(self, new_raw_object, new_filted_object):
        self.current_raw_object: ContinuousData | None = new_raw_object
        self.current_filted_object: ContinuousData | None = new_filted_object

        self.visible = True
        if self.current_raw_object is None:
            self.visible = False
            self.updatePlot()
            return
        data = self.current_raw_object.data
        if self.current_raw_object.timestamps is None:
            self._x_boundary = (0, len(data))

            # self.data_len = len(data)
        else:
            self._x_boundary = (self.current_raw_object.timestamps[0],
                                int(self.current_raw_object.timestamps[-1]) + 1)
            # self.data_len = int(self.current_raw_object.timestamps[-1]) + 1

        if self.current_filted_object is None:
            data_scale = np.max(np.abs(data)) / 2
        else:
            data = self.current_filted_object.data
            data_scale = np.max(np.abs(data)) / 2
            self.thr = self.current_filted_object.threshold

        # self.data_len = len(data)
        # self.data_scale = np.max(np.abs(data)) / 2
        # initial number of data points show in window
        self.num_data_show = self.DEFAULT_DATA_SHOW
        self._x_range = (0, self.num_data_show)
        self._y_range = (-data_scale, data_scale)

        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object
        self.current_showing_units = []
        # self.has_spikes = not self.current_spike_object is None
        self.updatePlot()

    def showing_units_changed(self, showing_unit_IDs):
        self.current_showing_units = showing_unit_IDs
        self.updatePlot()

    def event_data_changed(self, new_event_object: DiscreteData | None):
        if new_event_object is self.current_event_object:
            return
        self.current_event_object = new_event_object
        self.current_showing_events = []
        self.updatePlot()

    def showing_events_changed(self, showing_event_IDs):
        self.current_showing_events = showing_event_IDs
        self.updatePlot()
        # self.graphWidget.showing_events_changed(showing_event_IDs)

    # def spike_chan_changed(self, current_chan_info):
    #     self.current_chan_info = current_chan_info
    #     logger.debug(self.current_chan_info['Type'])
    #     if self.current_chan_info['Type'] == 'Spikes':
    #         self.visible = True
    #         self.has_thr = True
    #         self.has_spikes = True

    #         self.getRaw(self.data_object.getRaw(self.current_chan_info['ID']))
    #         self.thr = self.current_chan_info["Threshold"]

    #         # self.spikes = self.data_object.getSpikes(self.current_chan_info['ID'],
    #         #                                          self.current_chan_info['Label'])
    #         # logger.debug(self.current_chan_info)
    #         self.num_spike_units = self.current_chan_info["unitInfo"].shape[0]

    #     elif self.current_chan_info['Type'] == 'Raws':
    #         self.visible = True
    #         self.has_thr = False
    #         self.has_spikes = False

    #         self.getRaw(self.data_object.getRaw(self.current_chan_info['ID']))
    #         self.thr = 0.0
    #         # self.spikes = None
    #         self.num_spike_units = 0

    #     elif self.current_chan_info['Type'] == 'Filted':
    #         self.visible = True
    #         self.has_thr = True
    #         self.has_spikes = False

    #         self.getRaw(self.data_object.getRaw(self.current_chan_info['ID']))
    #         self.thr = self.current_chan_info["Threshold"]
    #         # self.spikes = None
    #         self.num_spike_units = 0

    #     elif self.current_chan_info['Type'] == 'Events':
    #         self.visible = False
    #         logger.critical('Not implement error.')

    #     self.spike_units_visible = [True] * self.num_spike_units

    # def filted_data_changed(self, filted_data):
    #     self.filted_data = filted_data
    #     if isinstance(self.filted_data, np.ndarray):
    #         self.has_filted_data = True
    #         self.data_scale = np.max(np.abs(self.filted_data)) / 2

    #     else:
    #         self.has_filted_data = False
    #     logger.debug('filted_data_changed')
    #     self.updatePlot()

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

    # def showing_spikes_data_changed(self, spikes_data):
    #     if self.has_spikes:
    #         self.current_wav_units = spikes_data['current_wav_units']
    #         self.current_showing_units = spikes_data['current_showing_units']
    #         # self.current_wavs_mask = np.isin(spikes_data['current_wav_units'],
    #         #                                  spikes_data['current_showing_units'])
    #         # self.num_unit = len(np.unique(self.current_wav_units))
    #     logger.debug('showing_spikes_data_changed')
    #     # self.current_wav_colors = self.getColor(self.current_wav_units)
    #     self.updatePlot()

    # def getRaw(self, raw):
    #     self.raw_data = raw
    #     self.data_len = len(raw)
    #     self.data_scale = np.max(np.abs(self.raw_data)) / 2
    #     self.num_data_show = 1000  # initial number of data points show in window

    # def getEvents(self, events):
    #     self.events = events
    #     self.has_events = True

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
        self.show_thr = show
        logger.debug(f'showThreshold: {show}')
        self.updatePlot()

    def showEvents(self, show):
        """Control from TimelineView."""
        self.show_events = show
        logger.debug(f'showEvents: {show}')
        self.updatePlot()

    def showSpikes(self, show):
        """Control from TimelineView."""
        self.show_spikes = show
        logger.debug(f'showSpikes: {show}')
        self.updatePlot()

    def showRaw(self, show):
        """Control from TimelineView."""
        self.show_raw = show
        logger.debug(f'showRaw: {show}')
        self.updatePlot()

    def updatePlot(self):
        # logger.debug('updatePlot')
        # logger.debug(self.visible)
        if self.visible:
            if self.show_raw:
                # logger.debug('draw raw')
                self.drawData('raw')
            else:
                # logger.debug('draw filted')
                self.drawData('filted')

            # if not self.show_raw and self.has_filted_data:
            #     self.drawData(self.filted_data)
            # else:
            #     self.drawData(self.raw_data)

            if self.show_thr and not self.current_filted_object is None:
                # logger.debug('draw thr')
                self.drawThreshold()

            if self.show_spikes and not self.current_spike_object is None:
                # logger.debug('draw spike')
                self.drawSpikes()

            if self.show_events and not self.current_event_object is None:
                # logger.debug('draw event')
                self.drawEvents()

        self.data_item.setVisible(self.visible)
        self.thr_item.setVisible(self.visible and
                                 self.show_thr and
                                 not self.current_filted_object is None)

        for item in self.spikes_item_list:
            item.setVisible(self.visible and
                            self.show_spikes and
                            not self.current_spike_object is None)

    def drawData(self, data_type):
        if data_type == 'raw':
            data = self.current_raw_object.data
        elif data_type == 'filted':
            data = self.current_filted_object.data

        if self.current_raw_object.timestamps is None:
            # generate x
            x = np.arange(start=self._x_range[0], stop=self._x_range[1]+1)
            data = data[self._x_range[0]: self._x_range[1] + 1]
            connect = 'auto'
        else:
            timestamps = self.current_raw_object.timestamps
            mask = (timestamps >= self._x_range[0]) & \
                (timestamps <= self._x_range[1])

            x = timestamps[mask]
            data = data[mask]
            connect = np.append(np.diff(x) <= 1, 0)

        # logger.debug(np.any(np.diff(x) < 0))
        # logger.debug(x)
        # logger.debug(connect)

        self.data_item.setData(x=x, y=data, connect=connect)
        self.plot_item.getViewBox().setXRange(*self._x_range, padding=0)
        self.plot_item.getViewBox().setYRange(*self._y_range, padding=0)

    def drawThreshold(self):
        if self.current_filted_object is None:
            return

        self.thr_item.setValue(self.current_filted_object.threshold)

    def drawEvents(self):
        self.removeItems(self.events_item_list)
        self.events_item_list = []

        current_showing_units = np.unique(self.current_event_object.unit_IDs)
        if self.current_showing_events == []:
            return

        self.events_item_list = self.tsToLines(self.current_event_object.timestamps,
                                               unit_IDs=self.current_event_object.unit_IDs,
                                               showing_unit_IDs=self.current_showing_events,
                                               data_type="events")

        [self.addItem(item) for item in self.events_item_list]

    def drawSpikes(self):
        self.removeItems(self.spikes_item_list)
        self.spikes_item_list = []

        if self.current_showing_units == []:
            return

        self.spikes_item_list = self.tsToLines(self.current_spike_object.timestamps,
                                               unit_IDs=self.current_spike_object.unit_IDs,
                                               showing_unit_IDs=self.current_showing_units,
                                               data_type="spikes")

        [self.addItem(item) for item in self.spikes_item_list]

    def removeItems(self, item_list):
        for item in item_list:
            self.removeItem(item)
        self.spikes_item_list = []

    def tsToLines(self, ts, showing_unit_IDs, unit_IDs, data_type):
        """_summary_

        Args:
            ts (_type_): _description_
            showing_unit_IDs (_type_): unit to show. e.g. [0,1,2,3]
            unit_IDs (_type_): all unit id array. e.g. [0,0,1,1,2]
            data_type (_type_): 'events' or 'spikes'

        Returns:
            _type_: _description_
        """
        item_list = []
        if data_type == "spikes":
            y_element = np.array([self._y_range[0], self.thr])
            unit_color_map = dict(zip(self.current_spike_object.unit_header['ID'], np.arange(
                self.current_spike_object.unit_header.shape[0], dtype=int)))

        elif data_type == "events":
            y_element = np.array([self._y_range[1], self.thr])
            unit_color_map = dict(zip(self.current_event_object.unit_header['ID'], np.arange(
                self.current_event_object.unit_header.shape[0], dtype=int)))
        else:
            print('Unknown type of timestamps.')
            return

        for ID in showing_unit_IDs:
            data_filtered = ts[unit_IDs == ID]

            color = self.color_palette_list[unit_color_map[int(ID)]]
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
            # current_range = self.plot_item.getViewBox().state['viewRange']
            new_num_data_show = int(self.num_data_show / (1 + delta / 10))
            self.num_data_show = min((max((new_num_data_show, self.MIN_DATA_SHOW)),
                                      self.MAX_DATA_SHOW))

            new_range = (self._x_range[0],
                         self._x_range[0] + self.num_data_show)
            # check boundary
            if new_range[0] < self._x_boundary[0]:
                new_range = (0, self.num_data_show)
            if new_range[1] > self._x_boundary[1]:
                new_range = (self._x_boundary[1] - self.num_data_show,
                             self._x_boundary[1])
            self._x_range = new_range

            self.updatePlot()

            # self.plot_item.getViewBox().setXRange(*new_range, padding=0)

        elif (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
            """scale y axis."""
            delta = int(event.delta() / 120)
            # current_range = self.plot_item.getViewBox().state['viewRange']
            data_scale = int(self._y_range[1] / (1 + delta / 10))
            self._y_range = (-data_scale, data_scale)
            self.updatePlot()
            # new_range = [-self.data_scale, self.data_scale]

            # self.plot_item.getViewBox().setYRange(*new_range, padding=0)

        else:
            """scroll the range."""
            delta = int(event.delta() / 120)
            # current_range = self.plot_item.getViewBox().state['viewRange']
            new_range = (self._x_range[0] - int(delta * self.num_data_show / 10),
                         self._x_range[1] - int(delta * self.num_data_show / 10))
            # check boundary
            if new_range[0] < self._x_boundary[0]:
                new_range = (0, self.num_data_show)
            if new_range[1] > self._x_boundary[1]:
                new_range = (self._x_boundary[1] - self.num_data_show,
                             self._x_boundary[1])
            self._x_range = new_range
            self.updatePlot()

            # self.plot_item.getViewBox().setXRange(*new_range, padding=0)

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        pass

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
        pass

    def graphMouseReleaseEvent(self, event):
        """Overwrite PlotItem.scene().mouseReleaseEvent."""
        pass
