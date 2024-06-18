import logging

import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QGuiApplication
from PyQt5.QtWidgets import QApplication

from pysortgui.DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from pysortgui.UI.TimelineView_ui import Ui_TimelineView
from pysortgui.Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class TimelineView(WidgetsInterface, Ui_TimelineView):
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

    def widgetVisibilityChanged(self, visible: bool):
        self.graphWidget.widgetVisibilityChanged(visible)

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

    def background_continuous_data_changed(self, new_bg_object, color, show_on_top):
        self.graphWidget.background_continuous_data_changed(new_bg_object,
                                                            color,
                                                            show_on_top)

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
        self.MAX_DATA_SHOW = np.inf
        self.DEFAULT_DATA_SHOW = 30000
        self.do_downsampling = True
        self.DOWNSAMPLE_SIZE = self.DEFAULT_DATA_SHOW * 5

        self.data_object = None
        self.current_raw_object: ContinuousData | None = None
        self.current_filted_object: ContinuousData | None = None
        self.current_bg_object: ContinuousData | None = None
        self.current_spike_object: DiscreteData | None = None
        self.current_event_object: DiscreteData | None = None
        # self.current_chan_info = None
        self.plot_visible = False  # determine when to show plot
        self.widget_visible = False
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
        self.show_bg = False
        self.bg_color = None

        self.redraw_data = False
        self.redraw_bg = False
        self.redraw_thr = False
        self.redraw_spikes = False
        self.redraw_events = False

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

        self.bg_data_item = pg.PlotDataItem(pen='w')
        self.bg_data_item.setZValue(-1)
        self.bg_data_item.setVisible(False)
        self.addItem(self.bg_data_item)

        self.thr_item = pg.InfiniteLine(pos=self.thr, angle=0, pen="g")
        self.thr_item.setVisible(False)
        self.addItem(self.thr_item)

        self.spikes_item_list = []
        self.events_item_list = []

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
        self.current_bg_object = None
        self.current_event_object = None
        self.updatePlot()

    def continuous_data_changed(self, new_raw_object, new_filted_object):
        self.current_raw_object: ContinuousData | None = new_raw_object
        self.current_filted_object: ContinuousData | None = new_filted_object
        self.current_spike_object = None

        self.plot_visible = True
        if self.current_raw_object is None:
            self.plot_visible = False
            self.updatePlot()
            return

        self.redraw_data = True
        data = self.current_raw_object._data
        if self.current_raw_object._timestamps is None:
            self._x_boundary = (0, len(data))
        else:
            self._x_boundary = (self.current_raw_object._timestamps[0],
                                int(self.current_raw_object._timestamps[-1]) + 1)

        if self.current_filted_object is None:
            data_scale = np.max(np.abs(data)) / 2
        else:
            data = self.current_filted_object._data
            data_scale = np.max(np.abs(data)) / 2
            self.thr = self.current_filted_object.threshold
            self.redraw_thr = True

        self.num_data_show = self.DEFAULT_DATA_SHOW
        self._x_range = (0, self.num_data_show)
        self._y_range = (-data_scale, data_scale)

        self.redraw_bg = True
        self.redraw_spikes = True
        self.redraw_events = True
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object
        self.current_showing_units = []
        # self.updatePlot()

    def showing_units_changed(self, showing_unit_IDs):
        self.redraw_spikes = True
        self.current_showing_units = showing_unit_IDs
        self.updatePlot()

    def event_data_changed(self, new_event_object: DiscreteData | None):
        if new_event_object is self.current_event_object:
            return
        # self.redraw_events = True
        self.current_event_object = new_event_object
        self.current_showing_events = []
        # self.updatePlot()

    def showing_events_changed(self, showing_event_IDs):
        self.redraw_events = True
        self.current_showing_events = showing_event_IDs
        self.updatePlot()

    def background_continuous_data_changed(self, new_bg_object, color, show_on_top):
        # if new_bg_object is self.current_bg_object:
        #     return
        self.redraw_bg = True
        self.current_bg_object = new_bg_object

        self.show_bg = True
        self.bg_color = color
        if self.current_bg_object is None:
            self.show_bg = False

        if show_on_top:
            self.bg_data_item.setZValue(1)
        else:
            self.bg_data_item.setZValue(-1)
        self.updatePlot()

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
        self.redraw_data = True
        logger.debug(f'showRaw: {show}')
        self.updatePlot()

    def updatePlot(self):
        # logger.debug('updatePlot')
        # logger.debug(self.plot_visible)
        visible = self.plot_visible and self.widget_visible

        if visible:
            import time
            if self.redraw_data:
                start = time.time()
                if self.show_raw:
                    self.drawData('raw')
                else:
                    self.drawData('filted')
                self.redraw_data = False
                logger.debug(f'redraw data: {time.time() - start}')

            if self.show_bg and self.redraw_bg:
                start = time.time()
                self.drawBackgroundData()
                self.redraw_bg = False
                logger.debug(f'redraw bg data: {time.time() - start}')

            if self.show_thr and self.redraw_thr:
                start = time.time()
                self.drawThreshold()
                self.redraw_thr = False
                logger.debug(f'redraw thr: {time.time() - start}')

            if self.show_spikes and self.redraw_spikes:
                start = time.time()
                self.drawSpikes()
                self.redraw_spikes = False
                logger.debug(f'redraw spikes: {time.time() - start}')

            if self.show_events and self.redraw_events:
                start = time.time()
                self.drawEvents()
                self.redraw_events = False
                logger.debug(f'redraw events: {time.time() - start}')

        self.plot_item.getViewBox().setXRange(*self._x_range, padding=0)
        self.plot_item.getViewBox().setYRange(*self._y_range, padding=0)

        self.data_item.setVisible(visible)
        self.bg_data_item.setVisible(visible and self.show_bg)
        self.thr_item.setVisible(visible and self.show_thr)

        for item in self.spikes_item_list:
            item.setVisible(visible and self.show_spikes)
        for item in self.events_item_list:
            item.setVisible(visible and self.show_events)

    def drawData(self, data_type):
        if data_type == 'raw':
            data = self.current_raw_object._data
        elif data_type == 'filted':
            data = self.current_filted_object._data

        if self.current_raw_object._timestamps is None:
            # generate x
            start = self._x_range[0]
            end = self._x_range[1]
            if start < self._x_boundary[0]:
                start = self._x_boundary[0]
            if end > self._x_boundary[1]:
                end = self._x_boundary[1]
            x = np.arange(start=start, stop=end)
            data = data[start: end]
            connect = 'auto'
        else:
            timestamps = self.current_raw_object._timestamps

            mask = (timestamps >= self._x_range[0]) & \
                (timestamps < self._x_range[1])

            x = timestamps[mask]
            data = data[mask]
            connect = np.append(np.diff(x) <= 1, 0)

        if self._x_range[1] - self._x_range[0] > self.DOWNSAMPLE_SIZE and self.do_downsampling:
            x, data, connect = self.downsampling(
                x, data, connect, DOWNSAMPLE_SIZE=self.DOWNSAMPLE_SIZE)

        self.data_item.setData(x=x, y=data, connect=connect)
        logger.debug(f'show {len(x)} data point')

    def drawBackgroundData(self):
        if self.current_bg_object is None:
            self.bg_data_item.setData(x=[], y=[])
            return

        data = self.current_bg_object._data

        if self.current_bg_object._timestamps is None:
            # generate x
            start = self._x_range[0]
            end = self._x_range[1]
            if start < self._x_boundary[0]:
                start = self._x_boundary[0]
            if end > self._x_boundary[1]:
                end = self._x_boundary[1]
            x = np.arange(start=start, stop=end)
            data = data[start: end]
            connect = 'auto'
        else:
            timestamps = self.current_bg_object._timestamps
            mask = (timestamps >= self._x_range[0]) & \
                (timestamps <= self._x_range[1])

            x = timestamps[mask]
            data = data[mask]
            connect = np.append(np.diff(x) <= 1, 0)

        if self._x_range[1] - self._x_range[0] > self.DOWNSAMPLE_SIZE and self.do_downsampling:
            x, data, connect = self.downsampling(
                x, data, connect, DOWNSAMPLE_SIZE=self.DOWNSAMPLE_SIZE)

        self.bg_data_item.setPen(pg.mkPen(self.bg_color))
        self.bg_data_item.setData(x=x, y=data, connect=connect)

    def drawThreshold(self):
        if self.current_filted_object is None:
            return

        self.thr_item.setValue(self.current_filted_object.threshold)

    def drawEvents(self):
        self.removeItems(self.events_item_list)
        self.events_item_list = []

        if self.current_showing_events == []:
            return
        if self.current_event_object is None:
            return

        self.events_item_list = self.tsToLines(self.current_event_object._timestamps,
                                               unit_IDs=self.current_event_object._unit_IDs,
                                               showing_unit_IDs=self.current_showing_events,
                                               data_type="events")

        [self.addItem(item) for item in self.events_item_list]

    def drawSpikes(self):
        self.removeItems(self.spikes_item_list)
        self.spikes_item_list = []

        if self.current_showing_units == []:
            return
        if self.current_spike_object is None:
            return

        self.spikes_item_list = self.tsToLines(self.current_spike_object._timestamps,
                                               unit_IDs=self.current_spike_object._unit_IDs,
                                               showing_unit_IDs=self.current_showing_units,
                                               data_type="spikes")

        [self.addItem(item) for item in self.spikes_item_list]

    def removeItems(self, item_list):
        for item in item_list:
            self.removeItem(item)
        # self.spikes_item_list = []

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
            logger.error('Unknown type of timestamps data.')
            return

        mask = (ts >= self._x_range[0]) & (ts <= self._x_range[1])
        sub_ts = ts[mask]
        sub_unit_IDs = unit_IDs[mask]

        # if len(sub_ts) > 30000 and self.do_downsampling:
        #     sub_ts = self.downsampling(
        #         sub_ts, DOWNSAMPLE_SIZE=self.DOWNSAMPLE_SIZE)
        #     sub_unit_IDs = self.downsampling(
        #         sub_unit_IDs, DOWNSAMPLE_SIZE=self.DOWNSAMPLE_SIZE)
        # logger.info(len(sub_ts))

        for ID in showing_unit_IDs:
            unit_ts = sub_ts[sub_unit_IDs == ID]
            if len(unit_ts) == 0:
                continue

            color = self.color_palette_list[unit_color_map[int(ID)]]
            color = (np.array(color) * 255).astype(int)

            n = unit_ts.shape[0]
            x = np.repeat(unit_ts, 2)
            y = np.tile(y_element, n)
            item_list.append(pg.PlotCurveItem(
                x=x, y=y, pen=pg.mkPen(color=color), connect="pairs"))

        return item_list

    def downsampling(self, x, y=None, connect=None, DOWNSAMPLE_SIZE=1):
        length = self._x_range[1] - self._x_range[0]
        ds = length // DOWNSAMPLE_SIZE
        if ds < 1:
            ds = 1
        x1 = x[::ds]
        x = x1
        if y is None and connect is None:
            return x

        y1 = y[::ds]
        y = y1
        if not isinstance(connect, str):
            connect = np.append(np.diff(x) <= ds, 0)
        # ds = len(x) // (self.DOWNSAMPLE_SIZE // 2)
        # x1 = np.empty((n, 2))
        # # start of x-values; try to select a somewhat centered point
        # stx = ds // 2
        # x1[:] = x[stx:stx + n * ds:ds, np.newaxis]
        # x = x1.reshape(n * 2)

        # y1 = np.empty((n, 2))
        # y2 = y[:n * ds].reshape((n, ds))
        # y1[:, 0] = y2.max(axis=1)
        # y1[:, 1] = y2.min(axis=1)
        # y = y1.reshape(n * 2)

        # if connect != 'auto':
        #     connect = np.append(np.diff(x) <= ds, 0)
        return x, y, connect

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
            # if new_range[0] < self._x_boundary[0]:
            #     new_range = (self._x_boundary[0], self.num_data_show)
            # if new_range[1] > self._x_boundary[1]:
            #     new_range = (self._x_boundary[1] - self.num_data_show,
            #                  self._x_boundary[1])

            if new_range[0] < self._x_boundary[0]:
                new_range = (self._x_boundary[0], self.num_data_show)

            self._x_range = new_range

            self.redraw_data = True
            self.redraw_bg = True
            self.redraw_events = True
            self.redraw_spikes = True
            self.updatePlot()

            # self.plot_item.getViewBox().setXRange(*new_range, padding=0)

        elif (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
            """scale y axis."""
            delta = int(event.delta() / 120)
            # current_range = self.plot_item.getViewBox().state['viewRange']
            data_scale = int(self._y_range[1] / (1 + delta / 10))
            self._y_range = (-data_scale, data_scale)

            # self.redraw_data = True
            # self.redraw_bg = True
            self.redraw_events = True
            self.redraw_spikes = True
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
                new_range = (self._x_boundary[0], self.num_data_show)
            if new_range[1] > self._x_boundary[1]:
                new_range = (self._x_boundary[1] - self.num_data_show,
                             self._x_boundary[1])

            if new_range[0] < self._x_boundary[0]:
                new_range = (self._x_boundary[0], self.num_data_show)

            self._x_range = new_range

            self.redraw_data = True
            self.redraw_bg = True
            self.redraw_events = True
            self.redraw_spikes = True
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
