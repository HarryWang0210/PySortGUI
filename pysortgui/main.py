#!/usr/bin/env python

'''
Created on Dec 5, 2022

@author: harry wang
'''
import argparse
# This Python file uses the following encoding: utf-8
import logging
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

# Load Widgets
from pysortgui.MainWindowDocks import MainWindowDocks

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py


# logging.basicConfig(
#     level=logging.DEBUG,
#     format="[%(asctime)s][%(levelname)-5s] %(message)s (%(filename)s:%(lineno)d)",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
logger = logging.getLogger(__name__)

organization = 'NYCU'
application = 'PySortGUI'


class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''

    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg


class QSSLoader:
    """ Load qss file to change style """

    def __init__(self):
        pass

    @staticmethod
    def read_qss_file(qss_file_name):
        with open(qss_file_name, 'r',  encoding='UTF-8') as file:
            return file.read()


class PySortGUI(MainWindowDocks):
    def __init__(self, parent=None):
        super().__init__(parent)
        if not hasattr(self, 'settings'):
            self.settings = QtCore.QSettings(organization, application)

        geom = QtWidgets.QDesktopWidget().availableGeometry()
        self.resize(int(geom.width()), int(geom.bottom()))
        self.move(geom.topLeft().x(), geom.topLeft().y())
        self.load_style()
        self.setupUi()
        self.connectMenuActions()
        self.connectWidgets()
        self.setWindowTitle("PySortGUI")
        self.restoreLayout()
        logger.info('PySortGUI setup finish.')

    def load_style(self):
        """
        Load the QSS format app style.
        """
        # style_file = '/home/user/qt-material/examples/exporter/dark_teal.qss'
        default_style_file = './UI/style.qss'
        import os
        path = os.path.split(__file__)[0] + os.path.sep
        style_file = os.path.abspath(path + default_style_file)
        # try:
        # import pkg_resources
        # style_file = pkg_resources.resource_filename(
        #     'pysortgui', default_style_file)
        style_sheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(style_sheet)

    def connectMenuActions(self):
        """Connect slots to actions on menu bar."""

        self.children_dict['FileMenu']["OpenFile"].triggered.connect(
            self.openFile, QtCore.Qt.UniqueConnection)
        self.children_dict['FileMenu']["OpenFolder"].triggered.connect(
            self.openFolder, QtCore.Qt.UniqueConnection)
        self.children_dict['FileMenu']["Save"].triggered.connect(
            self.saveChannel)
        self.children_dict['FileMenu']["SaveAll"].triggered.connect(
            self.saveAll)
        self.children_dict['FileMenu']["Export"].triggered.connect(
            self.export)
        self.children_dict['FileMenu']["Exit"].triggered.connect(
            self.close)

        # self.children_dict['EditMenu']["Undo"].triggered.connect(
        #     self.undo)
        # self.children_dict['EditMenu']["Redo"].triggered.connect(
        #     self.redo)
        self.children_dict['EditMenu']["CreateRef"].triggered.connect(
            self.createRef)
        self.children_dict['EditMenu']["DeleteRef"].triggered.connect(
            self.deleteRef)

        self.children_dict['EditMenu']["CopySpike"].triggered.connect(
            self.copySpike)
        self.children_dict['EditMenu']["DeleteSpike"].triggered.connect(
            self.deleteSpike)
        self.children_dict['EditMenu']["RenameSpike"].triggered.connect(
            self.renameSpike)

        self.children_dict['ViewMenu']["SelectEvents"].triggered.connect(
            self.selectEvents)
        self.children_dict['ViewMenu']["SetBackgroundChannel"].triggered.connect(
            self.setBackgroundChannel)

        self.children_dict['SettingMenu']["SaveLayout"].triggered.connect(
            self.saveLayout)
        self.children_dict['SettingMenu']["RestoreLayout"].triggered.connect(
            self.restoreLayout)

        self.children_dict['Help'].triggered.connect(self.help)

    def connectWidgets(self):
        """Connect slots between widgets."""

        # signal_data_file_name_changed
        self.children_dict['ChannelDetail'].signal_data_file_name_changed.connect(
            self.children_dict["TimelineView"].data_file_name_changed)
        self.children_dict['ChannelDetail'].signal_data_file_name_changed.connect(
            self.children_dict["WaveformsView"].data_file_name_changed)
        self.children_dict['ChannelDetail'].signal_data_file_name_changed.connect(
            self.children_dict["ISIView"].data_file_name_changed)
        self.children_dict['ChannelDetail'].signal_data_file_name_changed.connect(
            self.children_dict["ClustersView"].data_file_name_changed)
        self.children_dict['ChannelDetail'].signal_data_file_name_changed.connect(
            self.children_dict["UnitOperateTools"].data_file_name_changed)

        # event_data_changed
        self.children_dict['ChannelDetail'].signal_event_data_changed.connect(
            self.children_dict["TimelineView"].event_data_changed)

        # showing_events_changed: send selected events unit
        self.children_dict['ChannelDetail'].signal_showing_events_changed.connect(
            self.children_dict["TimelineView"].showing_events_changed)

        # continuous_data_changed
        self.children_dict['ChannelDetail'].signal_continuous_data_changed.connect(
            self.children_dict["TimelineView"].continuous_data_changed)

        # background_continuous_data_changed
        self.children_dict['ChannelDetail'].signal_background_continuous_data_changed.connect(
            self.children_dict["TimelineView"].background_continuous_data_changed)

        # spike_data_changed: ChannelDetail send spike data to UnitOperateTools
        self.children_dict['ChannelDetail'].signal_spike_data_changed.connect(
            self.children_dict["UnitOperateTools"].spike_data_changed)

        # updating_spike_data: UnitOperateTools send spike data to ChannelDetail
        self.children_dict['UnitOperateTools'].signal_updating_spike_data.connect(
            self.children_dict["ChannelDetail"].updating_spike_data)

        # showing_spike_data_changed: UnitOperateTools send spike data to others widgets
        self.children_dict['UnitOperateTools'].signal_showing_spike_data_changed.connect(
            self.children_dict["TimelineView"].showing_spike_data_changed)
        self.children_dict['UnitOperateTools'].signal_showing_spike_data_changed.connect(
            self.children_dict["ClustersView"].showing_spike_data_changed)
        self.children_dict['UnitOperateTools'].signal_showing_spike_data_changed.connect(
            self.children_dict["ISIView"].showing_spike_data_changed)
        self.children_dict['UnitOperateTools'].signal_showing_spike_data_changed.connect(
            self.children_dict["WaveformsView"].showing_spike_data_changed)

        # showing_units_changed: send selected spike units to others widgets
        self.children_dict['UnitOperateTools'].signal_showing_units_changed.connect(
            self.children_dict["TimelineView"].showing_units_changed)
        self.children_dict['UnitOperateTools'].signal_showing_units_changed.connect(
            self.children_dict["ClustersView"].showing_units_changed)
        self.children_dict['UnitOperateTools'].signal_showing_units_changed.connect(
            self.children_dict["ISIView"].showing_units_changed)
        self.children_dict['UnitOperateTools'].signal_showing_units_changed.connect(
            self.children_dict["WaveformsView"].showing_units_changed)

        # feature_on_selection_state_changed
        self.children_dict['UnitOperateTools'].signal_feature_on_selection_state_changed.connect(
            self.children_dict["ClustersView"].feature_on_selection_state_changed)

        # features_changed: 3 axis features
        self.children_dict['UnitOperateTools'].signal_features_changed.connect(
            self.children_dict["ClustersView"].features_changed)

        # isi_threshold_changed
        self.children_dict['UnitOperateTools'].signal_isi_threshold_changed.connect(
            self.children_dict["ISIView"].isi_threshold_changed)

        # manual_mode_state_changed: activate manual pen mode
        self.children_dict['UnitOperateTools'].signal_manual_mode_state_changed.connect(
            self.children_dict["ClustersView"].manual_mode_state_changed)
        # self.children_dict["UnitOperateTools"].signal_manual_mode_state_changed.connect(
        #     self.children_dict["WaveformsView"].manual_mode_state_changed)

        # manual_waveforms: send selected waveforms for manual sorting
        self.children_dict['ClustersView'].signal_manual_waveforms.connect(
            self.children_dict["UnitOperateTools"].manual_waveforms)
        # self.children_dict['WaveformsView'].signal_manual_waveforms.connect(
        #     self.children_dict["UnitOperateTools"].manual_waveforms)

        # select_point: send selected waveform, use to show single waveform
        self.children_dict["ClustersView"].signal_select_point.connect(
            self.children_dict["WaveformsView"].select_point)

    def openFile(self):
        """Open file manager and load selected file."""
        self.children_dict["ChannelDetail"].openFile()

    def openFolder(self):
        """Open file manager and load selected folder."""
        self.children_dict["ChannelDetail"].openFolder()

    def saveChannel(self):
        """Save single channel."""
        self.children_dict["ChannelDetail"].saveChannel()

    def saveAll(self):
        """Save all channels."""
        logger.error("saveAll: Unimplemented error")
        # self.saveChannel()
        pass

    def export(self):
        """Export imported file to pyephys format.
        """
        self.children_dict["ChannelDetail"].export()

    # def undo(self):
    #     """Undo the last change."""
    #     self.children_dict['ChannelDetail'].undo_stack.undo()
    #     print("undo")

    # def redo(self):
    #     """Redo the change."""
    #     self.children_dict['ChannelDetail'].undo_stack.redo()
    #     print("redo")
    def createRef(self):
        """Create the reference channel."""
        self.children_dict["ChannelDetail"].createReference()

    def deleteRef(self):
        """Delete the reference channel."""
        # logger.critical('deleteRef: NoImplementError')
        self.children_dict["ChannelDetail"].deleteReference()

    def copySpike(self):
        """Copy the spike data."""
        self.children_dict["ChannelDetail"].copySpike()

    def deleteSpike(self):
        """Delete the spike data."""
        self.children_dict["ChannelDetail"].deleteSpike()

    def renameSpike(self):
        """Rename the spike data."""
        logger.error('renameSpike: Unimplemented error')

    # View
    def selectEvents(self):
        """Select the event units to show"""
        self.children_dict['ChannelDetail'].selectEvents()

    def setBackgroundChannel(self):
        """Set background channel, only for display"""
        self.children_dict['ChannelDetail'].setBackgroundChannel()

    # Setting
    def saveLayout(self, id=0):
        """Save the layout changes."""
        self.settings.setValue(
            "geometry_{:d}".format(id), self.saveGeometry())
        self.settings.setValue(
            "windowState_{:d}".format(id), self.saveState())

    def restoreLayout(self, id=0):
        """Rstore_ the layout changes."""
        if not self.settings.value("geometry_{:d}".format(id)) is None:
            self.restoreGeometry(
                self.settings.value("geometry_{:d}".format(id)))
        # else:
        #     geom = QtGui.QDesktopWidget().availableGeometry()
        #     self.resize(geom.width(), geom.bottom())
        #     self.move(geom.topLeft().x(), geom.topLeft().y())

        if not self.settings.value("windowState_{:d}".format(id)) is None:
            self.restoreState(
                self.settings.value("windowState_{:d}".format(id)))

    def help(self):
        """Show help documentation."""
        logger.error('help: Unimplemented error')

    def closeEvent(self, event):
        """Close app."""
        super().closeEvent(event)


def launch_app():
    """Start this gui app
    """
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug level logger')

    args = parser.parse_args()

    if args.debug:
        root_logger = logging.getLogger('pysortgui')
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Enable debug logger")

    app = QApplication(sys.argv)
    window = PySortGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    launch_app()
