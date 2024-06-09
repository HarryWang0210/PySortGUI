import logging

from PyQt5.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class WidgetsInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Widgets Interface"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        # self.widget_active: bool = False

    def widgetVisibilityChanged(self, visible: bool):
        """Abstract Slot: detect whether widget is visible

        Args:
            visible (bool): _description_
        """
        pass
        # self.widget_active = visible
        # logger.debug(f'{self.window_title}: {self.widget_active}')
