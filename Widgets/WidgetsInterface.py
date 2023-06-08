from PyQt5.QtWidgets import QWidget


class WidgetsInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Widgets Interface"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
