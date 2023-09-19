import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QVBoxLayout, QWidget, QAbstractItemView
from PyQt5.QtGui import QStandardItem, QStandardItemModel
import pandas as pd


class DataFrameViewer(QMainWindow):
    def __init__(self, dataframe):
        super().__init__()

        self.dataframe = dataframe

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Pandas DataFrame Viewer')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.tree_view = QTreeView(self)
        self.tree_view.setSelectionMode(QAbstractItemView.SingleSelection)

        # Convert DataFrame to QStandardItemModel
        model = self.dataframe_to_model(self.dataframe)
        self.tree_view.setModel(model)

        selection_model = self.tree_view.selectionModel()
        selection_model.selectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.tree_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def dataframe_to_model(self, df):
        model = QStandardItemModel()

        # Set headers (index names)
        model.setHorizontalHeaderLabels(
            ["Name", "Area"] + list(df.columns))

        for ChanlID in df.index.get_level_values(0).unique():
            top_item = QStandardItem(str(ChanlID))  # Top level, ID
            sub_data = df.xs(ChanlID, level=0)
            first_items = [top_item, QStandardItem(str(sub_data.iloc[0, :].name))] + [
                QStandardItem(str(col_value)) for col_value in sub_data.iloc[0, :]]
            model.appendRow(first_items)

            for label, sub_row in sub_data.iloc[1:, :].iterrows():
                values = [QStandardItem(""),  QStandardItem(
                    str(label))] + [QStandardItem(str(col_value)) for col_value in sub_row]
                top_item.appendRow(values)
        return model

    def on_selection_changed(self, selected, deselected):
        model = self.tree_view.model()

        columns = [model.horizontalHeaderItem(
            ind).text() for ind in range(model.columnCount())]
        indexes = selected.indexes()
        items = [model.itemFromIndex(ind) for ind in indexes]

        if items[0].parent() != None:
            items[0] = items[0].parent()

        meta_data = [item.text() for item in items]
        meta_data = dict(zip(columns, meta_data))
        print("Selected:", meta_data)


def main():
    # Sample Pandas DataFrame with MultiIndex
    data = {
        ('John', 'New York'): [28, 176],
        ('John', 'London'): [24, 168],
        ('Jane', 'Paris'): [22, 162],
        ('Jane', 'Tokyo'): [30, 160]
    }
    df = pd.DataFrame(data, index=['Age', 'Height']).T

    app = QApplication(sys.argv)
    window = DataFrameViewer(df)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
