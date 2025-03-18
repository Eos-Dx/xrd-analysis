# hardware/Ulster/gui/main_window_ext/shape_table_extension.py
from PyQt5.QtWidgets import QDockWidget, QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt


class ShapeTableMixin:
    def createShapeTable(self):
        self.shapeDock = QDockWidget("Shapes", self)
        self.shapeTable = QTableWidget(0, 6, self)
        self.shapeTable.setHorizontalHeaderLabels(["ID", "Type", "X", "Y", "Width", "Height"])
        self.shapeTable.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.shapeTable.cellChanged.connect(self.onShapeTableCellChanged)
        self.shapeDock.setWidget(self.shapeTable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shapeDock)

    def updateShapeTable(self):
        # Expect self.image_view.shapes to be a list of shape dictionaries.
        shapes = getattr(self.image_view, 'shapes', [])
        self.shapeTable.blockSignals(True)
        self.shapeTable.setRowCount(len(shapes))
        for row, shape_info in enumerate(shapes):
            shape_id = shape_info.get("id", "")
            shape_type = shape_info.get("type", "")
            item = shape_info.get("item")
            rect = item.rect() if hasattr(item, 'rect') else item.boundingRect()
            self.shapeTable.setItem(row, 0, QTableWidgetItem(str(shape_id)))
            self.shapeTable.setItem(row, 1, QTableWidgetItem(shape_type))
            self.shapeTable.setItem(row, 2, QTableWidgetItem(f"{rect.x():.2f}"))
            self.shapeTable.setItem(row, 3, QTableWidgetItem(f"{rect.y():.2f}"))
            self.shapeTable.setItem(row, 4, QTableWidgetItem(f"{rect.width():.2f}"))
            self.shapeTable.setItem(row, 5, QTableWidgetItem(f"{rect.height():.2f}"))
        self.shapeTable.blockSignals(False)

    def onShapeTableCellChanged(self, row, column):
        try:
            shape_id = int(self.shapeTable.item(row, 0).text())
            for shape_info in self.image_view.shapes:
                if shape_info["id"] == shape_id:
                    item = shape_info["item"]
                    x = float(self.shapeTable.item(row, 2).text())
                    y = float(self.shapeTable.item(row, 3).text())
                    w = float(self.shapeTable.item(row, 4).text())
                    h = float(self.shapeTable.item(row, 5).text())
                    if hasattr(item, 'setRect'):
                        item.setRect(x, y, w, h)
                    break
        except Exception as e:
            print("Error updating shape from table:", e)
