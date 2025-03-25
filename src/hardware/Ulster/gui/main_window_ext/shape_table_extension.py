from PyQt5.QtWidgets import (
    QDockWidget, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QMenu
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QPen

class ShapeTableMixin:
    def createShapeTable(self):
        self.shapeDock = QDockWidget("Shapes", self)
        # Increase the column count to include a "Role" column.
        self.shapeTable = QTableWidget(0, 7, self)
        self.shapeTable.setHorizontalHeaderLabels([
            "ID", "Type", "X", "Y", "Width", "Height", "Role"
        ])
        self.shapeTable.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
        )
        self.shapeTable.cellChanged.connect(self.onShapeTableCellChanged)
        self.shapeDock.setWidget(self.shapeTable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shapeDock)
        self.setupShapeTableContextMenu()
        self.setupDeleteShortcut()

    def setupShapeTableContextMenu(self):
        self.shapeTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.shapeTable.customContextMenuRequested.connect(self.onShapeTableContextMenu)

    def onShapeTableContextMenu(self, pos):
        item = self.shapeTable.itemAt(pos)
        if item:
            row = item.row()
            menu = QMenu(self.shapeTable)
            includeAct = menu.addAction("Include")
            excludeAct = menu.addAction("Exclude")
            sampleHolderAct = menu.addAction("Sample Holder")
            action = menu.exec_(self.shapeTable.viewport().mapToGlobal(pos))
            if action == includeAct:
                self.updateShapeRole(row, "include")
            elif action == excludeAct:
                self.updateShapeRole(row, "exclude")
            elif action == sampleHolderAct:
                self.updateShapeRole(row, "sample holder")

    def updateShapeRole(self, row, role):
        try:
            shape_id = int(self.shapeTable.item(row, 0).text())
            for shape_info in self.image_view.shapes:
                if shape_info["id"] == shape_id:
                    shape_info["role"] = role
                    # Remove "isNew" flag now that a role was explicitly set.
                    if "isNew" in shape_info:
                        shape_info.pop("isNew")
                    # Update the pen color based on role.
                    if role == "include":
                        pen = QPen(QColor("green"), 2)
                    elif role == "exclude":
                        pen = QPen(QColor("red"), 2)
                    elif role == "sample holder":
                        pen = QPen(QColor("purple"), 2)
                    else:
                        pen = QPen(QColor("black"), 2)
                    shape_info["item"].setPen(pen)
                    # Force the scene to update so the change is visible.
                    self.image_view.scene.update()
                    break
            self.updateShapeTable()
        except Exception as e:
            print("Error updating shape role:", e)

    def updateShapeTable(self):
        shapes = getattr(self.image_view, 'shapes', [])
        self.shapeTable.blockSignals(True)
        self.shapeTable.setRowCount(len(shapes))
        for row, shape_info in enumerate(shapes):
            shape_id = shape_info.get("id", "")
            shape_type = shape_info.get("type", "")
            # Get the role; default to "include" if not present.
            role = shape_info.get("role", "include")
            item = shape_info.get("item")
            rect = item.rect() if hasattr(item, 'rect') else item.boundingRect()
            self.shapeTable.setItem(row, 0, QTableWidgetItem(str(shape_id)))
            self.shapeTable.setItem(row, 1, QTableWidgetItem(shape_type))
            self.shapeTable.setItem(row, 2, QTableWidgetItem(f"{rect.x():.2f}"))
            self.shapeTable.setItem(row, 3, QTableWidgetItem(f"{rect.y():.2f}"))
            self.shapeTable.setItem(row, 4, QTableWidgetItem(f"{rect.width():.2f}"))
            self.shapeTable.setItem(row, 5, QTableWidgetItem(f"{rect.height():.2f}"))
            self.shapeTable.setItem(row, 6, QTableWidgetItem(role))
            # Set the row background color:
            # Gray for new elements; otherwise, light green for include,
            # light coral for exclude, and purple for sample holder.
            if shape_info.get("isNew", False):
                color = QColor("gray")
            else:
                if role == "include":
                    color = QColor("lightgreen")
                elif role == "exclude":
                    color = QColor("lightcoral")
                elif role == "sample holder":
                    color = QColor("purple")
                else:
                    color = QColor("white")
            for col in range(7):
                self.shapeTable.item(row, col).setBackground(QBrush(color))
        self.shapeTable.blockSignals(False)

    def onShapeTableCellChanged(self, row, column):
        try:
            # Only allow editing for geometry columns.
            if column in [2, 3, 4, 5]:
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

    def setupDeleteShortcut(self):
        # Override the keyPressEvent for the shape table to capture the Delete key.
        self.shapeTable.keyPressEvent = self.shapeTableKeyPressEvent

    def shapeTableKeyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.deleteShapesFromTable()
        else:
            # Call the default handler.
            QTableWidget.keyPressEvent(self.shapeTable, event)

    def deleteShapesFromTable(self):
        # Delete selected rows from the table and remove corresponding shapes from the image view.
        selected_rows = sorted(
            {index.row() for index in self.shapeTable.selectedIndexes()},
            reverse=True
        )
        for row in selected_rows:
            try:
                shape_id = int(self.shapeTable.item(row, 0).text())
            except Exception as e:
                continue
            # Remove the shape from image_view.
            for shape_info in self.image_view.shapes:
                if shape_info["id"] == shape_id:
                    self.image_view.scene.removeItem(shape_info["item"])
                    self.image_view.shapes.remove(shape_info)
                    break
        self.updateShapeTable()
