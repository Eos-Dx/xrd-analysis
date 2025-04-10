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
                    # Remove any previous extra items (diagonals or center marker) if present.
                    if "diagonals" in shape_info:
                        for item in shape_info["diagonals"]:
                            self.image_view.scene.removeItem(item)
                        shape_info.pop("diagonals")
                    if "center_marker" in shape_info:
                        self.image_view.scene.removeItem(shape_info["center_marker"])
                        shape_info.pop("center_marker")

                    if role == "include":
                        pen = QPen(QColor("green"), 2)
                        shape_info["item"].setPen(pen)
                    elif role == "exclude":
                        pen = QPen(QColor("red"), 2)
                        shape_info["item"].setPen(pen)
                    elif role == "sample holder":
                        # Convert the shape to a square and draw diagonals and center marker.
                        item = shape_info["item"]
                        # Get current rectangle.
                        rect = item.rect() if hasattr(item, 'rect') else item.boundingRect()
                        # Compute center of the shape.
                        cx = rect.x() + rect.width() / 2
                        cy = rect.y() + rect.height() / 2
                        # Use the smaller dimension to force a square.
                        side = min(rect.width(), rect.height())
                        new_side = side
                        new_x = cx - new_side / 2
                        new_y = cy - new_side / 2
                        # Update shape to square.
                        item.setRect(new_x, new_y, new_side, new_side)
                        pen = QPen(QColor("purple"), 2)
                        item.setPen(pen)

                        # Draw diagonal lines.
                        from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsEllipseItem
                        diag1 = QGraphicsLineItem(new_x, new_y, new_x + new_side, new_y + new_side)
                        diag2 = QGraphicsLineItem(new_x + new_side, new_y, new_x, new_y + new_side)
                        diag_pen = QPen(QColor("purple"), 1)
                        diag1.setPen(diag_pen)
                        diag2.setPen(diag_pen)
                        self.image_view.scene.addItem(diag1)
                        self.image_view.scene.addItem(diag2)

                        # Draw center marker.
                        center_radius = 3
                        center_point = QGraphicsEllipseItem(cx - center_radius, cy - center_radius,
                                                            2 * center_radius, 2 * center_radius)
                        center_point.setBrush(QColor("purple"))
                        center_point.setPen(QPen(Qt.NoPen))
                        self.image_view.scene.addItem(center_point)

                        # Store extra items so they can be removed later.
                        shape_info["diagonals"] = [diag1, diag2]
                        shape_info["center_marker"] = center_point

                        # Calculate conversion factor: the real square is 18mm per side.
                        pixels_per_mm = new_side / 18.0
                        shape_info["pixels_per_mm"] = pixels_per_mm
                        # Also update the Zone Points mixin attributes.
                        self.pixel_to_mm_ratio = pixels_per_mm
                        self.include_center = (cx, cy)
                        # Now update the conversion label.
                        self.updateConversionLabel()
                    else:
                        pen = QPen(QColor("black"), 2)
                        shape_info["item"].setPen(pen)
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
                    # Also remove any extra items (diagonals, center marker) if present.
                    if "diagonals" in shape_info:
                        for extra_item in shape_info["diagonals"]:
                            self.image_view.scene.removeItem(extra_item)
                    if "center_marker" in shape_info:
                        self.image_view.scene.removeItem(shape_info["center_marker"])
                    self.image_view.shapes.remove(shape_info)
                    break
        self.updateShapeTable()

    def deleteAllShapesFromTable(self):
        # Delete all rows from the table and remove all corresponding shapes from the image view.
        shapes_to_delete = list(self.image_view.shapes)  # Make a copy to avoid modifying while iterating

        for shape_info in shapes_to_delete:
            item = shape_info.get("item")
            if item is not None:
                try:
                    self.image_view.scene.removeItem(item)
                except RuntimeError:
                    pass  # Item may already be deleted
            # Remove any extra items if they exist
            for extra_key in ["diagonals", "center_marker"]:
                extra_items = shape_info.get(extra_key)
                if isinstance(extra_items, list):
                    for extra_item in extra_items:
                        try:
                            self.image_view.scene.removeItem(extra_item)
                        except RuntimeError:
                            pass
                elif extra_items is not None:
                    try:
                        self.image_view.scene.removeItem(extra_items)
                    except RuntimeError:
                        pass

        # Clear the list of shapes
        self.image_view.shapes.clear()
        # Update the shape table
        self.updateShapeTable()
