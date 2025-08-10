from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
)


class ShapeTableMixin:

    def create_shape_table(self):
        self.shapeDock = QDockWidget("Shapes", self)
        # Increase the column count to include a "Role" column.
        self.shapeTable = QTableWidget(0, 7, self)
        self.shapeTable.setHorizontalHeaderLabels(
            ["ID", "Type", "X", "Y", "Width", "Height", "Role"]
        )
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
                self.update_shape_role(row, "include")
            elif action == excludeAct:
                self.update_shape_role(row, "exclude")
            elif action == sampleHolderAct:
                self.update_shape_role(row, "sample holder")

    def apply_shape_role(self, shape_info):
        """Update the appearance of the shape based on its role."""
        role = shape_info.get("role", "include")
        item = shape_info["item"]

        # Remove any previous extra items if present.
        if "diagonals" in shape_info:
            for line in shape_info["diagonals"]:
                self.image_view.scene.removeItem(line)
            shape_info.pop("diagonals")
        if "center_marker" in shape_info:
            self.image_view.scene.removeItem(shape_info["center_marker"])
            shape_info.pop("center_marker")

        if role == "include":
            pen = QPen(QColor("green"), 2)
            item.setPen(pen)
        elif role == "exclude":
            pen = QPen(QColor("red"), 2)
            item.setPen(pen)
        elif role == "sample holder":
            # Convert the shape to a square and update its appearance.
            rect = item.rect() if hasattr(item, "rect") else item.boundingRect()
            cx = rect.x() + rect.width() / 2
            cy = rect.y() + rect.height() / 2
            side = min(rect.width(), rect.height())
            new_side = side
            new_x = cx - new_side / 2
            new_y = cy - new_side / 2

            # Update shape to square.
            item.setRect(new_x, new_y, new_side, new_side)
            pen = QPen(QColor("purple"), 2)
            item.setPen(pen)

            # Draw diagonal lines.

            diag1 = QGraphicsLineItem(new_x, new_y, new_x + new_side, new_y + new_side)
            diag2 = QGraphicsLineItem(new_x + new_side, new_y, new_x, new_y + new_side)
            diag_pen = QPen(QColor("purple"), 1)
            diag1.setPen(diag_pen)
            diag2.setPen(diag_pen)
            self.image_view.scene.addItem(diag1)
            self.image_view.scene.addItem(diag2)

            # Draw center marker.
            center_radius = 3
            center_point = QGraphicsEllipseItem(
                cx - center_radius,
                cy - center_radius,
                2 * center_radius,
                2 * center_radius,
            )
            center_point.setBrush(QColor("purple"))
            center_point.setPen(QPen(Qt.NoPen))
            self.image_view.scene.addItem(center_point)

            # Store the extra graphics items so they can be removed later.
            shape_info["diagonals"] = [diag1, diag2]
            shape_info["center_marker"] = center_point

            # Conversion calculation.
            pixels_per_mm = new_side / 18.0  # Assuming 18mm per side for a real square.
            shape_info["pixels_per_mm"] = pixels_per_mm
            self.pixel_to_mm_ratio = pixels_per_mm
            self.include_center = (cx, cy)
            self.update_conversion_label()
        else:
            pen = QPen(QColor("black"), 2)
            item.setPen(pen)

        # Always update the scene to reflect changes.
        self.image_view.scene.update()

    def update_shape_role(self, row, role):
        try:
            shape_id = int(self.shapeTable.item(row, 0).text())
            # Update the role in shape_info and apply appearance changes.
            for shape_info in self.image_view.shapes:
                if shape_info["id"] == shape_id:
                    shape_info["role"] = role
                    # Remove "isNew" flag if present.
                    shape_info.pop("isNew", None)
                    # Update appearance using the new helper method.
                    self.apply_shape_role(shape_info)
                    break
            self.update_shape_table()  # Refresh the table to display the updated role.
        except Exception as e:
            print("Error updating shape role:", e)

    def update_shape_table(self):
        shapes = getattr(self.image_view, "shapes", [])
        self.shapeTable.blockSignals(True)
        self.shapeTable.setRowCount(len(shapes))
        for row, shape_info in enumerate(shapes):
            # If a role has already been defined, enforce the appearance.
            if "role" in shape_info:
                self.apply_shape_role(shape_info)
            else:
                # Default role is "include" if none is set.
                shape_info["role"] = "include"

            role = shape_info["role"]
            item = shape_info.get("item")
            rect = item.rect() if hasattr(item, "rect") else item.boundingRect()

            # Update table cells.
            self.shapeTable.setItem(
                row, 0, QTableWidgetItem(str(shape_info.get("id", "")))
            )
            self.shapeTable.setItem(
                row, 1, QTableWidgetItem(shape_info.get("type", ""))
            )
            self.shapeTable.setItem(row, 2, QTableWidgetItem(f"{rect.x():.2f}"))
            self.shapeTable.setItem(row, 3, QTableWidgetItem(f"{rect.y():.2f}"))
            self.shapeTable.setItem(row, 4, QTableWidgetItem(f"{rect.width():.2f}"))
            self.shapeTable.setItem(row, 5, QTableWidgetItem(f"{rect.height():.2f}"))
            self.shapeTable.setItem(row, 6, QTableWidgetItem(role))

            # Set the row background color based on the role:
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
                        if hasattr(item, "setRect"):
                            item.setRect(x, y, w, h)
                        break
        except Exception as e:
            print("Error updating shape from table:", e)

    def setupDeleteShortcut(self):
        # Override the keyPressEvent for the shape table to capture the Delete key.
        self.shapeTable.keyPressEvent = self.shapeTableKeyPressEvent

    def shapeTableKeyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_shapes_from_table()
        else:
            # Call the default handler.
            QTableWidget.keyPressEvent(self.shapeTable, event)

    def delete_shapes_from_table(self):
        # Delete selected rows from the table and remove corresponding shapes from the image view.
        selected_rows = sorted(
            {index.row() for index in self.shapeTable.selectedIndexes()},
            reverse=True,
        )
        for row in selected_rows:
            try:
                shape_id = int(self.shapeTable.item(row, 0).text())
            except Exception:
                continue
            for shape_info in self.image_view.shapes:
                if shape_info["id"] == shape_id:
                    self.image_view.scene.removeItem(shape_info["item"])
                    # Also remove extra items if present.
                    if "diagonals" in shape_info:
                        for item in shape_info["diagonals"]:
                            self.image_view.scene.removeItem(item)
                    if "center_marker" in shape_info:
                        self.image_view.scene.removeItem(shape_info["center_marker"])
                    self.image_view.shapes.remove(shape_info)
                    break
        self.update_shape_table()

    def delete_all_shapes_from_table(self):
        # Delete all rows from the table and remove all corresponding shapes from the image view.
        shapes_to_delete = list(self.image_view.shapes)
        for shape_info in shapes_to_delete:
            item = shape_info.get("item")
            if item is not None:
                try:
                    self.image_view.scene.removeItem(item)
                except RuntimeError:
                    pass
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
        self.image_view.shapes.clear()
        self.update_shape_table()
