import random
import math
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QGraphicsEllipseItem, QLabel, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QEvent, QPointF
from PyQt5.QtGui import QColor, QPen, QTransform
from hardware.Ulster.gui.extra.elements import HoverableEllipseItem
import sip
from hardware.Ulster.gui.image_view_ext.point_editing_extension import null_dict

class ZonePointsMixin:

    def create_zone_points_widget(self):
        """
        Creates a dock widget that generates and displays zone points.
        Auto-generated points appear as a red circle with an underlying transparent cyan circle.
        User-defined points (added via double left-click) appear as larger blue circles.
        The table lists all points.
        """
        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input controls.
        inputLayout = QHBoxLayout()
        inputLayout.addWidget(QLabel("N points"))
        self.pointCountSpinBox = QSpinBox()
        self.pointCountSpinBox.setMinimum(2)
        self.pointCountSpinBox.setMaximum(1000)
        self.pointCountSpinBox.setValue(10)
        inputLayout.addWidget(self.pointCountSpinBox)

        inputLayout.addWidget(QLabel("% offset"))
        self.shrinkSpinBox = QSpinBox()
        self.shrinkSpinBox.setMinimum(0)
        self.shrinkSpinBox.setMaximum(100)
        self.shrinkSpinBox.setValue(5)
        inputLayout.addWidget(self.shrinkSpinBox)

        real_x, real_y = 9.25, -6.6  # sensible fallback defaults
        try:
            active_stage_ids = self.config.get("active_translation_stages", [])
            translation_stages = self.config.get("translation_stages", [])
            # Take the first active stage, or fallback
            active_id = active_stage_ids[0] if active_stage_ids else None
            real_zero = None
            for stage in translation_stages:
                if stage.get("id") == active_id:
                    real_zero = stage.get("real_zero", {})
                    break
            if real_zero:
                real_x = real_zero.get("x_mm", real_x)
                real_y = real_zero.get("y_mm", real_y)
        except Exception as e:
            print(f"Error fetching real_zero for active stage: {e}")


        # Instead of a mmComboBox, add user-defined real center position controls.
        self.realXLabel = QLabel("X_pos, mm")
        inputLayout.addWidget(self.realXLabel)
        self.real_x_pos_mm = QDoubleSpinBox()
        self.real_x_pos_mm.setDecimals(2)
        self.real_x_pos_mm.setRange(-1000.0, 1000.0)
        self.real_x_pos_mm.setValue(real_x)  # default value; can be adjusted
        inputLayout.addWidget(self.real_x_pos_mm)

        self.realYLabel = QLabel("Y_pos, mm")
        inputLayout.addWidget(self.realYLabel)
        self.real_y_pos_mm = QDoubleSpinBox()
        self.real_y_pos_mm.setDecimals(2)
        self.real_y_pos_mm.setRange(-1000.0, 1000.0)
        self.real_y_pos_mm.setValue(real_y)  # default value; can be adjusted
        inputLayout.addWidget(self.real_y_pos_mm)

        # Conversion label remains to show the pixel-to-mm conversion factor.
        self.conversionLabel = QLabel("Conversion: 1.00 px/mm")
        inputLayout.addWidget(self.conversionLabel)

        self.generatePointsBtn = QPushButton("Generate Points")
        inputLayout.addWidget(self.generatePointsBtn)
        self.updateCoordinatesBtn = QPushButton("Update Coordinates")
        inputLayout.addWidget(self.updateCoordinatesBtn)
        self.updateCoordinatesBtn.clicked.connect(self.update_coordinates)

        layout.addLayout(inputLayout)

        # Table to display points.
        self.pointsTable = QTableWidget(0, 6)
        self.pointsTable.setHorizontalHeaderLabels([
            "ID", "X (px)", "Y (px)", "X (mm)", "Y (mm)", "Measurement"
        ])
        layout.addWidget(self.pointsTable)

        self.pointsTable.selectionModel().selectionChanged.connect(self.on_points_table_selection)

        # Install an event filter on the table to capture key presses (for Delete key)
        self.pointsTable.installEventFilter(self)

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)

        self.generatePointsBtn.clicked.connect(self.generate_zone_points)

        # Initialize the unified points dictionary.
        # We assume that self.image_view is your graphics view.
        self.image_view.points_dict = {
            "generated": {"points": [], "zones": []},
            "user": {"points": [], "zones": []}
        }
        self.pixel_to_mm_ratio = 1.0
        # self.include_center is determined from the center of the include shape.
        # That pixel coordinate corresponds to (self.real_x_pos_mm, self.real_y_pos_mm) in real mm.
        self.include_center = (0, 0)  # default value, should be set by the other mixin

    def update_conversion_label(self):
        self.conversionLabel.setText(f"Conversion: {self.pixel_to_mm_ratio:.2f} px/mm")

    def generate_zone_points(self):
        N = self.pointCountSpinBox.value()
        shrink_percent = self.shrinkSpinBox.value()
        shrink_factor = (100 - shrink_percent) / 100.0

        # For simplicity, assume one include shape is available.
        include_shape = None
        exclude_shapes = []
        for shape in self.image_view.shapes:
            role = shape.get("role", "include")
            if role == "include":
                include_shape = shape["item"]
                # Set self.include_center from the include shape.
                if hasattr(include_shape, 'center'):
                    self.include_center = include_shape.center
                else:
                    inc_rect = include_shape.rect() if hasattr(include_shape, 'rect') else include_shape.boundingRect()
                    self.include_center = (inc_rect.x() + inc_rect.width() / 2,
                                           inc_rect.y() + inc_rect.height() / 2)
            elif role == "exclude":
                exclude_shapes.append(shape["item"])
        if include_shape is None:
            print("No include shape defined. Cannot generate points.")
            return

        self.update_conversion_label()

        # Remove previously drawn generated items.
        for item in self.image_view.points_dict["generated"]["points"]:
            self.safe_remove_item(item)

        self.image_view.points_dict["generated"]["points"] = []

        # Remove previously drawn generated zones.
        for item in self.image_view.points_dict["generated"]["zones"]:
            self.safe_remove_item(item)
        self.image_view.points_dict["generated"]["zones"] = []

        # Candidate sampling area.
        inc_rect = include_shape.rect() if hasattr(include_shape, 'rect') else include_shape.boundingRect()
        x_min, y_min = inc_rect.x(), inc_rect.y()
        x_max, y_max = x_min + inc_rect.width(), y_min + inc_rect.height()

        candidates = []
        num_candidates = 10000
        attempts = 0

        def distance(p, q):
            return math.hypot(p[0] - q[0], p[1] - q[1])

        if hasattr(include_shape, 'center') and hasattr(include_shape, 'radius'):
            inc_center = include_shape.center
            reduced_radius = include_shape.radius * shrink_factor

            def sample_from_circle(center, radius):
                angle = random.uniform(0, 2 * math.pi)
                r = math.sqrt(random.random()) * radius
                return (center[0] + r * math.cos(angle), center[1] + r * math.sin(angle))

            while len(candidates) < num_candidates and attempts < num_candidates * 10:
                attempts += 1
                pt = sample_from_circle(inc_center, reduced_radius)
                if not include_shape.contains(include_shape.mapFromScene(pt[0], pt[1])):
                    continue
                if any(ex.contains(ex.mapFromScene(pt[0], pt[1])) for ex in exclude_shapes):
                    continue
                candidates.append(pt)
        else:
            path = include_shape.shape()
            bounding = include_shape.boundingRect()
            center_point = bounding.center()
            transform = QTransform()
            transform.translate(center_point.x(), center_point.y())
            transform.scale(shrink_factor, shrink_factor)
            transform.translate(-center_point.x(), -center_point.y())
            shrunk_path = transform.map(path)
            while len(candidates) < num_candidates and attempts < num_candidates * 10:
                attempts += 1
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                if not shrunk_path.contains(QPointF(x, y)):
                    continue
                if any(ex.contains(ex.mapFromScene(x, y)) for ex in exclude_shapes):
                    continue
                candidates.append((x, y))
        if not candidates:
            print("No candidate points found in the shrunk allowed region.")
            return

        bbox_area = (x_max - x_min) * (y_max - y_min)
        allowed_area = bbox_area * (len(candidates) / float(attempts))
        circle_area = allowed_area / N
        ideal_radius = math.sqrt(circle_area / math.pi)
        print(f"Allowed area: {allowed_area:.2f}, ideal radius: {ideal_radius:.2f}")

        # Farthest-point sampling.
        sum_x = sum(pt[0] for pt in candidates)
        sum_y = sum(pt[1] for pt in candidates)
        centroid = (sum_x / len(candidates), sum_y / len(candidates))
        initial = min(candidates, key=lambda pt: distance(pt, centroid))
        chosen = [initial]
        candidates.remove(initial)
        while len(chosen) < N and candidates:
            best_candidate = None
            best_dist = -1
            for cand in candidates:
                d = min(distance(cand, p) for p in chosen)
                if d > best_dist:
                    best_dist = d
                    best_candidate = cand
            if best_candidate is None:
                break
            chosen.append(best_candidate)
            candidates.remove(best_candidate)
        final_points = chosen[:N]

        # Draw generated points.
        for (x, y) in final_points:
            # Draw the cyan zone.
            cyan_item = QGraphicsEllipseItem(x - ideal_radius, y - ideal_radius,
                                             2 * ideal_radius, 2 * ideal_radius)
            cyan_color = QColor("cyan")
            cyan_color.setAlphaF(0.2)
            cyan_item.setBrush(cyan_color)
            cyan_item.setPen(QPen(Qt.NoPen))
            self.image_view.scene.addItem(cyan_item)
            self.image_view.points_dict["generated"]["zones"].append(cyan_item)
            # Draw the red point.
            red_item = HoverableEllipseItem(x - 4, y - 4, 8, 8)
            red_item.setBrush(QColor("red"))
            red_item.setPen(QPen(Qt.NoPen))
            red_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            red_item.setData(0, "generated")
            red_item.hoverCallback = self.pointHoverChanged
            self.image_view.scene.addItem(red_item)
            self.image_view.points_dict["generated"]["points"].append(red_item)

        self.update_points_table()

    def pointHoverChanged(self, item, hovered):
        """
        When a point is hovered, update its corresponding zone and highlight the table row.
        Handles both generated and user-defined points.
        """
        table = self.pointsTable
        row = None
        if item.data(0) == "generated":
            try:
                idx = self.image_view.points_dict["generated"]["points"].index(item)
                row = idx
                if idx < len(self.image_view.points_dict["generated"]["zones"]):
                    zone_item = self.image_view.points_dict["generated"]["zones"][idx]
                    if hovered:
                        highlight = QColor(255, 0, 0, 51)
                        zone_item.setBrush(highlight)
                    else:
                        orig = QColor("cyan")
                        orig.setAlphaF(0.2)
                        zone_item.setBrush(orig)
            except ValueError:
                pass
        elif item.data(0) == "user":
            try:
                idx = self.image_view.points_dict["user"]["points"].index(item)
                row = len(self.image_view.points_dict["generated"]["points"]) + idx
                if idx < len(self.image_view.points_dict["user"]["zones"]):
                    zone_item = self.image_view.points_dict["user"]["zones"][idx]
                    if hovered:
                        highlight = QColor(255, 0, 0, 51)
                        zone_item.setBrush(highlight)
                    else:
                        default_zone = QColor("blue")
                        default_zone.setAlphaF(0.2)
                        zone_item.setBrush(default_zone)
            except ValueError:
                pass

        if row is not None and table.rowCount() > row:
            highlight = QColor(255, 0, 0, 51)
            normal = QColor("white")
            for col in range(table.columnCount()):
                if table.item(row, col):
                    table.item(row, col).setBackground(highlight if hovered else normal)

    def delete_selected_points(self):
        # Get unique selected row indices.
        selected_rows = sorted({index.row() for index in self.pointsTable.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return

        n_generated = len(self.image_view.points_dict["generated"]["points"])
        gen_rows = [r for r in selected_rows if r < n_generated]
        user_rows = [r - n_generated for r in selected_rows if r >= n_generated]

        # Remove measurement widgets
        for r in selected_rows:
            # Remove widget in column 5 (Measurement)
            widget = self.pointsTable.cellWidget(r, 5)
            if widget is not None:
                self.pointsTable.removeCellWidget(r, 5)
                widget.deleteLater()

        # Remove generated/user points as before...
        # (your original logic)
        for r in sorted(gen_rows, reverse=True):
            point_item = self.image_view.points_dict["generated"]["points"].pop(r)
            zone_item = self.image_view.points_dict["generated"]["zones"].pop(r)
            self.safe_remove_item(zone_item)
            self.safe_remove_item(point_item)

        for r in sorted(user_rows, reverse=True):
            if r < len(self.image_view.points_dict["user"]["points"]):
                point_item = self.image_view.points_dict["user"]["points"].pop(r)
                if r < len(self.image_view.points_dict["user"]["zones"]):
                    zone_item = self.image_view.points_dict["user"]["zones"].pop(r)
                    self.image_view.scene.removeItem(zone_item)
                self.image_view.scene.removeItem(point_item)
        self.update_points_table()

    def delete_selected_points(self):
        """
        Deletes the points corresponding to the selected rows in the table from both the scene and the points dictionary.
        Also removes the associated measurement/history widgets.
        """
        # Get unique selected row indices, sorted descending so row indices don't shift during deletion
        selected_rows = sorted({index.row() for index in self.pointsTable.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return

        n_generated = len(self.image_view.points_dict["generated"]["points"])
        # Separate indices for generated and user points.
        gen_rows = [r for r in selected_rows if r < n_generated]
        user_rows = [r - n_generated for r in selected_rows if r >= n_generated]

        # Remove from measurement_widgets first to prevent shifting
        for r in selected_rows:
            # Remove measurement/history widget from both the table and the list
            if r < len(self.measurement_widgets):
                widget = self.measurement_widgets.pop(r)
                if widget:
                    widget.deleteLater()
            # Remove widget from table column 5 if present
            widget_in_table = self.pointsTable.cellWidget(r, 5)
            if widget_in_table:
                self.pointsTable.removeCellWidget(r, 5)
                widget_in_table.deleteLater()

        # Remove generated points and zones
        for r in sorted(gen_rows, reverse=True):
            if r < len(self.image_view.points_dict["generated"]["points"]):
                point_item = self.image_view.points_dict["generated"]["points"].pop(r)
                zone_item = self.image_view.points_dict["generated"]["zones"].pop(r)
                self.safe_remove_item(zone_item)
                self.safe_remove_item(point_item)

        # Remove user-defined points and zones
        for r in sorted(user_rows, reverse=True):
            if r < len(self.image_view.points_dict["user"]["points"]):
                point_item = self.image_view.points_dict["user"]["points"].pop(r)
                if r < len(self.image_view.points_dict["user"]["zones"]):
                    zone_item = self.image_view.points_dict["user"]["zones"].pop(r)
                    self.safe_remove_item(zone_item)
                self.safe_remove_item(point_item)

        # Update the points table, which will also clear orphaned table widgets
        self.update_points_table()

    def eventFilter(self, source, event):
        """
        Captures key press events on the points table. If the Delete key is pressed,
        the corresponding points are removed.
        """
        if source == self.pointsTable and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.delete_selected_points()
                return True
        # Pass other events to the parent class.
        return super().eventFilter(source, event)

    def update_coordinates(self):
        """Recalculates the coordinates in the table using the updated spin box values."""
        self.update_points_table()

    def update_points_table(self):
        # Gather current points as before
        # Gather current points as before
        points = []
        import sip
        for item in self.image_view.points_dict["generated"]["points"]:
            if sip.isdeleted(item):
                continue  # skip deleted item!
            center = item.sceneBoundingRect().center()
            points.append((center.x(), center.y(), "generated"))
        for item in self.image_view.points_dict["user"]["points"]:
            if sip.isdeleted(item):
                continue  # skip deleted item!
            center = item.sceneBoundingRect().center()
            points.append((center.x(), center.y(), "user"))

        # Adjust measurement_widgets list to match number of points
        # Delete extra widgets if points have been deleted
        while len(self.measurement_widgets) > len(points):
            widget = self.measurement_widgets.pop()
            if widget:
                widget.deleteLater()

        # Add empty entries for new points (no measurement yet)
        while len(self.measurement_widgets) < len(points):
            self.measurement_widgets.append(None)

        self.pointsTable.setRowCount(len(points))

        for idx, (x, y, ptype) in enumerate(points):
            self.pointsTable.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
            if self.pixel_to_mm_ratio:
                x_mm = self.real_x_pos_mm.value() - (x - self.include_center[0]) / self.pixel_to_mm_ratio
                y_mm = self.real_y_pos_mm.value() - (y - self.include_center[1]) / self.pixel_to_mm_ratio
                self.pointsTable.setItem(idx, 3, QTableWidgetItem(f"{x_mm:.2f}"))
                self.pointsTable.setItem(idx, 4, QTableWidgetItem(f"{y_mm:.2f}"))
            else:
                self.pointsTable.setItem(idx, 3, QTableWidgetItem("N/A"))
                self.pointsTable.setItem(idx, 4, QTableWidgetItem("N/A"))

            # Restore or set the measurement/history widget
            widget = self.measurement_widgets[idx]
            if widget is not None and not sip.isdeleted(widget):
                self.pointsTable.setCellWidget(idx, 5, widget)
            else:
                self.pointsTable.setItem(idx, 5, QTableWidgetItem(""))
                self.measurement_widgets[idx] = None  # Remove ref to deleted widget

    def delete_all_points(self):
        """
        Deletes all zone points (generated + user) and their associated zones and widgets from both the
        graphics scene and the points table. Also clears all measurement/history widgets.
        """
        # Remove all generated points and zones
        for item in self.image_view.points_dict["generated"]["points"]:
            self.image_view.scene.removeItem(item)
        for item in self.image_view.points_dict["generated"]["zones"]:
            self.image_view.scene.removeItem(item)
        self.image_view.points_dict["generated"]["points"].clear()
        self.image_view.points_dict["generated"]["zones"].clear()

        # Remove all user-defined points and zones
        for item in self.image_view.points_dict["user"]["points"]:
            self.image_view.scene.removeItem(item)
        for item in self.image_view.points_dict["user"]["zones"]:
            self.image_view.scene.removeItem(item)
        self.image_view.points_dict["user"]["points"].clear()
        self.image_view.points_dict["user"]["zones"].clear()

        # Clear measurement/history widgets
        for widget in self.measurement_widgets:
            if widget:
                widget.deleteLater()
        self.measurement_widgets.clear()

        # Update table (will clear all rows and widgets)
        self.update_points_table()

    def safe_remove_item(self, item):
        """Remove item from scene only if it is still present."""
        try:
            if item in self.image_view.scene.items():
                self.image_view.scene.removeItem(item)
        except Exception as e:
            print(f"Error removing item: {e}")

    def on_points_table_selection(self, selected, deselected):
        try:
            # First, clear all highlights
            def reset_point_style(item, ptype):
                if sip.isdeleted(item):
                    return  # Do not access deleted items
                if ptype == "generated":
                    item.setBrush(QColor("red"))
                elif ptype == "user":
                    item.setBrush(QColor("blue"))

            points = []
            types = []
            for item in self.image_view.points_dict["generated"]["points"]:
                points.append(item)
                types.append("generated")
            for item in self.image_view.points_dict["user"]["points"]:
                points.append(item)
                types.append("user")
            for item, ptype in zip(points, types):
                reset_point_style(item, ptype)

            # Now, highlight all selected rows
            for index in self.pointsTable.selectionModel().selectedRows():
                row = index.row()
                if row < len(self.image_view.points_dict["generated"]["points"]):
                    item = self.image_view.points_dict["generated"]["points"][row]
                    item.setBrush(QColor(255, 255, 0))  # yellow highlight for generated
                else:
                    user_row = row - len(self.image_view.points_dict["generated"]["points"])
                    if user_row < len(self.image_view.points_dict["user"]["points"]):
                        item = self.image_view.points_dict["user"]["points"][user_row]
                        item.setBrush(QColor(255, 255, 0))  # yellow highlight for user
        except Exception as e:
            print(e)

