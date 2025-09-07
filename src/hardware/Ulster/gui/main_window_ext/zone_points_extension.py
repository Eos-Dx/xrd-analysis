"""Main zone points extension functionality."""

from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import sip
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hardware.Ulster.gui.technical.widgets import MeasurementHistoryWidget

from .points.zone_geometry import compute_ideal_radius, farthest_point_sampling
from .points.zone_points_constants import ZonePointsConstants
from .points.zone_points_renderer import ZonePointsRenderer, ZonePointsTableManager
from .points.zone_points_ui_builder import ZonePointsGeometry, ZonePointsUIBuilder


class ZonePointsMixin:
    """
    Mixin for zone-based point generation and management in a Qt GUI.

    Host class must define/initialize:
        - self.config
        - self.image_view (with .scene, .shapes, .points_dict)
        - self.measurement_widgets (list)
        - self.include_center (tuple)
        - self.pixel_to_mm_ratio (float)
    """

    def create_zone_points_widget(self):
        """Create the zone points widget with all UI components."""
        self._initialize_state()

        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Create UI components using helper classes
        controls_layout = self._create_all_controls()
        layout.addLayout(controls_layout)

        # Splitter with left table and right measurements panel
        splitter = QSplitter(Qt.Horizontal)

        # Left: points table
        self.pointsTable = ZonePointsUIBuilder.create_points_table(self)
        splitter.addWidget(self.pointsTable)

        # Right: measurements tree (collapsible sections per point)
        self.measurementsTree = QTreeWidget()
        self.measurementsTree.setColumnCount(1)
        self.measurementsTree.setHeaderLabels(["Point"])
        self.measurementsTree.setExpandsOnDoubleClick(True)
        splitter.addWidget(self.measurementsTree)

        layout.addWidget(splitter)

        self._setup_event_handlers()

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)

    def _initialize_state(self):
        """Initialize required state attributes."""
        if not hasattr(self, "next_point_id"):
            self.next_point_id = 1
        if not hasattr(self, "measurement_widgets"):
            self.measurement_widgets = {}
        # Hidden parking parent to keep widgets alive when detaching from table
        if not hasattr(self, "_widgets_parking") or self._widgets_parking is None:
            self._widgets_parking = QWidget()
            self._widgets_parking.hide()
        # Mapping for tree items per point
        if not hasattr(self, "_measurement_items"):
            self._measurement_items = {}
        if not hasattr(self, "include_center"):
            self.include_center = (0, 0)
        if not hasattr(self, "pixel_to_mm_ratio"):
            self.pixel_to_mm_ratio = 1.0
        if not hasattr(self.image_view, "points_dict"):
            self.image_view.points_dict = {
                "generated": {"points": [], "zones": []},
                "user": {"points": [], "zones": []},
            }

    def _create_all_controls(self) -> QHBoxLayout:
        """Create all control layouts in a single horizontal layout."""
        layout = QHBoxLayout()

        # Point count and shrink controls
        controls = ZonePointsUIBuilder.create_controls_layout(self)
        for i in range(controls.count()):
            item = controls.itemAt(i)
            if item:
                layout.addWidget(item.widget())

        # Coordinate controls
        coord_controls = ZonePointsUIBuilder.create_coordinate_controls(self)
        for i in range(coord_controls.count()):
            item = coord_controls.itemAt(i)
            if item:
                layout.addWidget(item.widget())

        # Action buttons
        button_controls = ZonePointsUIBuilder.create_action_buttons(self)
        for i in range(button_controls.count()):
            item = button_controls.itemAt(i)
            if item:
                layout.addWidget(item.widget())

        return layout

    def _setup_event_handlers(self):
        """Set up all event handlers for the UI components."""
        self.generatePointsBtn.clicked.connect(self.generate_zone_points)
        self.updateCoordinatesBtn.clicked.connect(self.update_coordinates)
        self.pointsTable.selectionModel().selectionChanged.connect(
            self.on_points_table_selection
        )
        self.pointsTable.installEventFilter(self)

    def update_conversion_label(self):
        self.conversionLabel.setText(f"Conversion: {self.pixel_to_mm_ratio:.2f} px/mm")

    def generate_zone_points(self):
        """Main method to generate zone points."""
        self._reset_point_counter()

        # Get parameters from UI
        n_points = self.pointCountSpinBox.value()
        shrink_percent = self.shrinkSpinBox.value()
        shrink_factor = (100 - shrink_percent) / 100.0

        # Get shapes for inclusion and exclusion
        include_shape, exclude_shapes = self._get_inclusion_exclusion_shapes()
        if include_shape is None:
            print("No include shape defined. Cannot generate points.")
            return

        # Generate candidate points
        candidates, area = self._generate_candidate_points(
            include_shape, exclude_shapes, shrink_factor
        )
        if not candidates:
            print("No candidate points found in allowed region.")
            return

        # Sample final points and compute ideal radius
        final_points = farthest_point_sampling(candidates, n_points)
        ideal_radius = compute_ideal_radius(
            area * len(candidates) / ZonePointsConstants.MAX_CANDIDATES,
            n_points,
        )

        # Clear existing generated points and render new ones
        self._clear_generated_points()
        self._render_generated_points(final_points, ideal_radius)

        self.update_points_table()

    def _reset_point_counter(self):
        """Reset the point ID counter."""
        if not hasattr(self, "next_point_id"):
            self.next_point_id = 1
        else:
            self.next_point_id = 1

    def _get_inclusion_exclusion_shapes(
        self,
    ) -> Tuple[Optional[Any], List[Any]]:
        """Get inclusion and exclusion shapes from the image view."""
        include_shape = None
        exclude_shapes = []

        for shape in self.image_view.shapes:
            role = shape.get("role", "include")
            if role == "include":
                include_shape = shape["item"]
            elif role == "exclude":
                exclude_shapes.append(shape["item"])

        return include_shape, exclude_shapes

    def _generate_candidate_points(
        self, include_shape, exclude_shapes: List, shrink_factor: float
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Generate and filter candidate points based on shapes."""
        # Get initial candidates and area using geometry helper
        candidates, area, bounds = ZonePointsGeometry.get_shape_bounds_and_candidates(
            include_shape, shrink_factor
        )

        # Filter candidates by inclusion/exclusion shapes
        filtered_candidates = ZonePointsGeometry.filter_candidates_by_shapes(
            candidates, include_shape, exclude_shapes
        )

        return filtered_candidates, area

    def _clear_generated_points(self):
        """Clear all existing generated points and zones from the scene."""
        for item in self.image_view.points_dict["generated"]["points"]:
            self.safe_remove_item(item)
        for item in self.image_view.points_dict["generated"]["zones"]:
            self.safe_remove_item(item)

        self.image_view.points_dict["generated"]["points"].clear()
        self.image_view.points_dict["generated"]["zones"].clear()

    def _render_generated_points(
        self, points: List[Tuple[float, float]], ideal_radius: float
    ):
        """Render the generated points and zones on the scene."""
        for x, y in points:
            # Create and add zone (background circle)
            zone_item = ZonePointsRenderer.create_zone_item(x, y, ideal_radius)
            self.image_view.scene.addItem(zone_item)
            self.image_view.points_dict["generated"]["zones"].append(zone_item)

            # Create and add point (foreground dot)
            point_item = ZonePointsRenderer.create_point_item(
                x, y, self.next_point_id, "generated"
            )
            self.next_point_id += 1
            self.image_view.scene.addItem(point_item)
            self.image_view.points_dict["generated"]["points"].append(point_item)

    # --- Table and selection methods remain as before, with attribute checks as needed ---
    def update_coordinates(self):
        self.update_points_table()

    def safe_remove_item(self, item):
        try:
            if item in self.image_view.scene.items():
                self.image_view.scene.removeItem(item)
        except Exception as e:
            print(f"Error removing item: {e}")

    def on_points_table_selection(self, selected, deselected):
        """Handle table row selection by highlighting corresponding points in the scene."""
        # Skip if we're in the middle of updating the table to avoid re-entrancy issues
        if getattr(self, "_updating_points_table", False):
            return
        # Reset all points to their default colors
        self._reset_all_point_styles()

        # Highlight selected points
        self._highlight_selected_points()

    def _reset_all_point_styles(self):
        """Reset all points to their default colors."""

        def reset_point_style(item, point_type: str):
            if sip.isdeleted(item):
                return
            color = (
                ZonePointsConstants.POINT_COLOR_GENERATED
                if point_type == "generated"
                else ZonePointsConstants.POINT_COLOR_USER
            )
            item.setBrush(QColor(color))

        # Reset generated points
        for item in self.image_view.points_dict["generated"]["points"]:
            reset_point_style(item, "generated")

        # Reset user points
        for item in self.image_view.points_dict["user"]["points"]:
            reset_point_style(item, "user")

    def _highlight_selected_points(self):
        """Highlight points corresponding to selected table rows."""
        for index in self.pointsTable.selectionModel().selectedRows():
            row = index.row()
            n_generated = len(self.image_view.points_dict["generated"]["points"])

            if row < n_generated:
                # Selected row corresponds to a generated point
                item = self.image_view.points_dict["generated"]["points"][row]
                item.setBrush(ZonePointsConstants.POINT_COLOR_SELECTED)
            else:
                # Selected row corresponds to a user point
                user_row = row - n_generated
                if user_row < len(self.image_view.points_dict["user"]["points"]):
                    item = self.image_view.points_dict["user"]["points"][user_row]
                    item.setBrush(ZonePointsConstants.POINT_COLOR_SELECTED)

    def eventFilter(self, source, event):
        # Safety check: ensure pointsTable exists before comparing
        if (
            hasattr(self, "pointsTable")
            and source == self.pointsTable
            and event.type() == QEvent.KeyPress
        ):
            if event.key() == Qt.Key_Delete:
                self.delete_selected_points()
                return True
        return super().eventFilter(source, event)

    def update_points_table_safe(self):
        """Minimal safe table update for restore operations (no widgets)."""
        try:
            if not hasattr(self, "pointsTable") or self.pointsTable is None:
                print("Info: pointsTable not available for safe update")
                return

            # Guard against re-entrancy and selection signals
            self._updating_points_table = True
            try:
                self.pointsTable.blockSignals(True)
                points = self._build_points_snapshot()
                self.pointsTable.setRowCount(len(points))

                for idx, (x, y, ptype, point_id) in enumerate(points):
                    from PyQt5.QtWidgets import QTableWidgetItem

                    self.pointsTable.setItem(
                        idx,
                        0,
                        QTableWidgetItem("" if point_id is None else str(point_id)),
                    )
                    self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
                    self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
                    self.pointsTable.setItem(idx, 3, QTableWidgetItem("N/A"))
                    self.pointsTable.setItem(idx, 4, QTableWidgetItem("N/A"))
            finally:
                self.pointsTable.blockSignals(False)
                self._updating_points_table = False

            print(f"Safe table update completed with {len(points)} points")

        except Exception as e:
            print(f"Error in safe table update: {e}")

    def update_points_table(self):
        """Update the points table with current point data and measurement widgets."""
        try:
            # Safety check - ensure we have the required attributes
            if not hasattr(self, "pointsTable") or self.pointsTable is None:
                print("Info: pointsTable is not initialized, skipping table update")
                return

            # Skip if zone points widget hasn't been created yet
            if not hasattr(self, "zonePointsDock"):
                print("Info: Zone points widget not created yet, using safe update")
                self.update_points_table_safe()
                return

            # Check if we have the measurement_widgets attribute
            if not hasattr(self, "measurement_widgets"):
                self.measurement_widgets = {}

            # Guard against re-entrancy and selection signals during updates
            self._updating_points_table = True
            try:
                self.pointsTable.blockSignals(True)

                # 1) Build the current points snapshot
                points = self._build_points_snapshot()

                # 3) Clean up deleted measurement widgets
                self._cleanup_deleted_widgets(points)

                # 4) Set the table row count and populate rows
                self.pointsTable.setRowCount(len(points))

                # 5) Populate table rows and reattach widgets
                self._populate_table_rows(points)
            finally:
                self.pointsTable.blockSignals(False)
                self._updating_points_table = False

            print(
                f"Updated table with {len(points)} points. Widget keys: {list(self.measurement_widgets.keys())}"
            )

        except Exception as e:
            print(f"Error updating points table: {e}")
            import traceback

            traceback.print_exc()

    def _build_points_snapshot(
        self,
    ) -> List[Tuple[float, float, str, Optional[int]]]:
        """Build a snapshot of all current points with their data."""
        points = []

        # Safety check - ensure image_view and points_dict exist
        if not hasattr(self, "image_view") or not hasattr(
            self.image_view, "points_dict"
        ):
            print("Warning: image_view or points_dict not available")
            return points

        try:
            # Generated points
            for item in self.image_view.points_dict["generated"]["points"]:
                try:
                    if item is None or sip.isdeleted(item):
                        continue
                    c = item.sceneBoundingRect().center()
                    pid = item.data(1)
                    points.append(
                        (
                            c.x(),
                            c.y(),
                            "generated",
                            int(pid) if pid is not None else None,
                        )
                    )
                except Exception as e:
                    print(f"Error processing generated point: {e}")
                    continue

            # User points
            for item in self.image_view.points_dict["user"]["points"]:
                try:
                    if item is None or sip.isdeleted(item):
                        continue
                    c = item.sceneBoundingRect().center()
                    pid = item.data(1)
                    points.append(
                        (
                            c.x(),
                            c.y(),
                            "user",
                            int(pid) if pid is not None else None,
                        )
                    )
                except Exception as e:
                    print(f"Error processing user point: {e}")
                    continue

        except Exception as e:
            print(f"Error building points snapshot: {e}")

        return points

    def _cleanup_deleted_widgets(
        self, points: List[Tuple[float, float, str, Optional[int]]]
    ):
        """Clean up measurement widgets for points that no longer exist."""
        current_point_ids = {pid for (_, _, _, pid) in points if pid is not None}

        # Remove widgets for deleted points
        for pid in list(self.measurement_widgets.keys()):
            if pid not in current_point_ids:
                widget = self.measurement_widgets.pop(pid)
                if widget and not sip.isdeleted(widget):
                    widget.setParent(None)
                    widget.deleteLater()
                    print(f"Cleaned up widget for deleted point ID {pid}")

    def _populate_table_rows(
        self, points: List[Tuple[float, float, str, Optional[int]]]
    ):
        """Populate table rows with point data and reattach measurement widgets."""
        for idx, (x, y, ptype, point_id) in enumerate(points):
            # Set basic point data
            self.pointsTable.setItem(
                idx,
                0,
                QTableWidgetItem("" if point_id is None else str(point_id)),
            )
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))

            # Set coordinate data
            if self.pixel_to_mm_ratio:
                x_mm = (
                    self.real_x_pos_mm.value()
                    - (x - self.include_center[0]) / self.pixel_to_mm_ratio
                )
                y_mm = (
                    self.real_y_pos_mm.value()
                    - (y - self.include_center[1]) / self.pixel_to_mm_ratio
                )
                self.pointsTable.setItem(idx, 3, QTableWidgetItem(f"{x_mm:.2f}"))
                self.pointsTable.setItem(idx, 4, QTableWidgetItem(f"{y_mm:.2f}"))
            else:
                self.pointsTable.setItem(idx, 3, QTableWidgetItem("N/A"))
                self.pointsTable.setItem(idx, 4, QTableWidgetItem("N/A"))

            # Do not attach measurement widgets in the table anymore. They live in the right panel.

    def _attach_measurement_widget(self, row_index: int, point_id: Optional[int]):
        """Deprecated for table. Measurement widgets are managed in the right panel."""
        if point_id is None:
            return
        # No-op: widgets are added via add_measurement_widget_to_panel

    def _create_measurement_widget(self, point_id: int) -> Any:
        """Create a new measurement widget for a point."""
        return MeasurementHistoryWidget(
            masks=getattr(self, "masks", {}),
            ponis=getattr(self, "ponis", {}),
            parent=self,
            point_id=point_id,
        )

    def add_measurement_widget_to_panel(self, point_id: int):
        """Add a measurement widget for a point to the right tree (if not exists)."""
        if getattr(self, "_restoring_state", False):
            return
        # If already exists, do nothing
        if point_id in self._measurement_items:
            top_item, child_item, w = self._measurement_items.get(
                point_id, (None, None, None)
            )
            if w is not None and not sip.isdeleted(w):
                return
        # Create tree items
        top_item = QTreeWidgetItem(self.measurementsTree, [f"Point #{point_id}"])
        child_item = QTreeWidgetItem(top_item, [""])
        self.measurementsTree.addTopLevelItem(top_item)
        top_item.setExpanded(True)
        # Create widget and place into child row, column 0
        w = self._create_measurement_widget(point_id)
        self.measurementsTree.setItemWidget(child_item, 0, w)
        self.measurement_widgets[point_id] = w
        self._measurement_items[point_id] = (top_item, child_item, w)

    def remove_measurement_widget_from_panel(self, point_id: int):
        """Remove the measurement widget and its items from the tree."""
        top_item, child_item, w = self._measurement_items.pop(
            point_id, (None, None, None)
        )
        if w and not sip.isdeleted(w):
            try:
                # Detach from tree cell
                self.measurementsTree.setItemWidget(child_item, 0, None)
            except Exception:
                pass
            w.setParent(None)
            w.deleteLater()
        if top_item is not None:
            try:
                index = self.measurementsTree.indexOfTopLevelItem(top_item)
                if index != -1:
                    self.measurementsTree.takeTopLevelItem(index)
            except Exception:
                pass
        self.measurement_widgets.pop(point_id, None)

    def delete_selected_points(self):
        """Delete selected points and preserve measurement widget history."""
        # 1) Get selected rows and extract point IDs to delete
        selected_rows = sorted(
            {ix.row() for ix in self.pointsTable.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        # 2) Collect point IDs that will be deleted
        pids_to_delete = set()
        for r in selected_rows:
            id_item = self.pointsTable.item(r, 0)
            if id_item is not None:
                try:
                    pid = int(id_item.text())
                    pids_to_delete.add(pid)
                except (ValueError, TypeError):
                    pass

        print(f"Deleting point IDs: {pids_to_delete}")

        # 3) Remove points from scene by ID (not by index to avoid shifting issues)
        for pid in pids_to_delete:
            self._remove_point_items_by_id(pid)

        # 4) Remove deleted point IDs from measurement widgets (right panel)
        for pid in pids_to_delete:
            self.remove_measurement_widget_from_panel(pid)

        # 5) Rebuild the table with remaining points
        self.update_points_table()

    def delete_all_points(self):
        for item in self.image_view.points_dict["generated"]["points"]:
            self.safe_remove_item(item)
        for item in self.image_view.points_dict["generated"]["zones"]:
            self.safe_remove_item(item)
        self.image_view.points_dict["generated"]["points"].clear()
        self.image_view.points_dict["generated"]["zones"].clear()
        for item in self.image_view.points_dict["user"]["points"]:
            self.safe_remove_item(item)
        for item in self.image_view.points_dict["user"]["zones"]:
            self.safe_remove_item(item)
        self.image_view.points_dict["user"]["points"].clear()
        self.image_view.points_dict["user"]["zones"].clear()
        # Clear measurement tree
        for pid in list(getattr(self, "_measurement_items", {}).keys()):
            self.remove_measurement_widget_from_panel(pid)
        self.measurement_widgets = {}
        self.next_point_id = 1
        self.update_points_table()

    def _remove_point_items_by_id(self, point_id):
        # Try generated first
        gp = self.image_view.points_dict["generated"]["points"]
        gz = self.image_view.points_dict["generated"]["zones"]
        for i, item in enumerate(gp):
            if not sip.isdeleted(item) and item.data(1) == point_id:
                # remove both point and its matching zone
                point_item = gp.pop(i)
                zone_item = gz.pop(i) if i < len(gz) else None
                if zone_item:
                    self.safe_remove_item(zone_item)
                self.safe_remove_item(point_item)
                return

        # Then user points
        up = self.image_view.points_dict["user"]["points"]
        uz = self.image_view.points_dict["user"]["zones"]
        for i, item in enumerate(up):
            if not sip.isdeleted(item) and item.data(1) == point_id:
                point_item = up.pop(i)
                zone_item = uz.pop(i) if i < len(uz) else None
                if zone_item:
                    self.safe_remove_item(zone_item)
                self.safe_remove_item(point_item)
                return

    def _snapshot_history_widgets(self):
        """Return {point_id: [measurement_dict, ...]} from existing widgets."""
        snap = {}
        for pid, w in list(getattr(self, "measurement_widgets", {}).items()):
            if w is not None and not sip.isdeleted(w):
                snap[pid] = list(getattr(w, "measurements", []))
        return snap
