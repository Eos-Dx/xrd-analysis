"""Main zone points extension functionality."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import sip
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QSplitter,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hardware.difra.gui.technical.widgets import MeasurementHistoryWidget

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
        self.zonePointsDock.setObjectName("ZonePointsDock")
        container = QWidget()
        
        # Set smaller font for all controls to fit smaller screens
        try:
            from PyQt5.QtGui import QFont
            control_font = QFont()
            control_font.setPointSize(9)  # Smaller font for controls (menu-size)
            container.setFont(control_font)
        except Exception:
            pass
        
        layout = QVBoxLayout(container)
        # Tighten margins/spacing to reduce vertical footprint
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # Create UI components using helper classes (packed into a compact bar)
        controls_layout = self._create_all_controls()
        try:
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(6)
        except Exception:
            pass
        from PyQt5.QtWidgets import QSizePolicy

        controls_bar = QWidget()
        controls_bar.setLayout(controls_layout)
        controls_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        try:
            controls_bar.setMaximumHeight(32)  # compact toolbar-like bar
        except Exception:
            pass
        layout.addWidget(controls_bar)

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
        
        # Set minimum height to be compact - just enough for toolbar and a few table rows
        # This gives more vertical space to other zones (image view, etc.)
        try:
            # Minimum: title bar (~20px) + compact toolbar (~32px) + 2-3 table rows (~80px)
            self.zonePointsDock.setMinimumHeight(130)
        except Exception:
            pass
        
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)
        try:
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
        except Exception:
            pass

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
        self.pointsTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pointsTable.customContextMenuRequested.connect(
            self._show_points_table_context_menu
        )

    def update_conversion_label(self):
        self.conversionLabel.setText(f"Conversion: {self.pixel_to_mm_ratio:.2f} px/mm")

    def generate_zone_points(self):
        """Main method to generate zone points."""
        self._reset_point_counter()

        # Get parameters from UI
        n_points = self.pointCountSpinBox.value()
        shrink_percent = self.shrinkSpinBox.value()
        shrink_factor = (100 - shrink_percent) / 100.0
        # Keep generated point centers visually away from the include border.
        edge_clearance_px = float(ZonePointsConstants.POINT_RADIUS)

        # Get shapes for inclusion and exclusion
        include_shape, exclude_shapes = self._get_inclusion_exclusion_shapes()
        if include_shape is None:
            print("No include shape defined. Cannot generate points.")
            return

        # Generate candidate points
        candidates, area = self._generate_candidate_points(
            include_shape, exclude_shapes, shrink_factor, edge_clearance_px
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
        self,
        include_shape,
        exclude_shapes: List,
        shrink_factor: float,
        edge_clearance_px: float = 0.0,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Generate and filter candidate points based on shapes."""
        # Get initial candidates and area using geometry helper
        candidates, area, bounds = ZonePointsGeometry.get_shape_bounds_and_candidates(
            include_shape,
            shrink_factor,
            edge_clearance_px=edge_clearance_px,
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
            point_id = self.next_point_id
            point_uid = self._new_point_uid(point_id)
            point_item = ZonePointsRenderer.create_point_item(
                x, y, point_id, "generated", point_uid=point_uid
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

    def _measurement_sequence_active(self) -> bool:
        try:
            return (
                int(getattr(self, "total_points", 0)) > 0
                and hasattr(self, "start_btn")
                and not self.start_btn.isEnabled()
            )
        except Exception:
            return False

    def _show_points_table_context_menu(self, pos):
        if not hasattr(self, "pointsTable") or self.pointsTable is None:
            return

        menu = QMenu(self.pointsTable)
        delete_action = menu.addAction("Delete Selected Point(s)")
        skip_action = menu.addAction("Mark Selected as Skipped...")
        chosen = menu.exec_(self.pointsTable.viewport().mapToGlobal(pos))
        if chosen == delete_action:
            self.delete_selected_points()
        elif chosen == skip_action:
            self.mark_selected_points_skipped()

    def _prompt_skip_reason(self, title: str, prompt: str) -> Optional[str]:
        reason, ok = QInputDialog.getText(self, title, prompt)
        if not ok:
            return None
        return str(reason or "").strip() or "user_skipped"

    def _find_sorted_position_for_row(self, row: int) -> Optional[int]:
        sorted_indices = list(getattr(self, "sorted_indices", []) or [])
        for pos, idx in enumerate(sorted_indices):
            if int(idx) == int(row):
                return pos
        return None

    def _session_point_index_for_row(self, row: int) -> int:
        pos = self._find_sorted_position_for_row(row)
        mapped = getattr(self, "_session_point_indices", None)
        if pos is not None and isinstance(mapped, (list, tuple)) and pos < len(mapped):
            try:
                return int(mapped[pos])
            except Exception:
                pass
        return int(row) + 1

    def _point_has_measurements(self, point_id: Optional[int]) -> bool:
        if point_id is None:
            return False
        widget = getattr(self, "measurement_widgets", {}).get(point_id)
        if widget is None:
            return False
        try:
            return len(getattr(widget, "measurements", []) or []) > 0
        except Exception:
            return False

    def _is_row_measured(self, row: int, point_id: Optional[int]) -> bool:
        if self._point_has_measurements(point_id):
            return True

        sorted_pos = self._find_sorted_position_for_row(row)
        if sorted_pos is not None:
            try:
                return sorted_pos < int(getattr(self, "current_measurement_sorted_index", 0))
            except Exception:
                return False
        return False

    def _append_skipped_point_record(
        self, row: int, point_id: Optional[int], reason: str
    ) -> None:
        x_mm = None
        y_mm = None
        try:
            x_item = self.pointsTable.item(row, 3)
            y_item = self.pointsTable.item(row, 4)
            if x_item is not None:
                txt = str(x_item.text() or "").strip()
                if txt and txt != "N/A":
                    x_mm = float(txt)
            if y_item is not None:
                txt = str(y_item.text() or "").strip()
                if txt and txt != "N/A":
                    y_mm = float(txt)
        except Exception:
            pass

        payload = {
            "point_index": int(row),
            "point_id": int(point_id) if point_id is not None else None,
            "x": x_mm,
            "y": y_mm,
            "reason": str(reason),
        }
        for container_name in ("state", "state_measurements"):
            container = getattr(self, container_name, None)
            if not isinstance(container, dict):
                continue
            skipped = list(container.get("skipped_points", []) or [])
            skipped = [
                item
                for item in skipped
                if int(item.get("point_index", -1)) != int(row)
            ]
            skipped.append(dict(payload))
            container["skipped_points"] = skipped

    def _apply_skipped_visual(self, point_id: Optional[int]) -> None:
        if point_id is None:
            return

        skip_point_color = QColor(255, 165, 0)
        skip_zone_color = QColor(255, 165, 0)
        skip_zone_color.setAlphaF(0.18)

        gp = self.image_view.points_dict["generated"]["points"]
        gz = self.image_view.points_dict["generated"]["zones"]
        up = self.image_view.points_dict["user"]["points"]
        uz = self.image_view.points_dict["user"]["zones"]

        for i, item in enumerate(gp):
            if sip.isdeleted(item):
                continue
            if item.data(1) == point_id:
                item.setBrush(skip_point_color)
                if i < len(gz) and not sip.isdeleted(gz[i]):
                    gz[i].setBrush(skip_zone_color)
                return

        for i, item in enumerate(up):
            if sip.isdeleted(item):
                continue
            if item.data(1) == point_id:
                item.setBrush(skip_point_color)
                if i < len(uz) and not sip.isdeleted(uz[i]):
                    uz[i].setBrush(skip_zone_color)
                return

    def _remove_row_from_active_measurement_plan(self, row: int) -> None:
        sorted_indices = list(getattr(self, "sorted_indices", []) or [])
        pos = self._find_sorted_position_for_row(row)
        if pos is not None and 0 <= pos < len(sorted_indices):
            sorted_indices.pop(pos)
        for idx, value in enumerate(sorted_indices):
            if int(value) > int(row):
                sorted_indices[idx] = int(value) - 1
        self.sorted_indices = sorted_indices

        mapped = getattr(self, "_session_point_indices", None)
        if isinstance(mapped, list) and pos is not None and 0 <= pos < len(mapped):
            mapped.pop(pos)

        for container_name in ("state", "state_measurements"):
            container = getattr(self, container_name, None)
            if not isinstance(container, dict):
                continue
            points = container.get("measurement_points", None)
            if not isinstance(points, list):
                continue
            if pos is not None and 0 <= pos < len(points):
                points.pop(pos)
            for point in points:
                try:
                    pidx = int(point.get("point_index", -1))
                    if pidx > int(row):
                        point["point_index"] = pidx - 1
                except Exception:
                    continue

        current_idx = int(getattr(self, "current_measurement_sorted_index", 0))
        if pos is not None and pos < current_idx:
            current_idx -= 1
        if current_idx < 0:
            current_idx = 0
        self.current_measurement_sorted_index = current_idx
        self.total_points = len(self.sorted_indices)

        try:
            self.progressBar.setMaximum(self.total_points)
            self.progressBar.setValue(min(self.current_measurement_sorted_index, self.total_points))
        except Exception:
            pass

        if self.total_points <= 0:
            if hasattr(self, "_append_capture_log"):
                self._append_capture_log("Measurement sequence complete")
            if hasattr(self, "_set_measurement_controls_idle"):
                self._set_measurement_controls_idle()
            elif hasattr(self, "start_btn"):
                self.start_btn.setEnabled(True)

    def _skip_point_by_row(self, row: int, reason: str) -> bool:
        reason = str(reason or "").strip() or "user_skipped"
        sorted_pos = self._find_sorted_position_for_row(row)
        current_idx = int(getattr(self, "current_measurement_sorted_index", 0))
        is_current = sorted_pos is not None and sorted_pos == current_idx

        capture_thread = getattr(self, "capture_thread", None)
        if is_current and capture_thread is not None and hasattr(capture_thread, "isRunning"):
            try:
                if capture_thread.isRunning():
                    QMessageBox.warning(
                        self,
                        "Skip Busy Point",
                        "Current point capture is already running. Skip it after capture finishes.",
                    )
                    return False
            except Exception:
                pass

        point_id = None
        id_item = self.pointsTable.item(row, 0)
        if id_item is not None:
            try:
                point_id = int(id_item.text())
            except Exception:
                point_id = None

        session_point_index = self._session_point_index_for_row(row)
        session_manager = getattr(self, "session_manager", None)
        if (
            session_manager is not None
            and hasattr(session_manager, "is_session_active")
            and session_manager.is_session_active()
            and hasattr(session_manager, "mark_point_skipped")
        ):
            try:
                session_manager.mark_point_skipped(
                    point_index=session_point_index,
                    reason=reason,
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Skip Failed",
                    f"Failed to mark point as skipped in session container:\n{exc}",
                )
                return False

        self._append_skipped_point_record(row=row, point_id=point_id, reason=reason)
        self._apply_skipped_visual(point_id)

        if self._measurement_sequence_active():
            self._remove_row_from_active_measurement_plan(row)
            if (
                is_current
                and not bool(getattr(self, "paused", False))
                and not bool(getattr(self, "stopped", False))
                and int(getattr(self, "total_points", 0)) > int(getattr(self, "current_measurement_sorted_index", 0))
                and hasattr(self, "measure_next_point")
            ):
                self.measure_next_point()

        if hasattr(self, "_append_measurement_log"):
            self._append_measurement_log(f"[CAPTURE] Point skipped (reason: {reason})")
        return True

    def _delete_row_and_container_point(self, row: int, point_id: Optional[int]) -> bool:
        if point_id is None:
            return False

        sorted_pos = self._find_sorted_position_for_row(row)
        current_idx = int(getattr(self, "current_measurement_sorted_index", 0))
        is_current = sorted_pos is not None and sorted_pos == current_idx
        capture_thread = getattr(self, "capture_thread", None)
        if is_current and capture_thread is not None and hasattr(capture_thread, "isRunning"):
            try:
                if capture_thread.isRunning():
                    QMessageBox.warning(
                        self,
                        "Delete Busy Point",
                        "Current point capture is already running and cannot be deleted now.",
                    )
                    return False
            except Exception:
                pass

        session_manager = getattr(self, "session_manager", None)
        if (
            session_manager is not None
            and hasattr(session_manager, "is_session_active")
            and session_manager.is_session_active()
            and hasattr(session_manager, "delete_point")
        ):
            session_point_index = self._session_point_index_for_row(row)
            try:
                deleted = bool(session_manager.delete_point(point_index=session_point_index))
                if not deleted:
                    # Point may not be seeded into container yet (e.g. before first Start).
                    pass
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Delete Failed",
                    f"Cannot delete this point from the session container:\n{exc}",
                )
                return False

        if self._measurement_sequence_active():
            self._remove_row_from_active_measurement_plan(row)
            if (
                is_current
                and not bool(getattr(self, "paused", False))
                and not bool(getattr(self, "stopped", False))
                and int(getattr(self, "total_points", 0)) > int(getattr(self, "current_measurement_sorted_index", 0))
                and hasattr(self, "measure_next_point")
            ):
                self.measure_next_point()

        self._remove_point_items_by_id(point_id)
        self.remove_measurement_widget_from_panel(point_id)
        return True

    def _request_delete_point_by_id(self, point_id: int) -> bool:
        row = None
        if hasattr(self, "pointsTable") and self.pointsTable is not None:
            for idx in range(self.pointsTable.rowCount()):
                item = self.pointsTable.item(idx, 0)
                if item is None:
                    continue
                try:
                    if int(item.text()) == int(point_id):
                        row = idx
                        break
                except Exception:
                    continue

        if row is None:
            return False

        measured = self._is_row_measured(row=row, point_id=point_id)
        if measured:
            reason = self._prompt_skip_reason(
                "Point Already Measured",
                "This point is already measured and cannot be deleted.\n"
                "Provide skip reason to mark it as SKIPPED:",
            )
            if reason is None:
                return False
            changed = self._skip_point_by_row(row=row, reason=reason)
            if changed:
                self.update_points_table()
            return changed

        changed = self._delete_row_and_container_point(row=row, point_id=point_id)
        if changed:
            self.update_points_table()
        return changed

    def mark_selected_points_skipped(self):
        selected_rows = sorted(
            {ix.row() for ix in self.pointsTable.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        reason = self._prompt_skip_reason("Skip Selected Points", "Skip reason:")
        if reason is None:
            return

        changed_any = False
        for row in selected_rows:
            changed_any = self._skip_point_by_row(row=row, reason=reason) or changed_any

        if changed_any:
            self.update_points_table()

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
        """Delete selected points, enforcing measured/skipped rules."""
        selected_rows = sorted(
            {ix.row() for ix in self.pointsTable.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        active_measurement = self._measurement_sequence_active()
        skip_reason_for_measured = None
        changed_any = False

        for r in selected_rows:
            id_item = self.pointsTable.item(r, 0)
            if id_item is None:
                continue
            try:
                pid = int(id_item.text())
            except (ValueError, TypeError):
                continue

            measured = self._is_row_measured(row=r, point_id=pid)
            if measured:
                if skip_reason_for_measured is None:
                    skip_reason_for_measured = self._prompt_skip_reason(
                        "Measured Point",
                        "Measured points cannot be deleted.\n"
                        "Provide reason to mark selected measured point(s) as SKIPPED:",
                    )
                    if skip_reason_for_measured is None:
                        continue
                changed_any = (
                    self._skip_point_by_row(
                        row=r,
                        reason=skip_reason_for_measured,
                    )
                    or changed_any
                )
                continue

            changed_any = (
                self._delete_row_and_container_point(row=r, point_id=pid)
                or changed_any
            )

            # When measuring is active, removing any pending point from the plan
            # effectively "deletes" it from upcoming sequence.
            if active_measurement and hasattr(self, "_append_measurement_log"):
                self._append_measurement_log(f"[CAPTURE] Point #{pid} deleted from pending plan")

        if changed_any:
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
    @staticmethod
    def _new_point_uid(counter: int) -> str:
        try:
            counter_int = int(counter)
        except Exception:
            counter_int = 0
        return f"{counter_int}_{uuid.uuid4().hex[:8]}"
