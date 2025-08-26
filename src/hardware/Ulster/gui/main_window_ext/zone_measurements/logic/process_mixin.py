# zone_measurements/logic/process_mixin.py

import hashlib
import json
import time
from copy import copy
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

from hardware.Ulster.gui.technical.capture import CaptureWorker, validate_folder
from hardware.Ulster.gui.technical.measurement_worker import MeasurementWorker
from hardware.Ulster.gui.technical.widgets import MeasurementHistoryWidget
from hardware.Ulster.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class ZoneMeasurementsProcessMixin:
    def start_measurements(self):
        """
        Starts the measurements for all sorted points.
        Prepares measurement folder, state, sorts points, starts progress bar, etc.
        """

        self.manual_save_state()  # Save current state before starting measurements
        # Folder validation and state saving
        self.measurement_folder = Path(self.folderLineEdit.text().strip())
        self.state_path_measurements = (
            self.measurement_folder / f"{self.fileNameLineEdit.text()}_state.json"
        )

        # ===== FOLDER EXISTENCE CHECK =====
        if not self.measurement_folder.exists():
            # Show a dialog or message box (PyQt5 example)
            QMessageBox.warning(
                self,
                "Folder Error",
                "Selected folder does not exist. Please select the correct folder.",
            )
            return  # Exit the function early
        # ==================================

        try:
            self.state_measurements = copy(self.state)
        except Exception as e:
            logger.error("Error copying state for measurements", error=str(e))
            QMessageBox.warning(self, "No state", "Save it.")
            return  # Exit the function early

        try:
            from hardware.Ulster.hardware.auxiliary import encode_image_to_base64

            self.state_measurements["image_base64"] = encode_image_to_base64(
                self.image_view.current_image_path
            )
            with open(self.state_path_measurements, "w") as f:
                import json

                json.dump(self.state_measurements, f, indent=4)
        except Exception as e:
            logger.error("Error saving state with encoded image", error=str(e))

        if self.pointsTable.rowCount() == 0:
            logger.warning("No points available for measurement")
            return

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stopped = False
        self.paused = False

        # Consolidate and sort measurement points
        generated_points = self.image_view.points_dict["generated"]["points"]
        user_points = self.image_view.points_dict["user"]["points"]
        all_points = []
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            x_mm = (
                self.real_x_pos_mm.value()
                - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            )
            y_mm = (
                self.real_y_pos_mm.value()
                - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            )
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = (
                self.real_x_pos_mm.value()
                - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            )
            y_mm = (
                self.real_y_pos_mm.value()
                - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            )
            all_points.append((offset + j, x_mm, y_mm))
        all_points_sorted = sorted(all_points, key=lambda tup: (tup[1], tup[2]))
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        self.current_measurement_sorted_index = 0

        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        self.integration_time = self.integrationSpinBox.value()
        self.initial_estimate = self.total_points * self.integration_time
        self.measurementStartTime = time.time()
        self.timeRemainingLabel.setText(
            f"Estimated time: {self.initial_estimate:.0f} sec"
        )
        logger.info(
            "Starting measurements in sorted order",
            total_points=self.total_points,
            integration_time=self.integration_time,
        )

        # Filter out-of-bounds points and create measurement list using controller limits
        try:
            limits = (
                self.stage_controller.get_limits()
                if hasattr(self, "stage_controller")
                else None
            )
        except Exception:
            limits = None
        # Fallback to defaults if controller is unavailable
        if not limits:
            limits = {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}
        x_min, x_max = limits["x"]
        y_min, y_max = limits["y"]

        measurement_points = []
        skipped_points = []
        valid_idx = 0

        for orig_idx, (pt_idx, x_mm, y_mm) in enumerate(all_points_sorted):
            # Check if point is within axis limits
            if (x_min <= x_mm <= x_max) and (y_min <= y_mm <= y_max):
                # Point is valid - include in measurement
                id_str = f"{valid_idx}:{pt_idx}:{x_mm:.6f}:{y_mm:.6f}"
                unique_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()[:16]
                measurement_points.append(
                    {
                        "unique_id": unique_id,  # unique identifier for this point
                        "index": valid_idx,  # order of measurement
                        "point_index": pt_idx,  # original index
                        "x": x_mm,
                        "y": y_mm,
                        # Optionally: add more, e.g. type ("user" or "generated")
                    }
                )
                valid_idx += 1
            else:
                # Point is out of bounds - skip and log
                skipped_points.append((pt_idx, x_mm, y_mm))
                logger.warning(
                    f"Skipping measurement point {pt_idx} at ({x_mm:.3f}, {y_mm:.3f}) mm - "
                    f"outside limits X[{x_min:.1f},{x_max:.1f}] Y[{y_min:.1f},{y_max:.1f}] mm"
                )

        # Update sorted indices to only include valid points
        self.sorted_indices = [mp["point_index"] for mp in measurement_points]

        # Log summary of filtering
        if skipped_points:
            logger.info(
                f"Filtered measurement points: {len(measurement_points)} valid, "
                f"{len(skipped_points)} skipped due to axis limits"
            )

        # Check if we have any valid points left
        if not measurement_points:
            logger.error("No valid measurement points within axis limits")
            QMessageBox.warning(
                self,
                "No Valid Points",
                f"All measurement points exceed the axis limits of X[{x_min:.1f},{x_max:.1f}] and Y[{y_min:.1f},{y_max:.1f}] mm. "
                "Please adjust your measurement grid.",
            )
            return

        self.state["measurement_points"] = measurement_points
        # Also store skipped points for reference
        self.state["skipped_points"] = [
            {
                "point_index": pt_idx,
                "x": x_mm,
                "y": y_mm,
                "reason": "axis_limit_exceeded",
            }
            for pt_idx, x_mm, y_mm in skipped_points
        ]

        # Also save this in state_measurements if you use a copy
        self.state_measurements["measurement_points"] = measurement_points
        self.state_measurements["skipped_points"] = self.state["skipped_points"]
        self.manual_save_state()
        self.measure_next_point()

    def measure_next_point(self):
        """
        Moves to the next point and triggers capture on both detectors.
        Advances through sorted_indices and updates progress.
        """
        if self.stopped:
            logger.debug("Measurement stopped")
            return
        if self.paused:
            logger.debug("Measurement is paused. Waiting for resume")
            return
        if self.current_measurement_sorted_index >= self.total_points:
            logger.info("All points measured")
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            return

        index = self.sorted_indices[self.current_measurement_sorted_index]
        gp = self.image_view.points_dict["generated"]["points"]
        up = self.image_view.points_dict["user"]["points"]
        if index < len(gp):
            self._point_item = gp[index]
            self._zone_item = self.image_view.points_dict["generated"]["zones"][index]
        else:
            user_index = index - len(gp)
            self._point_item = up[user_index]
            self._zone_item = self.image_view.points_dict["user"]["zones"][user_index]

        self.update_xy_pos()
        center = self._point_item.sceneBoundingRect().center()
        self._x_mm = (
            self.real_x_pos_mm.value()
            - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
        )
        self._y_mm = (
            self.real_y_pos_mm.value()
            - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
        )

        # Move the stage using the controller
        new_x, new_y = self.stage_controller.move_stage(
            self._x_mm, self._y_mm, move_timeout=15
        )

        # Build a common filename base (without extension or detector label)
        import os
        import time

        self._timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._base_name = self.fileNameLineEdit.text().strip()
        txt_filename_base = os.path.join(
            self.measurement_folder,
            f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}",
        )

        # Launch the dual-capture worker in its own thread
        self.capture_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integration_time,
            txt_filename_base=txt_filename_base,
        )
        self.capture_thread = QThread()
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        self.capture_thread.start()

    def on_capture_finished(self, success: bool, result_files: dict):
        """
        Callback after detector(s) finish capturing.
        Handles errors, triggers post-processing, colors UI.
        Adds detector meta to measurements_meta for each measurement file.
        """
        if not success:
            logger.error("Measurement capture failed")
            return
        logger.info("Measurement capture successful", files=list(result_files.keys()))

        # Build detector meta as before
        detector_lookup = {d["alias"]: d for d in self.config["detectors"]}

        measurements = self.state_measurements.get("measurements_meta", {})
        measurement_points = self.state_measurements["measurement_points"]
        current_index = self.current_measurement_sorted_index
        x = self._x_mm
        y = self._y_mm
        point_unique_id = measurement_points[current_index]["unique_id"]

        for alias, txt_filename in result_files.items():
            detector_meta = detector_lookup.get(alias, {})
            measurements[Path(txt_filename).name] = {
                "x": x,
                "y": y,
                "unique_id": point_unique_id,  # <-- use the precomputed one!
                "base_file": self._base_name,
                "integration_time": self.integration_time,
                "detector_alias": alias,
                "detector_id": detector_meta.get("id"),
                "detector_type": detector_meta.get("type"),
                "detector_size": detector_meta.get("size"),
                "pixel_size_um": detector_meta.get("pixel_size_um"),
                "faulty_pixels": detector_meta.get("faulty_pixels"),
            }

        self.state_measurements["measurements_meta"] = measurements

        # Save updated state
        with open(self.state_path_measurements, "w") as f:
            json.dump(self.state_measurements, f, indent=4)

        # === The rest is unchanged (your logic) ===
        current_row = self.sorted_indices[self.current_measurement_sorted_index]
        self.spawn_measurement_thread(current_row, result_files)

        # Visual feedback
        green_brush = QColor(0, 255, 0)
        self._point_item.setBrush(green_brush)
        try:
            if self._zone_item:
                green_zone = QColor(0, 255, 0)
                green_zone.setAlphaF(0.2)
                self._zone_item.setBrush(green_zone)
        except Exception as e:
            logger.warning("Error updating zone item color", error=str(e))
        QTimer.singleShot(1000, self.measurement_finished)

    def spawn_measurement_thread(self, row, file_map):
        """
        Spawns a MeasurementWorker in a new thread for post-processing measurement files.
        Connects signals for result handling and thread cleanup.
        """
        thread = QThread(self)
        worker = MeasurementWorker(
            row=row,
            filenames=file_map,
            masks=self.masks,
            ponis=self.ponis,
            parent=self,
            hf_cutoff_fraction=0.2,
            columns_to_remove=30,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.measurement_ready.connect(self.add_measurement_to_table)
        worker.measurement_ready.connect(thread.quit)
        worker.measurement_ready.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        if not hasattr(self, "_measurement_threads"):
            self._measurement_threads = []
        self._measurement_threads.append((thread, worker))
        thread.start()

    def measurement_finished(self):
        """
        Called after one measurement completes.
        Advances progress, updates time estimates, and triggers next point if not done.
        """
        if self.stopped:
            logger.debug("Measurement stopped in measurement_finished")
            return

        self.current_measurement_sorted_index += 1
        self.progressBar.setValue(self.current_measurement_sorted_index)
        elapsed = time.time() - self.measurementStartTime
        if self.current_measurement_sorted_index > 0:
            avg_time = elapsed / self.current_measurement_sorted_index
            remaining = avg_time * (
                self.total_points - self.current_measurement_sorted_index
            )
            percent_complete = (
                self.current_measurement_sorted_index / self.total_points
            ) * 100
            self.timeRemainingLabel.setText(
                f"{percent_complete:.0f}% done, {remaining:.0f} sec remaining"
            )

        if (
            self.current_measurement_sorted_index < self.total_points
            and not self.paused
            and not self.stopped
        ):
            self.measure_next_point()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                logger.info("All measurement points completed")
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                self.start_btn.setEnabled(True)

    def add_measurement_to_table(self, row, results, timestamp=None):
        """Add measurement results to the appropriate point's widget (right panel, not the table).
        Also updates the widget title and the tree item text to include "#ID X:Y mm".
        """
        # --- Determine a stable point_id from table ---
        point_id = self._get_point_id_from_table_row(row)
        if point_id is None:
            logger.warning("Could not determine point_id for measurement", row=row)
            return

        # Extract X:Y in mm from the table row if available
        x_mm = None
        y_mm = None
        try:
            x_item = self.pointsTable.item(row, 3)
            y_item = self.pointsTable.item(row, 4)
            if x_item is not None and y_item is not None:
                x_mm = (
                    float(x_item.text())
                    if x_item.text() not in (None, "", "N/A")
                    else None
                )
                y_mm = (
                    float(y_item.text())
                    if y_item.text() not in (None, "", "N/A")
                    else None
                )
        except Exception:
            pass

        # Ensure a measurement widget exists in the right-side panel
        add_to_panel = getattr(self, "add_measurement_widget_to_panel", None)
        if callable(add_to_panel):
            add_to_panel(point_id)

        # --- Get or create the measurement widget (without using the table column) ---
        widget = self._get_or_create_measurement_widget(point_id)
        if widget is None:
            logger.error("Could not get/create measurement widget", point_id=point_id)
            return

        # Update widget title to include #ID and X:Y in mm, and store coords in widget
        try:
            if x_mm is not None and y_mm is not None:
                if hasattr(widget, "set_mm_coordinates"):
                    widget.set_mm_coordinates(x_mm, y_mm)
                else:
                    widget.setWindowTitle(
                        f"Measurement History: Point #{point_id} {x_mm:.2f}:{y_mm:.2f} mm"
                    )
            else:
                widget.setWindowTitle(f"Measurement History: Point #{point_id}")
        except Exception:
            pass

        # Update the tree item text to reflect the same
        try:
            items_map = getattr(self, "_measurement_items", {})
            if point_id in items_map:
                top_item, child_item, _w = items_map.get(point_id, (None, None, None))
                if top_item is not None:
                    if x_mm is not None and y_mm is not None:
                        top_item.setText(
                            0, f"Point #{point_id} {x_mm:.2f}:{y_mm:.2f} mm"
                        )
                    else:
                        top_item.setText(0, f"Point #{point_id}")
        except Exception:
            pass

        # --- Add the measurement to the widget ---
        widget.add_measurement(results, timestamp or getattr(self, "_timestamp", ""))
        logger.debug("Added measurement to widget", point_id=point_id, row=row)

    def _get_point_id_from_table_row(self, row: int) -> Optional[int]:
        """Extract point_id from table row."""
        point_id = None

        # Try to get point_id from table cell first
        item0 = self.pointsTable.item(row, 0)
        if item0 is not None:
            txt = item0.text().strip()
            if txt:
                try:
                    point_id = int(txt)
                except ValueError:
                    pass

        # Fallback: read from underlying graphics item data
        if point_id is None:
            gp = self.image_view.points_dict["generated"]["points"]
            up = self.image_view.points_dict["user"]["points"]

            if row < len(gp):
                pid = gp[row].data(1)
                point_id = int(pid) if pid is not None else None
            else:
                urow = row - len(gp)
                if 0 <= urow < len(up):
                    pid = up[urow].data(1)
                    point_id = int(pid) if pid is not None else None

        return point_id

    def _get_or_create_measurement_widget(
        self, point_id: int
    ) -> Optional[MeasurementHistoryWidget]:
        """Get existing widget or create a new one (managed in the right panel, not in the table)."""
        # Check if we already have a widget for this point_id
        widget = getattr(self, "measurement_widgets", {}).get(point_id)
        if widget is not None and not getattr(widget, "isHidden", None) is None:
            return widget

        # Prefer to let the ZonePoints UI create/manage the widget in the right panel if available
        add_to_panel = getattr(self, "add_measurement_widget_to_panel", None)
        if callable(add_to_panel):
            add_to_panel(point_id)
            widget = getattr(self, "measurement_widgets", {}).get(point_id)
            if widget is not None:
                return widget

        # Fallback: create a standalone widget and store it in the mapping
        widget = MeasurementHistoryWidget(
            masks=getattr(self, "masks", {}),
            ponis=getattr(self, "ponis", {}),
            parent=self,
            point_id=point_id,
        )
        if not hasattr(self, "measurement_widgets"):
            self.measurement_widgets = {}
        self.measurement_widgets[point_id] = widget
        return widget

    def pause_measurements(self):
        """
        Toggles between pause and resume of the measurement sequence.
        """
        if not hasattr(self, "paused"):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pause_btn.setText("Resume")
            logger.info("Measurements paused")
        else:
            self.paused = False
            self.pause_btn.setText("Pause")
            logger.info("Measurements resumed")
            self.measure_next_point()

    def stop_measurements(self):
        """
        Stops the measurement process, resets all progress/UI states.
        """
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurement stopped.")
        self.start_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        logger.info("Measurements stopped and reset")
