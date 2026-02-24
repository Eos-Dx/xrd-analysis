import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _pm():
    from hardware.difra.gui.main_window_ext.zone_measurements.logic import process_mixin as pm

    return pm


class ZoneMeasurementsProcessResultsMixin:
    def _append_capture_log(self, message: str):
        try:
            self._append_measurement_log(f"[CAPTURE] {message}")
        except Exception:
            pass

    def _append_session_log(self, message: str):
        try:
            self._append_measurement_log(f"[SESSION] {message}")
        except Exception:
            pass

    def on_capture_finished(self, success: bool, result_files: dict):
        pm = _pm()
        current_index = self.current_measurement_sorted_index
        point_index_1based = current_index + 1
        session_manager = getattr(self, "session_manager", None)

        if not success:
            pm.logger.error("Measurement capture failed")
            marked_failed = False
            if (
                session_manager is not None
                and hasattr(session_manager, "is_session_active")
                and session_manager.is_session_active()
                and hasattr(session_manager, "fail_point_measurement")
            ):
                try:
                    session_manager.fail_point_measurement(
                        point_index=point_index_1based,
                        reason="capture_failed",
                        timestamp_end=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    marked_failed = True
                except Exception:
                    pm.logger.warning("Failed to mark failed measurement in session container", exc_info=True)
            self._append_capture_log(f"Point {point_index_1based}: capture failed")
            if marked_failed:
                self._append_session_log(
                    f"Point {point_index_1based}: marked failed in session container"
                )
            else:
                self._append_session_log(
                    f"Point {point_index_1based}: capture failed before session write"
                )
            return

        pm.logger.info("Measurement capture successful", files=list(result_files.keys()))
        self._append_capture_log(f"Point {point_index_1based}: capture complete")

        detector_lookup = {d["alias"]: d for d in self.config["detectors"]}
        measurements = self.state_measurements.get("measurements_meta", {})
        measurement_points = self.state_measurements["measurement_points"]
        x = self._x_mm
        y = self._y_mm
        point_unique_id = measurement_points[current_index]["unique_id"]

        for alias, npy_filename in result_files.items():
            if not npy_filename:
                pm.logger.warning("Capture returned empty file path", detector_alias=alias)
                continue
            detector_meta = detector_lookup.get(alias, {})
            entry = {
                "x": x,
                "y": y,
                "unique_id": point_unique_id,
                "base_file": self._base_name,
                "integration_time": self.integration_time,
                "detector_alias": alias,
                "detector_id": detector_meta.get("id"),
                "detector_type": detector_meta.get("type"),
                "detector_size": detector_meta.get("size"),
                "pixel_size_um": detector_meta.get("pixel_size_um"),
                "faulty_pixels": detector_meta.get("faulty_pixels"),
            }
            gh = getattr(self, "calibration_group_hash", None)
            if gh:
                entry["CALIBRATION_GROUP_HASH"] = gh
            measurements[Path(npy_filename).name] = entry

        self.state_measurements["measurements_meta"] = measurements

        with open(self.state_path_measurements, "w") as f:
            json.dump(self.state_measurements, f, indent=4)

        pm.logger.info(
            "Measurement state file updated",
            state_file=str(self.state_path_measurements),
            entries=len(measurements),
        )
        self._append_capture_log("Measurement metadata saved to state file")

        if session_manager is not None and hasattr(session_manager, "is_session_active") and session_manager.is_session_active():
            self._append_session_log(f"Point {point_index_1based}: writing to session container")
            try:
                pm.logger.info(f"=== ADDING MEASUREMENT TO H5 (Point {point_index_1based}) ===")
                pm.logger.info(f"Session path: {session_manager.session_path}")

                all_data = {}
                raw_files_data = {}

                detector_lookup = {d["alias"]: d for d in self.config["detectors"]}
                poni_alias_map = {}
                for alias, npy_file in result_files.items():
                    if not npy_file:
                        continue
                    npy_path = Path(npy_file)
                    if not npy_path.exists():
                        pm.logger.warning("Capture file missing on disk", detector_alias=alias, file=str(npy_file))
                        continue
                    detector_meta = detector_lookup.get(alias, {})
                    detector_id = detector_meta.get("id", alias)
                    poni_alias_map[alias] = detector_id
                    pm.logger.info(f"Loading {alias} data from: {npy_path.name}")
                    all_data[detector_id] = np.load(npy_file)
                    pm.logger.info(f"  Data shape: {all_data[detector_id].shape}")

                    base_name = npy_path.stem
                    folder = npy_path.parent

                    detector_controller = self.detector_controller.get(alias)
                    if detector_controller and hasattr(detector_controller, "get_raw_file_patterns"):
                        patterns = detector_controller.get_raw_file_patterns()
                    else:
                        patterns = ["*.txt", "*.dsc", "*.t3pa"]
                        pm.logger.warning(
                            f"Detector {alias} has no get_raw_file_patterns(), using default patterns"
                        )

                    raw_files = {}
                    for pattern in patterns:
                        ext = pattern[1:] if pattern.startswith("*") else pattern
                        raw_file = folder / f"{base_name}{ext}"
                        if raw_file.exists():
                            try:
                                with open(raw_file, "rb") as f:
                                    file_format = ext[1:] if ext.startswith(".") else ext
                                    blob_key = f"raw_{file_format}"
                                    raw_files[blob_key] = f.read()
                                pm.logger.debug(f"Read raw file for blob: {raw_file.name} -> {blob_key}")
                            except Exception as e:
                                pm.logger.warning(f"Failed to read raw file {raw_file}: {e}")

                    if raw_files:
                        raw_files_data[detector_id] = raw_files
                        pm.logger.info(
                            f"  Found {len(raw_files)} raw files for {alias}: {list(raw_files.keys())}"
                        )
                    else:
                        pm.logger.warning(
                            f"  No raw files found for {alias} using patterns {patterns}"
                        )

                pm.logger.info(f"Loaded data from {len(all_data)} detectors")

                detector_metadata = {}
                for detector_id in all_data.keys():
                    detector_metadata[detector_id] = {
                        "integration_time_ms": self.integration_time * 1000,
                        "detector_id": detector_id,
                        "x_mm": x,
                        "y_mm": y,
                        "timestamp": self._timestamp,
                        "unique_id": point_unique_id,
                    }

                if not all_data:
                    pm.logger.error(
                        "No detector payload produced for successful capture; marking failed",
                        point_index=point_index_1based,
                    )
                    if hasattr(session_manager, "fail_point_measurement"):
                        session_manager.fail_point_measurement(
                            point_index=point_index_1based,
                            reason="capture_success_without_payload",
                            timestamp_end=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    raise RuntimeError("No detector payload produced")

                raw_files_by_detector_id = raw_files_data
                pm.logger.info(f"Writing to H5: /measurements/pt_{point_index_1based:03d}/meas_NNNNNNNNN")
                pm.logger.info(f"  Detectors: {list(all_data.keys())}")
                pm.logger.info(f"  Raw files: {len(raw_files_by_detector_id)} detector(s) with blobs")

                if hasattr(session_manager, "complete_point_measurement"):
                    session_manager.complete_point_measurement(
                        point_index=point_index_1based,
                        measurement_data=all_data,
                        detector_metadata=detector_metadata,
                        poni_alias_map=poni_alias_map,
                        raw_files=raw_files_by_detector_id if raw_files_by_detector_id else None,
                        timestamp_end=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                else:
                    session_manager.add_measurement(
                        point_index=point_index_1based,
                        measurement_data=all_data,
                        detector_metadata=detector_metadata,
                        poni_alias_map=poni_alias_map,
                        raw_files=raw_files_by_detector_id if raw_files_by_detector_id else None,
                    )
                pm.logger.info(f"✓ Measurement added to H5 container for point {point_index_1based}")
                self._append_session_log(
                    f"Point {point_index_1based}: saved to session ({len(all_data)} detector(s))"
                )
            except Exception as e:
                pm.logger.error("=" * 60)
                pm.logger.error("✗ CRITICAL ERROR: Failed to add measurement to H5")
                pm.logger.error("=" * 60)
                pm.logger.error(f"Error type: {type(e).__name__}")
                pm.logger.error(f"Error message: {e}")
                pm.logger.error(f"Point index: {point_index_1based}")
                pm.logger.error(f"Detectors: {list(result_files.keys())}")
                pm.logger.error(
                    f"Session path: {session_manager.session_path if session_manager is not None else 'N/A'}"
                )
                pm.logger.error("=" * 60, exc_info=True)
                pm.logger.warning("Continuing measurement workflow despite H5 write failure...")
                self._append_session_log(
                    f"Point {point_index_1based}: session write failed ({type(e).__name__})"
                )
                if hasattr(session_manager, "fail_point_measurement"):
                    try:
                        session_manager.fail_point_measurement(
                            point_index=point_index_1based,
                            reason=f"h5_write_failed:{type(e).__name__}",
                            timestamp_end=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        self._append_session_log(
                            f"Point {point_index_1based}: marked failed after session write error"
                        )
                    except Exception:
                        pm.logger.warning("Failed to persist failed status for point measurement", exc_info=True)
        else:
            pm.logger.warning("⚠ Session manager not active - measurements will NOT be saved to H5!")
            self._append_session_log("No active session container; point saved to files only")

        pm.logger.info("Spawning measurement thread for post-processing...")
        current_row = self.sorted_indices[self.current_measurement_sorted_index]
        self.spawn_measurement_thread(current_row, result_files)
        self._append_capture_log("Post-processing started")

        pm.logger.info("Updating UI visual feedback...")
        green_brush = pm.QColor(0, 255, 0)
        self._point_item.setBrush(green_brush)
        try:
            if self._zone_item:
                green_zone = pm.QColor(0, 255, 0)
                green_zone.setAlphaF(0.2)
                self._zone_item.setBrush(green_zone)
        except Exception as e:
            pm.logger.warning("Error updating zone item color", error=str(e))

        pm.logger.info("Scheduling measurement_finished in 1000ms...")
        pm.QTimer.singleShot(1000, self.measurement_finished)
        self._append_capture_log("Next point scheduled")
        pm.logger.info("<<< on_capture_finished complete")

    def spawn_measurement_thread(self, row, file_map):
        pm = _pm()
        if not self._zone_technical_imports_available():
            pm.logger.error("Cannot spawn measurement thread - technical imports not available")
            return

        thread = pm.QThread(self)
        MeasurementWorker = self._get_zone_technical_module("MeasurementWorker")
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
        pm = _pm()
        pm.logger.info(
            f">>> measurement_finished called (point {self.current_measurement_sorted_index + 1}/{self.total_points})"
        )

        if self.stopped:
            pm.logger.debug("Measurement stopped in measurement_finished")
            return

        pm.logger.info("Advancing to next point...")
        self.current_measurement_sorted_index += 1
        self.progressBar.setValue(self.current_measurement_sorted_index)
        pm.logger.info(f"Progress: {self.current_measurement_sorted_index}/{self.total_points}")
        elapsed = time.time() - self.measurementStartTime
        if self.current_measurement_sorted_index > 0:
            avg_time = elapsed / self.current_measurement_sorted_index
            remaining = avg_time * (self.total_points - self.current_measurement_sorted_index)
            percent_complete = (self.current_measurement_sorted_index / self.total_points) * 100
            self.timeRemainingLabel.setText(f"{percent_complete:.0f}% done, {remaining:.0f} sec remaining")

        if self.current_measurement_sorted_index < self.total_points and not self.paused and not self.stopped:
            pm.logger.info(
                f"Moving to next point ({self.current_measurement_sorted_index + 1}/{self.total_points})"
            )
            self.measure_next_point()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                pm.logger.info("=== ALL MEASUREMENT POINTS COMPLETED ===")
                self._append_capture_log("Measurement sequence complete")
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                self.start_btn.setEnabled(True)
            else:
                pm.logger.warning(f"Measurement stopped: paused={self.paused}, stopped={self.stopped}")

        pm.logger.info("<<< measurement_finished complete")

    def add_measurement_to_table(self, row, results, timestamp=None):
        pm = _pm()
        point_id = self._get_point_id_from_table_row(row)
        if point_id is None:
            pm.logger.warning("Could not determine point_id for measurement", row=row)
            return

        x_mm = None
        y_mm = None
        try:
            x_item = self.pointsTable.item(row, 3)
            y_item = self.pointsTable.item(row, 4)
            if x_item is not None and y_item is not None:
                x_mm = float(x_item.text()) if x_item.text() not in (None, "", "N/A") else None
                y_mm = float(y_item.text()) if y_item.text() not in (None, "", "N/A") else None
        except Exception:
            pass

        add_to_panel = getattr(self, "add_measurement_widget_to_panel", None)
        if callable(add_to_panel):
            add_to_panel(point_id)

        widget = self._get_or_create_measurement_widget(point_id)
        if widget is None:
            pm.logger.error("Could not get/create measurement widget", point_id=point_id)
            return

        try:
            if x_mm is not None and y_mm is not None:
                if hasattr(widget, "set_mm_coordinates"):
                    widget.set_mm_coordinates(x_mm, y_mm)
                else:
                    widget.setWindowTitle(f"Measurement History: Point #{point_id} {x_mm:.2f}:{y_mm:.2f} mm")
            else:
                widget.setWindowTitle(f"Measurement History: Point #{point_id}")
        except Exception:
            pass

        try:
            items_map = getattr(self, "_measurement_items", {})
            if point_id in items_map:
                top_item, _child_item, _w = items_map.get(point_id, (None, None, None))
                if top_item is not None:
                    if x_mm is not None and y_mm is not None:
                        top_item.setText(0, f"Point #{point_id} {x_mm:.2f}:{y_mm:.2f} mm")
                    else:
                        top_item.setText(0, f"Point #{point_id}")
        except Exception:
            pass

        widget.add_measurement(results, timestamp or getattr(self, "_timestamp", ""))
        pm.logger.debug("Added measurement to widget", point_id=point_id, row=row)

    def _get_point_id_from_table_row(self, row: int) -> Optional[int]:
        point_id = None
        item0 = self.pointsTable.item(row, 0)
        if item0 is not None:
            txt = item0.text().strip()
            if txt:
                try:
                    point_id = int(txt)
                except ValueError:
                    pass

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

    def _get_or_create_measurement_widget(self, point_id: int):
        pm = _pm()
        widget = getattr(self, "measurement_widgets", {}).get(point_id)
        if widget is not None and not getattr(widget, "isHidden", None) is None:
            return widget

        add_to_panel = getattr(self, "add_measurement_widget_to_panel", None)
        if callable(add_to_panel):
            add_to_panel(point_id)
            widget = getattr(self, "measurement_widgets", {}).get(point_id)
            if widget is not None:
                return widget

        if not self._zone_technical_imports_available():
            pm.logger.error("Cannot create measurement widget - technical imports not available")
            return None

        MeasurementHistoryWidget = self._get_zone_technical_module("MeasurementHistoryWidget")
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
        pm = _pm()
        if not hasattr(self, "paused"):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pause_btn.setText("Resume")
            pm.logger.info("Measurements paused")
        else:
            self.paused = False
            self.pause_btn.setText("Pause")
            pm.logger.info("Measurements resumed")
            self.measure_next_point()

    def stop_measurements(self):
        pm = _pm()
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurement stopped.")
        self.start_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        pm.logger.info("Measurements stopped and reset")

    def _confirm_poni_settings_before_measurement(self):
        pm = _pm()
        try:
            active_aliases = self.hardware_controller.active_detector_aliases
        except Exception:
            dev_mode = self.config.get("DEV", False)
            ids = self.config.get("dev_active_detectors", []) if dev_mode else self.config.get("active_detectors", [])
            active_aliases = [d.get("alias") for d in self.config.get("detectors", []) if d.get("id") in ids]

        ponis = getattr(self, "ponis", {}) or {}
        poni_files = getattr(self, "poni_files", {}) or {}
        missing = [a for a in active_aliases if not ponis.get(a)]
        if missing:
            pm.QMessageBox.warning(
                self,
                "Missing PONI Calibration",
                "PONI calibration must be set for detectors: "
                + ", ".join(missing)
                + "\nLoad/select a valid technical container before starting measurements.",
            )
            return False

        # No confirmation popup: start measurements immediately when required PONI exists.
        return True
