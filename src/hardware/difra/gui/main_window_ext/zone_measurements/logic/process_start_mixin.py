import hashlib
import json
import time
import uuid
from copy import copy
from pathlib import Path


def _pm():
    from hardware.difra.gui.main_window_ext.zone_measurements.logic import process_mixin as pm

    return pm


class ZoneMeasurementsProcessStartMixin:
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

    def _ensure_writable_session_for_measurement(self) -> bool:
        pm = _pm()

        session_manager = getattr(self, "session_manager", None)
        if session_manager is None:
            return True
        if not hasattr(session_manager, "is_session_active"):
            return True
        if not session_manager.is_session_active():
            return True
        if not hasattr(session_manager, "is_locked"):
            return True
        if not session_manager.is_locked():
            return True

        info = {}
        try:
            info = session_manager.get_session_info() or {}
        except Exception:
            info = {}

        sample_id = info.get("sample_id") or "UNKNOWN"
        session_id = info.get("session_id") or "UNKNOWN"

        try:
            session_manager.close_session()
        except Exception as exc:
            pm.logger.warning(
                "Failed to close locked session before auto new-session flow",
                error=str(exc),
            )

        if hasattr(self, "update_session_status"):
            try:
                self.update_session_status()
            except Exception:
                pass

        pm.QMessageBox.information(
            self,
            "Session Locked",
            "The active session container is locked and cannot accept new measurements.\n\n"
            f"Closed locked session:\nSample ID: {sample_id}\nSession ID: {session_id}\n\n"
            "A new session is required. Session creation dialog will open now.",
        )

        image_path = ""
        try:
            image_path = getattr(getattr(self, "image_view", None), "current_image_path", "") or ""
        except Exception:
            image_path = ""

        if hasattr(self, "_handle_new_sample_image"):
            self._handle_new_sample_image(image_path)
        else:
            pm.QMessageBox.warning(
                self,
                "Session Required",
                "Please create a new session before starting measurements.",
            )
            return False

        if not session_manager.is_session_active() or session_manager.is_locked():
            pm.logger.warning(
                "Measurement start cancelled: writable session was not created after locked-session rollover"
            )
            return False

        return True

    def start_measurements(self):
        pm = _pm()

        self.manual_save_state()
        self.measurement_folder = Path(self.folderLineEdit.text().strip())
        self.state_path_measurements = self.measurement_folder / f"{self.fileNameLineEdit.text()}_state.json"
        pm.logger.info(
            "Measurement start requested",
            measurement_folder=str(self.measurement_folder),
            state_file=str(self.state_path_measurements),
        )

        if not self.measurement_folder.exists():
            pm.QMessageBox.warning(
                self,
                "Folder Error",
                "Selected folder does not exist. Please select the correct folder.",
            )
            self._append_capture_log("Start failed: save folder does not exist")
            return

        if not self._ensure_writable_session_for_measurement():
            self._append_session_log("Start cancelled: no writable session container")
            return

        if not self._confirm_poni_settings_before_measurement():
            self._append_capture_log("Start cancelled: PONI confirmation rejected")
            return

        try:
            from .preflight_dialog import PreflightDialog

            d = PreflightDialog(
                self,
                session_manager=getattr(self, "session_manager", None),
            )
            if d.exec_() != d.Accepted:
                self._append_capture_log("Start cancelled: preflight checklist not confirmed")
                return
        except Exception as e:
            pm.logger.warning("Preflight dialog failed; proceeding without it", error=str(e))

        group_hash = getattr(self, "calibration_group_hash", None)
        if not group_hash:
            try:
                group_hash = uuid.uuid4().hex[:16]
            except Exception:
                group_hash = None
            setattr(self, "calibration_group_hash", group_hash)
        if group_hash:
            try:
                if isinstance(getattr(self, "state", None), dict):
                    self.state["CALIBRATION_GROUP_HASH"] = group_hash
            except Exception:
                pass

        try:
            self.state_measurements = copy(self.state)
        except Exception as e:
            pm.logger.error("Error copying state for measurements", error=str(e))
            pm.QMessageBox.warning(self, "No state", "Save it.")
            return

        try:
            from hardware.difra.hardware.auxiliary import encode_image_to_base64

            self.state_measurements["image_base64"] = encode_image_to_base64(self.image_view.current_image_path)
            with open(self.state_path_measurements, "w") as f:
                json.dump(self.state_measurements, f, indent=4)
        except Exception as e:
            pm.logger.error("Error saving state with encoded image", error=str(e))

        if self.pointsTable.rowCount() == 0:
            pm.logger.warning("No points available for measurement")
            self._append_capture_log("Start cancelled: no measurement points")
            return

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stopped = False
        self.paused = False

        try:
            if hasattr(self, "attenuationCheckBox") and self.attenuationCheckBox.isChecked():
                self._capture_attenuation_background()
        except Exception as e:
            pm.logger.warning(
                "Failed to capture attenuation background; will continue without it",
                error=str(e),
            )

        generated_points = self.image_view.points_dict["generated"]["points"]
        user_points = self.image_view.points_dict["user"]["points"]
        all_points = []
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
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
        self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")
        pm.logger.info(
            "Starting measurements in sorted order",
            total_points=self.total_points,
            integration_time=self.integration_time,
        )
        self._append_capture_log(
            f"Start: {self.total_points} points, T={self.integration_time:.2f}s"
        )

        try:
            if hasattr(self, "_get_stage_limits"):
                limits = self._get_stage_limits()
            else:
                limits = (
                    self.stage_controller.get_limits()
                    if hasattr(self, "stage_controller")
                    else None
                )
        except Exception:
            limits = None
        if not limits:
            limits = {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}
        x_min, x_max = limits["x"]
        y_min, y_max = limits["y"]

        measurement_points = []
        skipped_points = []
        valid_idx = 0

        for _orig_idx, (pt_idx, x_mm, y_mm) in enumerate(all_points_sorted):
            if (x_min <= x_mm <= x_max) and (y_min <= y_mm <= y_max):
                id_str = f"{valid_idx}:{pt_idx}:{x_mm:.6f}:{y_mm:.6f}"
                unique_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()[:16]
                measurement_points.append(
                    {
                        "unique_id": unique_id,
                        "index": valid_idx,
                        "point_index": pt_idx,
                        "x": x_mm,
                        "y": y_mm,
                    }
                )
                valid_idx += 1
            else:
                skipped_points.append((pt_idx, x_mm, y_mm))
                pm.logger.warning(
                    f"Skipping measurement point {pt_idx} at ({x_mm:.3f}, {y_mm:.3f}) mm - "
                    f"outside limits X[{x_min:.1f},{x_max:.1f}] Y[{y_min:.1f},{y_max:.1f}] mm"
                )

        self.sorted_indices = [mp["point_index"] for mp in measurement_points]

        if skipped_points:
            pm.logger.info(
                f"Filtered measurement points: {len(measurement_points)} valid, "
                f"{len(skipped_points)} skipped due to axis limits"
            )
            self._append_capture_log(
                f"Filtered points: {len(measurement_points)} valid, {len(skipped_points)} skipped"
            )

        if not measurement_points:
            pm.logger.error("No valid measurement points within axis limits")
            pm.QMessageBox.warning(
                self,
                "No Valid Points",
                f"All measurement points exceed the axis limits of X[{x_min:.1f},{x_max:.1f}] and Y[{y_min:.1f},{y_max:.1f}] mm. "
                    "Please adjust your measurement grid.",
                )
            self._append_capture_log("Start failed: all points are outside stage limits")
            return

        self.state["measurement_points"] = measurement_points
        self.state["skipped_points"] = [
            {
                "point_index": pt_idx,
                "x": x_mm,
                "y": y_mm,
                "reason": "axis_limit_exceeded",
            }
            for pt_idx, x_mm, y_mm in skipped_points
        ]

        self.state_measurements["measurement_points"] = measurement_points
        self.state_measurements["skipped_points"] = self.state["skipped_points"]
        gh = getattr(self, "calibration_group_hash", None)
        if gh:
            self.state_measurements["CALIBRATION_GROUP_HASH"] = gh
        self.manual_save_state()

        if hasattr(self, "session_manager") and self.session_manager.is_session_active():
            try:
                points_for_session = []
                for pt in measurement_points:
                    pt_idx = pt["point_index"]
                    gp = self.image_view.points_dict["generated"]["points"]
                    up = self.image_view.points_dict["user"]["points"]

                    if pt_idx < len(gp):
                        point_item = gp[pt_idx]
                    else:
                        user_idx = pt_idx - len(gp)
                        point_item = up[user_idx]

                    center = point_item.sceneBoundingRect().center()
                    pixel_x = center.x()
                    pixel_y = center.y()
                    points_for_session.append(
                        {
                            "pixel_coordinates": [float(pixel_x), float(pixel_y)],
                            "physical_coordinates_mm": [pt["x"], pt["y"]],
                        }
                    )

                pm.logger.info("=== SESSION CONTAINER POPULATION ===")
                pm.logger.info(f"Adding {len(points_for_session)} points to session container...")
                self._append_session_log(
                    f"Initializing session container: {len(points_for_session)} points"
                )
                self.session_manager.add_points(points_for_session)
                pm.logger.info(f"✓ Added {len(points_for_session)} points to session container")
                self._append_session_log(
                    f"Session points written: {len(points_for_session)}"
                )

                if hasattr(self, "_add_zones_to_session"):
                    pm.logger.info("Adding zones to session container...")
                    num_shapes = len(self.state.get("shapes", []))
                    pm.logger.info(f"Found {num_shapes} shapes in state")
                    self._add_zones_to_session()
                    pm.logger.info("✓ Zones processing complete")
                    self._append_session_log(f"Session zones synced: {num_shapes}")
                else:
                    pm.logger.warning("⚠ _add_zones_to_session method not found")

                if hasattr(self, "_add_mapping_to_session"):
                    pm.logger.info("Adding mapping to session container...")
                    self._add_mapping_to_session()
                    pm.logger.info("✓ Mapping added")
                    self._append_session_log("Session image mapping updated")
                else:
                    pm.logger.warning("⚠ _add_mapping_to_session method not found")

                pm.logger.info("=== SESSION CONTAINER INITIALIZED ===")
                self._append_session_log("Session container initialization complete")
            except Exception as e:
                pm.logger.error(f"Failed to add points to session container: {e}", exc_info=True)
                self._append_session_log(
                    f"Session initialization failed: {type(e).__name__}"
                )

        self.measure_next_point()
