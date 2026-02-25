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

    @staticmethod
    def _as_text(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _json_default(value):
        """Convert numpy/path-like objects to JSON-serializable values."""
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return list(value)
        return str(value)

    @staticmethod
    def _new_measurement_point_uid(counter: int) -> str:
        """Return point UID as '<integer_counter>_<8 hex symbols>'."""
        try:
            counter_int = int(counter)
        except Exception:
            counter_int = 0
        return f"{counter_int}_{uuid.uuid4().hex[:8]}"

    def _point_item_uid(self, point_item, counter: int) -> str:
        """Read or assign stable point UID on graphics item (data key 2)."""
        try:
            existing = point_item.data(2)
            if isinstance(existing, bytes):
                existing = existing.decode("utf-8", errors="replace")
            if isinstance(existing, str) and existing.strip():
                return existing.strip()
        except Exception:
            pass

        uid = self._new_measurement_point_uid(counter)
        try:
            point_item.setData(2, uid)
        except Exception:
            pass
        return uid

    @staticmethod
    def _point_distance_sq(p1, p2) -> float:
        if p1 is None or p2 is None:
            return float("inf")
        try:
            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])
            return dx * dx + dy * dy
        except Exception:
            return float("inf")

    def _session_has_i0_measurement(self) -> bool:
        """Return True when active session already has an I0 attenuation measurement."""
        session_manager = getattr(self, "session_manager", None)
        if session_manager is None:
            return False

        try:
            if (
                not hasattr(session_manager, "is_session_active")
                or not session_manager.is_session_active()
            ):
                return False
        except Exception:
            return False

        try:
            return getattr(session_manager, "i0_counter", None) is not None
        except Exception:
            return False

    def _dump_state_measurements(self):
        """Write state_measurements JSON with numpy-safe serialization."""
        if not hasattr(self, "state_path_measurements") or not self.state_path_measurements:
            return
        with open(self.state_path_measurements, "w") as f:
            json.dump(
                self.state_measurements,
                f,
                indent=4,
                default=self._json_default,
            )

    def _set_measurement_controls_idle(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        if hasattr(self, "skip_btn") and self.skip_btn is not None:
            self.skip_btn.setEnabled(False)
        self.paused = False
        self.stopped = False

    def _choose_resume_or_remeasure(self, measured_count: int, pending_count: int) -> str:
        pm = _pm()
        reply = pm.QMessageBox.question(
            self,
            "Restored Session Detected",
            (
                "Loaded session already contains measurements.\n\n"
                f"Measured points: {measured_count}\n"
                f"Pending points: {pending_count}\n\n"
                "Yes: continue pending points only.\n"
                "No: re-measure all points (existing measurements will be kept)."
            ),
            pm.QMessageBox.Yes | pm.QMessageBox.No,
            pm.QMessageBox.Yes,
        )
        if reply == pm.QMessageBox.Yes:
            return "resume"
        return "remeasure"

    def _confirm_remeasure_completed_session(self, total_points: int) -> bool:
        pm = _pm()
        reply = pm.QMessageBox.question(
            self,
            "Session Already Complete",
            (
                "All points in the restored session are already measured.\n\n"
                f"Total measured points: {total_points}\n\n"
                "Re-measure all points?\n"
                "(Existing measurements will be kept.)"
            ),
            pm.QMessageBox.Yes | pm.QMessageBox.No,
            pm.QMessageBox.No,
        )
        return reply == pm.QMessageBox.Yes

    def _resolve_session_point_plan(self, measurement_points):
        pm = _pm()

        default_plan = {
            "mode": "new",
            "measurement_points": list(measurement_points),
            "session_point_indices": [idx for idx in range(1, len(measurement_points) + 1)],
            "measured_count": 0,
        }

        session_manager = getattr(self, "session_manager", None)
        if session_manager is None:
            return default_plan
        if not hasattr(session_manager, "is_session_active") or not session_manager.is_session_active():
            return default_plan

        session_path = getattr(session_manager, "session_path", None)
        schema = getattr(session_manager, "schema", None)
        if not session_path or schema is None:
            return default_plan

        try:
            import h5py

            session_points = []
            with h5py.File(session_path, "r") as h5f:
                points_group = h5f.get(schema.GROUP_POINTS)
                if points_group is None:
                    return default_plan

                for point_id in sorted(points_group.keys()):
                    point_name = str(point_id)
                    if not point_name.startswith("pt_"):
                        continue
                    try:
                        point_index = int(point_name.split("_")[-1])
                    except Exception:
                        continue
                    point_group = points_group[point_id]
                    status = self._as_text(
                        point_group.attrs.get(schema.ATTR_POINT_STATUS, "")
                    ).strip().lower()
                    point_uid = self._as_text(
                        point_group.attrs.get("point_uid", "")
                    ).strip()
                    physical = point_group.attrs.get(
                        getattr(schema, "ATTR_PHYSICAL_COORDINATES_MM", "physical_coordinates_mm"),
                        None,
                    )
                    physical_xy = None
                    try:
                        if physical is not None and len(physical) >= 2:
                            physical_xy = (float(physical[0]), float(physical[1]))
                    except Exception:
                        physical_xy = None
                    session_points.append(
                        {
                            "point_index": int(point_index),
                            "status": status,
                            "point_uid": point_uid,
                            "physical_xy": physical_xy,
                        }
                    )

            if not session_points:
                return default_plan

            has_session_uids = any(
                str(sp.get("point_uid") or "").strip() for sp in session_points
            )
            has_session_coords = any(sp.get("physical_xy") is not None for sp in session_points)
            if (
                len(session_points) != len(measurement_points)
                and not has_session_uids
                and not has_session_coords
            ):
                pm.logger.warning(
                    "Session/GUI point count mismatch without UID/coordinate metadata; rebuilding session points from current grid",
                    session_points=int(len(session_points)),
                    gui_points=int(len(measurement_points)),
                )
                return default_plan

            measured_status = self._as_text(schema.POINT_STATUS_MEASURED).strip().lower()
            pending_indices = [
                int(sp["point_index"])
                for sp in session_points
                if sp["status"] != measured_status
            ]
            measured_count = len(session_points) - len(pending_indices)

            if not pending_indices:
                return {
                    **default_plan,
                    "mode": "complete",
                    "measurement_points": [],
                    "session_point_indices": [],
                    "measured_count": measured_count,
                }

            measurement_uid_to_idx = {}
            for idx, mp in enumerate(measurement_points):
                uid = str(mp.get("unique_id") or "").strip()
                if uid and uid not in measurement_uid_to_idx:
                    measurement_uid_to_idx[uid] = idx

            session_to_measure_idx = {}
            used_indices = set()

            # 1) Exact UID matching when available (robust across restore/reorder).
            for sp in session_points:
                sp_idx = int(sp["point_index"])
                sp_uid = str(sp.get("point_uid") or "").strip()
                if not sp_uid:
                    continue
                mp_idx = measurement_uid_to_idx.get(sp_uid)
                if mp_idx is None or mp_idx in used_indices:
                    continue
                session_to_measure_idx[sp_idx] = mp_idx
                used_indices.add(mp_idx)

            # 2) Coordinate nearest-neighbour fallback.
            for sp in session_points:
                sp_idx = int(sp["point_index"])
                if sp_idx in session_to_measure_idx:
                    continue
                sp_xy = sp.get("physical_xy")
                if sp_xy is None:
                    continue

                best_idx = None
                best_d2 = float("inf")
                for idx, mp in enumerate(measurement_points):
                    if idx in used_indices:
                        continue
                    d2 = self._point_distance_sq(sp_xy, (mp.get("x"), mp.get("y")))
                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = idx

                if best_idx is not None:
                    session_to_measure_idx[sp_idx] = best_idx
                    used_indices.add(best_idx)

            # 3) Index fallback for any remaining points.
            for sp in session_points:
                sp_idx = int(sp["point_index"])
                if sp_idx in session_to_measure_idx:
                    continue
                pos = sp_idx - 1
                if 0 <= pos < len(measurement_points) and pos not in used_indices:
                    session_to_measure_idx[sp_idx] = pos
                    used_indices.add(pos)

            resumed_points = []
            resumed_session_indices = []
            for session_point_index in pending_indices:
                mp_idx = session_to_measure_idx.get(int(session_point_index))
                if mp_idx is None:
                    continue
                resumed_points.append(measurement_points[mp_idx])
                resumed_session_indices.append(int(session_point_index))

            if len(resumed_points) != len(pending_indices):
                pm.logger.warning(
                    "Resume mapping could not match all pending points; using mapped subset",
                    pending_points=int(len(pending_indices)),
                    mapped_points=int(len(resumed_points)),
                    session_points=int(len(session_points)),
                    gui_points=int(len(measurement_points)),
                )

            if not resumed_points:
                pm.logger.warning(
                    "No pending session points could be mapped; rebuilding from current grid",
                    pending_points=int(len(pending_indices)),
                )
                return default_plan

            return {
                **default_plan,
                "mode": "resume",
                "measurement_points": resumed_points,
                "session_point_indices": resumed_session_indices,
                "measured_count": measured_count,
            }
        except Exception as exc:
            pm.logger.warning(
                "Failed to inspect existing session points for resume; starting from full point set",
                error=str(exc),
                exc_info=True,
            )
            return default_plan

    def _existing_session_point_count(self) -> int:
        session_manager = getattr(self, "session_manager", None)
        if session_manager is None:
            return 0
        if not hasattr(session_manager, "is_session_active") or not session_manager.is_session_active():
            return 0

        session_path = getattr(session_manager, "session_path", None)
        schema = getattr(session_manager, "schema", None)
        if not session_path or schema is None:
            return 0

        try:
            import h5py

            with h5py.File(session_path, "r") as h5f:
                points_group = h5f.get(schema.GROUP_POINTS)
                if points_group is None:
                    return 0
                return int(
                    len([name for name in points_group.keys() if str(name).startswith("pt_")])
                )
        except Exception:
            return 0

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

        if hasattr(self, "refresh_sidecar_status"):
            if not self.refresh_sidecar_status(show_message=True):
                self._append_capture_log(
                    "Start cancelled: A2K sidecar heartbeat unavailable"
                )
                return

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
            self._dump_state_measurements()
        except Exception as e:
            pm.logger.error("Error saving state with encoded image", error=str(e))

        if self.pointsTable.rowCount() == 0:
            pm.logger.warning("No points available for measurement")
            self._append_capture_log("Start cancelled: no measurement points")
            return

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        if hasattr(self, "skip_btn") and self.skip_btn is not None:
            self.skip_btn.setEnabled(True)
        self.stopped = False
        self.paused = False
        self._session_point_indices = []

        generated_points = self.image_view.points_dict["generated"]["points"]
        user_points = self.image_view.points_dict["user"]["points"]
        all_points = []
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            uid = self._point_item_uid(item, i + 1)
            all_points.append((i, x_mm, y_mm, uid))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            uid = self._point_item_uid(item, offset + j + 1)
            all_points.append((offset + j, x_mm, y_mm, uid))
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

        for _orig_idx, (pt_idx, x_mm, y_mm, point_uid) in enumerate(all_points_sorted):
            if (x_min <= x_mm <= x_max) and (y_min <= y_mm <= y_max):
                unique_id = str(point_uid).strip() if point_uid else self._new_measurement_point_uid(valid_idx + 1)
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
        self.total_points = len(self.sorted_indices)
        self.progressBar.setMaximum(self.total_points)
        self.initial_estimate = self.total_points * self.integration_time
        self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")

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
            self._set_measurement_controls_idle()
            self._append_capture_log("Start failed: all points are outside stage limits")
            return

        full_measurement_points = list(measurement_points)
        session_plan = self._resolve_session_point_plan(measurement_points)
        should_seed_session_points = True
        if session_plan.get("mode") == "resume":
            measured_count = int(session_plan.get("measured_count", 0))
            pending_count = int(len(session_plan.get("measurement_points", []) or []))
            choice = self._choose_resume_or_remeasure(
                measured_count=measured_count,
                pending_count=pending_count,
            )
            if choice == "resume":
                measurement_points = list(session_plan.get("measurement_points", []) or [])
                should_seed_session_points = False
                self._session_point_indices = list(
                    session_plan.get("session_point_indices", []) or []
                )
                pm.logger.info(
                    "Resuming restored session with pending points only",
                    measured_points=measured_count,
                    pending_points=int(len(measurement_points)),
                )
                self._append_session_log(
                    f"Resume mode: {measured_count} point(s) already measured, "
                    f"{len(measurement_points)} pending"
                )
            else:
                measurement_points = full_measurement_points
                should_seed_session_points = False
                self._session_point_indices = [
                    idx for idx in range(1, len(measurement_points) + 1)
                ]
                pm.logger.info(
                    "User selected full re-measurement for restored session",
                    total_points=int(len(measurement_points)),
                )
                self._append_session_log(
                    "Re-measure mode: all points will be captured again"
                )

            self.sorted_indices = [mp["point_index"] for mp in measurement_points]
            self.total_points = len(self.sorted_indices)
            self.progressBar.setMaximum(self.total_points)
            self.initial_estimate = self.total_points * self.integration_time
            self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")
        elif session_plan.get("mode") == "complete":
            if not self._confirm_remeasure_completed_session(total_points=len(full_measurement_points)):
                pm.logger.info("Restore start skipped: all session points already measured")
                self._append_capture_log("Start skipped: restored session already complete")
                self._set_measurement_controls_idle()
                return

            measurement_points = full_measurement_points
            should_seed_session_points = False
            self._session_point_indices = [
                idx for idx in range(1, len(measurement_points) + 1)
            ]
            self.sorted_indices = [mp["point_index"] for mp in measurement_points]
            self.total_points = len(self.sorted_indices)
            self.progressBar.setMaximum(self.total_points)
            self.initial_estimate = self.total_points * self.integration_time
            self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")
            self._append_session_log(
                "Re-measure mode: completed session will be measured again"
            )
        else:
            self._session_point_indices = list(
                session_plan.get("session_point_indices", []) or []
            )

        # Default plan uses compact 1..N mapping; when an active session already
        # contains seeded points we must remap by original point indices.
        if session_plan.get("mode") == "new":
            existing_points_count = self._existing_session_point_count()
            if existing_points_count > 0:
                self._session_point_indices = []

        if not self._session_point_indices:
            mapped_from_original = []
            existing_points_count = self._existing_session_point_count()
            if existing_points_count > 0:
                for i, mp in enumerate(measurement_points):
                    try:
                        original_idx = int(mp.get("point_index", i))
                    except Exception:
                        continue
                    session_point_index = original_idx + 1
                    if 1 <= session_point_index <= existing_points_count:
                        mapped_from_original.append(session_point_index)
                if len(mapped_from_original) == len(measurement_points):
                    self._session_point_indices = mapped_from_original
                    pm.logger.info(
                        "Mapped measurement order to existing session points",
                        mapped_points=int(len(self._session_point_indices)),
                        session_points=int(existing_points_count),
                    )

            if not self._session_point_indices:
                self._session_point_indices = [
                    idx for idx in range(1, len(measurement_points) + 1)
                ]

        if not measurement_points:
            pm.logger.info("No pending measurement points after resume filtering")
            self._append_capture_log("Start skipped: no pending points after filtering")
            self._set_measurement_controls_idle()
            return

        self._reuse_existing_i0_from_session = False
        try:
            if hasattr(self, "attenuationCheckBox") and self.attenuationCheckBox.isChecked():
                if self._session_has_i0_measurement():
                    self._reuse_existing_i0_from_session = True
                    pm.logger.info(
                        "Skipping I0 background capture: session already contains I0 measurement",
                        i0_counter=getattr(getattr(self, "session_manager", None), "i0_counter", None),
                    )
                    self._append_capture_log("I0 already recorded in session; reusing existing I0")
                else:
                    self._capture_attenuation_background()
        except Exception as e:
            pm.logger.warning(
                "Failed to capture attenuation background; will continue without it",
                error=str(e),
            )

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
                pm.logger.info("=== SESSION CONTAINER POPULATION ===")
                existing_points_count = self._existing_session_point_count()
                if should_seed_session_points and existing_points_count > 0:
                    should_seed_session_points = False
                    if not self._session_point_indices:
                        mapped_indices = []
                        for idx, mp in enumerate(measurement_points):
                            try:
                                mapped_indices.append(int(mp.get("point_index", idx)) + 1)
                            except Exception:
                                mapped_indices.append(int(idx) + 1)
                        self._session_point_indices = mapped_indices
                    pm.logger.info(
                        "Session already contains points; skipping point regeneration",
                        existing_points=int(existing_points_count),
                        planned_points=int(len(measurement_points)),
                    )
                    self._append_session_log(
                        f"Session points already exist ({existing_points_count}); reusing them"
                    )
                if should_seed_session_points:
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
                                "point_uid": str(pt.get("unique_id") or ""),
                            }
                        )

                    pm.logger.info(f"Adding {len(points_for_session)} points to session container...")
                    self._append_session_log(
                        f"Initializing session container: {len(points_for_session)} points"
                    )
                    self.session_manager.add_points(points_for_session)
                    pm.logger.info(f"✓ Added {len(points_for_session)} points to session container")
                    self._append_session_log(
                        f"Session points written: {len(points_for_session)}"
                    )
                else:
                    pm.logger.info(
                        "Reusing existing points from session; skipping point regeneration"
                    )
                    self._append_session_log(
                        "Session points reused from existing container"
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
