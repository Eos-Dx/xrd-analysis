import os
import shutil
import time
from pathlib import Path

from hardware.difra.gui.container_api import get_container_version


def _pm():
    from hardware.difra.gui.main_window_ext.zone_measurements.logic import process_mixin as pm

    return pm


class ZoneMeasurementsProcessCaptureMixin:
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

    def _current_session_point_index(self) -> int:
        try:
            current_index = int(getattr(self, "current_measurement_sorted_index", 0))
        except Exception:
            current_index = 0

        mapped = getattr(self, "_session_point_indices", None)
        if isinstance(mapped, (list, tuple)) and 0 <= current_index < len(mapped):
            try:
                return int(mapped[current_index])
            except Exception:
                pass
        return current_index + 1

    def _move_stage(self, x_mm: float, y_mm: float, timeout_s: float):
        pm = _pm()
        if getattr(self, "hardware_client", None) is not None:
            pm.logger.info(
                "Stage move requested via hardware client",
                target_x_mm=float(x_mm),
                target_y_mm=float(y_mm),
                timeout_s=float(timeout_s),
            )
            self.hardware_client.move_to(x_mm, axis="x", timeout_s=timeout_s)
            result = self.hardware_client.move_to(y_mm, axis="y", timeout_s=timeout_s)
            pm.logger.info(
                "Stage move completed via hardware client",
                final_x_mm=float(result[0]),
                final_y_mm=float(result[1]),
            )
            return result
        if hasattr(self, "stage_controller") and self.stage_controller is not None:
            pm.logger.info(
                "Stage move requested via stage controller",
                target_x_mm=float(x_mm),
                target_y_mm=float(y_mm),
                timeout_s=float(timeout_s),
            )
            result = self.stage_controller.move_stage(x_mm, y_mm, move_timeout=timeout_s)
            pm.logger.info(
                "Stage move completed via stage controller",
                final_x_mm=float(result[0]),
                final_y_mm=float(result[1]),
            )
            return result
        pm.logger.error("Stage move requested without initialized stage")
        raise RuntimeError("Stage not initialized")

    def measure_next_point(self):
        pm = _pm()
        if self.stopped:
            pm.logger.debug("Measurement stopped")
            return
        if self.paused:
            pm.logger.debug("Measurement is paused. Waiting for resume")
            return
        sorted_count = len(getattr(self, "sorted_indices", []) or [])
        if sorted_count != int(getattr(self, "total_points", 0)):
            pm.logger.warning(
                "Measurement point count mismatch; syncing to sorted indices",
                total_points=int(getattr(self, "total_points", 0)),
                sorted_indices_count=int(sorted_count),
            )
            self.total_points = sorted_count
            try:
                self.progressBar.setMaximum(self.total_points)
            except Exception:
                pass

        if self.current_measurement_sorted_index >= sorted_count:
            pm.logger.info("All points measured")
            try:
                self.progressBar.setValue(sorted_count)
            except Exception:
                pass
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
        self._x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
        self._y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio

        point_index_1based = self.current_measurement_sorted_index + 1
        pm.logger.info(
            "Preparing measurement point",
            point_index=point_index_1based,
            total_points=self.total_points,
            target_x_mm=float(self._x_mm),
            target_y_mm=float(self._y_mm),
            integration_time_s=float(getattr(self, "integration_time", 0.0)),
        )
        self._append_capture_log(
            f"Point {point_index_1based}/{self.total_points}: move to ({self._x_mm:.3f}, {self._y_mm:.3f}) mm"
        )

        self._timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._base_name = self.fileNameLineEdit.text().strip()
        txt_filename_base = os.path.join(
            self.measurement_folder,
            f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}",
        )

        attenuation_enabled = getattr(self, "attenuationCheckBox", None)
        if attenuation_enabled and self.attenuationCheckBox.isChecked():
            self._start_attenuation_then_normal(txt_filename_base)
            return

        try:
            self._move_stage(self._x_mm, self._y_mm, timeout_s=15)
            self._append_hw_log(f"Stage positioned: ({self._x_mm:.3f}, {self._y_mm:.3f}) mm")
        except TimeoutError:
            pm.logger.warning(
                "Stage movement timed out before capture",
                point_index=point_index_1based,
                target_x_mm=float(self._x_mm),
                target_y_mm=float(self._y_mm),
            )
            self._append_hw_log("Stage move timeout before capture")
            pm.QMessageBox.warning(
                self,
                "Stage Timeout",
                "Stage movement timed out. Please check the hardware and try again. That's SAD",
            )
            return
        except Exception as e:
            pm.logger.error(
                "Stage movement failed before capture",
                point_index=point_index_1based,
                error=str(e),
            )
            self._append_hw_log(f"Stage move error before capture: {e}")
            pm.QMessageBox.warning(
                self,
                "Stage Error",
                f"Stage movement failed: {str(e)}",
            )
            return

        self._start_normal_capture(txt_filename_base)

    def _start_normal_capture(self, txt_filename_base: str):
        pm = _pm()
        if not self._zone_technical_imports_available():
            pm.logger.error("Cannot start normal capture - technical imports not available")
            try:
                self._append_capture_log("Error: technical imports unavailable")
            except Exception:
                pass
            return

        point_index_1based = self.current_measurement_sorted_index + 1
        session_point_index = self._current_session_point_index()
        self._append_capture_log(
            f"Normal capture start: point {point_index_1based}/{self.total_points}, t={self.integration_time:.3f}s"
        )

        session_manager = getattr(self, "session_manager", None)
        if (
            session_manager is not None
            and hasattr(session_manager, "is_session_active")
            and session_manager.is_session_active()
            and hasattr(session_manager, "begin_point_measurement")
        ):
            try:
                session_manager.begin_point_measurement(
                    point_index=session_point_index,
                    timestamp_start=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                self._append_session_log(f"Point {session_point_index}: opened in session container")
                if hasattr(session_manager, "log_event"):
                    session_manager.log_event(
                        message="Normal detector capture started",
                        event_type="capture_started",
                        details={
                            "point_index": session_point_index,
                            "x_mm": float(getattr(self, "_x_mm", 0.0)),
                            "y_mm": float(getattr(self, "_y_mm", 0.0)),
                            "integration_time_s": float(getattr(self, "integration_time", 0.0)),
                        },
                    )
            except Exception as exc:
                pm.logger.warning(
                    "Failed to mark measurement start in session container",
                    error=str(exc),
                )
                self._append_session_log(
                    f"Point {session_point_index}: failed to mark session start ({type(exc).__name__})"
                )

        container_version = get_container_version(
            self.config if hasattr(self, "config") else None
        )
        pm.logger.info(
            "Creating normal capture worker",
            point_index=point_index_1based,
            integration_time_s=float(self.integration_time),
            base_file=str(txt_filename_base),
            container_version=str(container_version),
        )
        CaptureWorker = self._get_zone_technical_module("CaptureWorker")
        self.capture_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integration_time,
            txt_filename_base=txt_filename_base,
            frames=1,
            naming_mode="normal",
            container_version=container_version,
            hardware_client=getattr(self, "hardware_client", None),
        )
        self.capture_thread = pm.QThread()
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        self.capture_thread.start()
        self._append_capture_log("Normal capture worker started")

    def _get_loading_position(self):
        try:
            att = self.config.get("attenuation", {})
            pos = att.get("loading_position")
            if pos and isinstance(pos, dict):
                return float(pos.get("x")), float(pos.get("y"))
        except Exception:
            pass
        try:
            if hasattr(self, "_get_home_load_positions"):
                positions = self._get_home_load_positions()
            else:
                positions = self.stage_controller.get_home_load_positions()
            return positions.get("load", (None, None))
        except Exception:
            return (None, None)

    def _capture_attenuation_background(self):
        pm = _pm()
        frames = int(getattr(self, "attenFramesSpin", None).value()) if hasattr(self, "attenFramesSpin") else 100
        short_t = float(getattr(self, "attenTimeSpin", None).value()) if hasattr(self, "attenTimeSpin") else 0.00005
        pm.logger.info(
            "Attenuation background requested",
            frames=int(frames),
            integration_time_s=float(short_t),
        )

        load_x, load_y = self._get_loading_position()
        if load_x is None or load_y is None:
            pm.logger.warning("Loading position not configured; skipping attenuation background capture")
            self._append_capture_log("I0 skipped: loading position is not configured")
            self._attenuation_bg_files = None
            return

        try:
            self._move_stage(load_x, load_y, timeout_s=20)
            self._append_hw_log(f"Stage moved to load position: ({load_x:.3f}, {load_y:.3f}) mm")
        except Exception as e:
            pm.logger.warning(
                "Failed to move to loading position; skipping attenuation background capture",
                error=str(e),
            )
            self._append_capture_log(f"I0 skipped: cannot move to loading position ({e})")
            self._attenuation_bg_files = None
            return

        self._append_capture_log("Attenuation: moved to loading position")
        self._append_capture_log(
            f"Attenuation I0 capture: frames={frames}, t={short_t:.6f}s"
        )

        group_ts = time.strftime("%Y%m%d_%H%M%S")
        base_name = self.fileNameLineEdit.text().strip()
        group_base = os.path.join(self.measurement_folder, f"{base_name}_{group_ts}")
        container_version = get_container_version(
            self.config if hasattr(self, "config") else None
        )

        if getattr(self, "hardware_client", None) is None:
            pm.logger.warning(
                "Hardware client unavailable; skipping attenuation background capture"
            )
            self._append_capture_log("I0 skipped: hardware client unavailable")
            self._attenuation_bg_files = None
            return

        try:
            raw_outputs = self.hardware_client.capture_exposure(
                exposure_s=short_t,
                frames=max(int(frames), 1),
                timeout_s=max(30.0, float(short_t) * max(int(frames), 1) + 30.0),
            )
        except Exception as e:
            pm.logger.warning(
                "Hardware client attenuation capture failed",
                error=str(e),
            )
            self._append_capture_log(f"I0 capture failed: {e}")
            self._attenuation_bg_files = None
            return

        results = {}
        for alias, controller in self.detector_controller.items():
            try:
                per_alias_base = f"{group_base}__{alias}_ATTENUATION0"
                src_raw = raw_outputs.get(alias)
                if src_raw is None and len(raw_outputs) == 1:
                    src_raw = next(iter(raw_outputs.values()))
                if not src_raw:
                    results[alias] = None
                    continue

                target_txt = Path(per_alias_base + ".txt")
                target_txt.parent.mkdir(parents=True, exist_ok=True)
                src_path = Path(src_raw)
                if src_path.resolve() != target_txt.resolve():
                    shutil.copy2(src_path, target_txt)
                    src_dsc = src_path.with_suffix(".dsc")
                    if src_dsc.exists():
                        shutil.copy2(src_dsc, target_txt.with_suffix(".dsc"))
                npy_path = controller.convert_to_container_format(
                    str(target_txt), container_version
                )
                results[alias] = npy_path
            except Exception as e:
                pm.logger.warning("Error capturing attenuation background", detector=alias, error=str(e))
                results[alias] = None

        self._attenuation_bg_files = results
        n_ok = sum(1 for v in results.values() if v)
        self._append_capture_log(f"I0 saved for {n_ok} detector(s)")

        if hasattr(self, "session_manager") and self.session_manager.is_session_active():
            try:
                import numpy as np

                all_data = {}
                for alias, npy_file in results.items():
                    if npy_file:
                        all_data[alias] = np.load(npy_file)

                if all_data:
                    detector_lookup = {d.get("alias"): d for d in self.config.get("detectors", [])}
                    measurement_data = {}
                    detector_metadata = {}
                    poni_alias_map = {}
                    for alias, signal in all_data.items():
                        detector_meta = detector_lookup.get(alias, {})
                        detector_id = detector_meta.get("id", alias)
                        measurement_data[detector_id] = signal
                        detector_metadata[detector_id] = {
                            "integration_time_ms": short_t * 1000,
                            "detector_id": detector_id,
                            "timestamp": group_ts,
                            "loading_position_mm": [load_x, load_y],
                            "n_frames": frames,
                        }
                        poni_alias_map[alias] = detector_id

                    self.session_manager.add_attenuation_measurement(
                        measurement_data=measurement_data,
                        detector_metadata=detector_metadata,
                        poni_alias_map=poni_alias_map,
                        mode="without",
                    )
                    pm.logger.info("Added I₀ (without sample) to session container", detectors=list(all_data.keys()))
                    self._append_session_log(
                        f"I0 saved to session container ({len(all_data)} detector(s))"
                    )
            except Exception as e:
                pm.logger.error(f"Failed to add I₀ to session container: {e}", exc_info=True)
                self._append_session_log(f"I0 session save failed: {type(e).__name__}")

    def _record_attenuation_files(self, key: str, files: dict):
        try:
            mp = self.state_measurements.get("measurement_points", [])
            idx = self.current_measurement_sorted_index
            uid = mp[idx].get("unique_id") if 0 <= idx < len(mp) else None
        except Exception:
            uid = None
        if uid is None:
            return
        try:
            att = self.state_measurements.setdefault("attenuation_files", {})
            entry = att.setdefault(uid, {})
            entry[key] = files or {}
            if hasattr(self, "state_path_measurements") and self.state_path_measurements:
                if hasattr(self, "_dump_state_measurements"):
                    self._dump_state_measurements()
                else:
                    import json

                    with open(self.state_path_measurements, "w") as f:
                        json.dump(self.state_measurements, f, indent=4)
        except Exception as e:
            print(f"Warning: failed to record attenuation files: {e}")

    def _start_attenuation_then_normal(self, txt_filename_base: str):
        pm = _pm()
        frames = int(getattr(self, "attenFramesSpin", None).value()) if hasattr(self, "attenFramesSpin") else 100
        short_t = float(getattr(self, "attenTimeSpin", None).value()) if hasattr(self, "attenTimeSpin") else 0.00005

        if getattr(self, "_attenuation_bg_files", None):
            try:
                self._record_attenuation_files("without_sample", self._attenuation_bg_files)
            except Exception:
                pass
        else:
            pm.QMessageBox.warning(
                self,
                "Attenuation Background Missing",
                "Background attenuation (without sample) was not captured; proceeding with with-sample and normal measurements.",
            )

        try:
            self._move_stage(self._x_mm, self._y_mm, timeout_s=15)
            self._append_hw_log(f"Stage returned to point: ({self._x_mm:.3f}, {self._y_mm:.3f}) mm")
        except Exception:
            pass

        if not self._zone_technical_imports_available():
            pm.logger.error("Cannot start attenuation capture - technical imports not available")
            try:
                self._append_capture_log("Error: technical imports unavailable for attenuation")
            except Exception:
                pass
            return

        self._append_capture_log(
            f"Attenuation I capture: frames={frames}, t={short_t:.6f}s"
        )

        container_version = get_container_version(
            self.config if hasattr(self, "config") else None
        )
        CaptureWorker = self._get_zone_technical_module("CaptureWorker")
        self._attn2_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=short_t,
            txt_filename_base=txt_filename_base,
            frames=frames,
            naming_mode="attenuation_with",
            container_version=container_version,
            hardware_client=getattr(self, "hardware_client", None),
        )
        self._attn2_thread = pm.QThread()
        self._attn2_worker.moveToThread(self._attn2_thread)
        self._attn2_thread.started.connect(self._attn2_worker.run)

        def _after_attn_with(success2, result_files2):
            if success2:
                self._append_capture_log("Attenuation I capture complete")
            else:
                self._append_capture_log("Attenuation I capture failed")

            moved_map = result_files2 or {}

            try:
                self._record_attenuation_files("with_sample", moved_map)
            except Exception:
                pass

            if hasattr(self, "session_manager") and self.session_manager.is_session_active():
                try:
                    import numpy as np

                    all_data = {}
                    for alias, npy_file in moved_map.items():
                        if npy_file:
                            all_data[alias] = np.load(npy_file)

                    if all_data:
                        detector_lookup = {d.get("alias"): d for d in self.config.get("detectors", [])}
                        measurement_data = {}
                        detector_metadata = {}
                        poni_alias_map = {}
                        for alias, signal in all_data.items():
                            detector_meta = detector_lookup.get(alias, {})
                            detector_id = detector_meta.get("id", alias)
                            measurement_data[detector_id] = signal
                            detector_metadata[detector_id] = {
                                "integration_time_ms": short_t * 1000,
                                "detector_id": detector_id,
                                "timestamp": self._timestamp,
                                "point_position_mm": [self._x_mm, self._y_mm],
                                "n_frames": frames,
                            }
                            poni_alias_map[alias] = detector_id

                        self.session_manager.add_attenuation_measurement(
                            measurement_data=measurement_data,
                            detector_metadata=detector_metadata,
                            poni_alias_map=poni_alias_map,
                            mode="with",
                        )
                        session_point_index = self._current_session_point_index()
                        self._append_session_log(
                            f"I saved to session container at point {session_point_index}"
                        )
                        try:
                            self.session_manager.link_attenuation_to_points(
                                num_points=1,
                                start_point_idx=session_point_index,
                            )
                            pm.logger.info(
                                f"Linked attenuation to point {session_point_index}"
                            )
                            self._append_session_log(
                                f"Attenuation linked to point {session_point_index}"
                            )
                        except Exception as e:
                            pm.logger.warning(f"Failed to link attenuation to point: {e}", exc_info=True)
                            self._append_session_log(
                                f"Attenuation link failed: {type(e).__name__}"
                            )

                        pm.logger.info(
                            f"Added I (with sample) to session container at point {session_point_index}",
                            detectors=list(all_data.keys()),
                        )
                except Exception as e:
                    pm.logger.error(f"Failed to add I to session container: {e}", exc_info=True)
                    self._append_session_log(f"I session save failed: {type(e).__name__}")

            self._append_capture_log("Starting normal capture after attenuation")
            self._start_normal_capture(txt_filename_base)

        self._attn2_worker.finished.connect(_after_attn_with)
        self._attn2_worker.finished.connect(self._attn2_thread.quit)
        self._attn2_worker.finished.connect(self._attn2_worker.deleteLater)
        self._attn2_thread.finished.connect(self._attn2_thread.deleteLater)
        self._attn2_thread.start()
