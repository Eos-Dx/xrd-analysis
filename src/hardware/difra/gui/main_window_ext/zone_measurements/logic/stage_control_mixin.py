# zone_measurements/logic/stage_control_mixin.py

import json
import logging
import os
import socket
import time
import uuid
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt

from hardware.difra.gui.main_window_ext.zone_measurements.logic.stage_manual_motion_mixin import (
    StageManualMotionMixin,
)

class StageControlMixin(StageManualMotionMixin):
    _PIXET_DETECTOR_TYPES = {"Pixet", "PixetLegacy", "PixetSidecar"}

    def _append_hw_log(self, message: str) -> None:
        try:
            self._append_measurement_log(f"[HW] {message}")
        except Exception:
            pass

    @staticmethod
    def _active_detector_configs(config: Dict) -> List[Dict]:
        cfg = config or {}
        detectors = list(cfg.get("detectors", []) or [])
        dev_mode = bool(cfg.get("DEV", False))
        selected_ids = (
            cfg.get("dev_active_detectors", [])
            if dev_mode
            else cfg.get("active_detectors", [])
        )
        return [det for det in detectors if det.get("id") in selected_ids]

    @classmethod
    def _is_sidecar_required_for_config(cls, config: Dict) -> bool:
        for det in cls._active_detector_configs(config):
            det_type = str(det.get("type", "")).strip()
            if det_type in cls._PIXET_DETECTOR_TYPES:
                return True
        return False

    @classmethod
    def _resolve_sidecar_endpoint_for_config(
        cls, config: Dict, env: Optional[Dict[str, str]] = None
    ) -> Tuple[str, int]:
        env_vars = env if env is not None else os.environ

        host_env = str(env_vars.get("PIXET_SIDECAR_HOST", "")).strip()
        port_env_raw = str(env_vars.get("PIXET_SIDECAR_PORT", "")).strip()
        try:
            port_env = int(port_env_raw) if port_env_raw else 0
        except Exception:
            port_env = 0

        if host_env and port_env > 0:
            return host_env, port_env

        for det in cls._active_detector_configs(config):
            det_type = str(det.get("type", "")).strip()
            if det_type not in cls._PIXET_DETECTOR_TYPES:
                continue
            sidecar_cfg = det.get("pixet_sidecar", {}) or {}
            host_cfg = str(
                sidecar_cfg.get("host", det.get("sidecar_host", ""))
            ).strip()
            port_cfg_raw = str(
                sidecar_cfg.get("port", det.get("sidecar_port", ""))
            ).strip()
            try:
                port_cfg = int(port_cfg_raw) if port_cfg_raw else 0
            except Exception:
                port_cfg = 0

            host = host_env or host_cfg or "127.0.0.1"
            port = port_env if port_env > 0 else (port_cfg if port_cfg > 0 else 51001)
            return host, port

        host = host_env or "127.0.0.1"
        port = port_env if port_env > 0 else 51001
        return host, port

    @staticmethod
    def _probe_sidecar_endpoint(
        host: str, port: int, timeout_s: float = 0.35
    ) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        req = {
            "id": f"gui-heartbeat-{uuid.uuid4().hex[:8]}",
            "cmd": "ping",
            "args": {},
        }
        payload = (json.dumps(req, ensure_ascii=True) + "\n").encode("utf-8")
        try:
            with socket.create_connection((host, int(port)), timeout=timeout_s) as sock:
                sock.settimeout(timeout_s)
                sock.sendall(payload)
                response_bytes = b""
                while b"\n" not in response_bytes:
                    chunk = sock.recv(4096)
                    if not chunk:
                        elapsed = (time.perf_counter() - start) * 1000.0
                        return False, "connection closed by sidecar", elapsed
                    response_bytes += chunk
            line = response_bytes.split(b"\n", 1)[0]
            response = json.loads(line.decode("utf-8"))
            if not bool(response.get("ok", False)):
                elapsed = (time.perf_counter() - start) * 1000.0
                return False, str(response.get("error", "sidecar ping failed")), elapsed
            result = response.get("result", {}) or {}
            status = str(result.get("status", "")).strip().lower()
            elapsed = (time.perf_counter() - start) * 1000.0
            if status != "ok":
                return False, f"unexpected ping status: {status or 'missing'}", elapsed
            return True, "ok", elapsed
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            return False, str(exc), elapsed

    def _set_sidecar_indicator(
        self,
        *,
        required: bool,
        alive: bool,
        host: str,
        port: int,
        latency_ms: Optional[float],
        error: str = "",
    ) -> None:
        if not hasattr(self, "sidecarIndicator") or not hasattr(self, "sidecarStatusLabel"):
            return

        if not required:
            self.sidecarIndicator.setStyleSheet(
                "background-color: gray; border-radius: 8px;"
            )
            self.sidecarStatusLabel.setText("N/A (no PIXET sidecar detector active)")
            return

        endpoint = f"{host}:{port}"
        if alive:
            self.sidecarIndicator.setStyleSheet(
                "background-color: green; border-radius: 8px;"
            )
            hb = f"{latency_ms:.0f}ms" if latency_ms is not None else "n/a"
            self.sidecarStatusLabel.setText(f"ACTIVE {endpoint} | hb {hb}")
            return

        self.sidecarIndicator.setStyleSheet(
            "background-color: red; border-radius: 8px;"
        )
        err = error.strip() or "heartbeat failed"
        self.sidecarStatusLabel.setText(f"DOWN {endpoint} | {err}")

    def _set_sidecar_lock_state(self, locked: bool, reason: str = "") -> None:
        previous = bool(getattr(self, "_sidecar_locked", False))
        self._sidecar_locked = bool(locked)
        self._sidecar_lock_reason = str(reason or "")
        if previous == self._sidecar_locked:
            return

        if self._sidecar_locked:
            self._append_hw_log(
                f"A2K sidecar heartbeat lost; measurement controls locked ({self._sidecar_lock_reason or 'unknown'})"
            )
            if hasattr(self, "hardware_state_changed"):
                self.hardware_state_changed.emit(False)
        else:
            self._append_hw_log("A2K sidecar heartbeat restored")
            if hasattr(self, "hardware_state_changed") and getattr(
                self, "hardware_initialized", False
            ):
                self.hardware_state_changed.emit(True)

    def refresh_sidecar_status(self, show_message: bool = False) -> bool:
        required = self._is_sidecar_required_for_config(getattr(self, "config", {}))
        host, port = self._resolve_sidecar_endpoint_for_config(
            getattr(self, "config", {})
        )

        if not required:
            self._set_sidecar_indicator(
                required=False,
                alive=True,
                host=host,
                port=port,
                latency_ms=None,
                error="",
            )
            self._set_sidecar_lock_state(False, "")
            self._sidecar_alive = True
            return True

        previous_alive = bool(getattr(self, "_sidecar_alive", False))
        was_locked = bool(getattr(self, "_sidecar_locked", False))
        alive, error, latency_ms = self._probe_sidecar_endpoint(host, port)
        self._sidecar_alive = bool(alive)
        self._set_sidecar_indicator(
            required=True,
            alive=alive,
            host=host,
            port=port,
            latency_ms=latency_ms,
            error=error,
        )
        if alive:
            self._set_sidecar_lock_state(False, "")
            if was_locked and hasattr(self, "_apply_readiness_to_controls"):
                self._apply_readiness_to_controls(bool(getattr(self, "hardware_initialized", False)))
            return True

        self._set_sidecar_lock_state(True, error)
        if (not was_locked) and hasattr(self, "_apply_readiness_to_controls"):
            self._apply_readiness_to_controls(False)

        should_warn = bool(show_message) or (
            bool(getattr(self, "hardware_initialized", False)) and previous_alive
        )
        if should_warn:
            try:
                from PyQt5.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "A2K Sidecar Disconnected",
                    f"Detector sidecar heartbeat failed at {host}:{port}.\n\n"
                    f"Reason: {error}\n\n"
                    "Measurement controls are locked until sidecar is active again.",
                )
            except Exception:
                pass
        return False

    @staticmethod
    def _detector_protocol_from_class(controller_class_name: str) -> str:
        if controller_class_name == "PixetSidecarDetectorController":
            return "legacy-sidecar"
        if controller_class_name == "PixetLegacyDetectorController":
            return "legacy-direct"
        if controller_class_name == "PixetDetectorController":
            return "ctypes-direct"
        if controller_class_name == "DummyDetectorController":
            return "demo"
        return "direct"

    @staticmethod
    def _stage_protocol_from_class(controller_class_name: str) -> str:
        if controller_class_name == "XYStageLibController":
            return "ctypes-dll"
        if controller_class_name == "MarlinStageController":
            return "serial"
        if controller_class_name == "DummyStageController":
            return "demo"
        return "direct"

    def _log_hardware_init_details(self, client, stage_ok: bool, det_ok: bool, source: str) -> None:
        backend_mode = str(getattr(client, "last_backend", "grpc")).strip() or "grpc"
        detector_backend = str(os.environ.get("DETECTOR_BACKEND", "")).strip() or "unset"

        stage_controller = getattr(client, "stage_controller", None)
        if stage_controller is not None:
            stage_cls = stage_controller.__class__.__name__
            stage_protocol = self._stage_protocol_from_class(stage_cls)
        elif backend_mode == "grpc" and stage_ok:
            stage_cls = "GrpcMotionProxy"
            stage_protocol = "grpc"
        else:
            stage_cls = "None"
            stage_protocol = "direct"
        stage_cfg = self._selected_stage_config()
        stage_type = stage_cfg.get("type", "unknown")
        stage_alias = stage_cfg.get("alias", "unknown")

        logging.info(
            "Hardware init (%s): stage_ok=%s detector_ok=%s backend=%s stage_alias=%s stage_type=%s stage_class=%s stage_protocol=%s detector_backend=%s",
            source,
            stage_ok,
            det_ok,
            backend_mode,
            stage_alias,
            stage_type,
            stage_cls,
            stage_protocol,
            detector_backend,
        )
        self._append_hw_log(
            f"Stage {stage_alias}: {stage_type} via {stage_protocol} ({stage_cls})"
        )

        detector_cfg_by_alias = {
            str(det.get("alias", "")): det for det in self.config.get("detectors", [])
        }
        for alias, controller in (getattr(client, "detector_controllers", {}) or {}).items():
            cls_name = controller.__class__.__name__
            det_protocol = self._detector_protocol_from_class(cls_name)
            det_cfg = detector_cfg_by_alias.get(str(alias), {})
            det_type = det_cfg.get("type", "unknown")
            det_id = det_cfg.get("id", "unknown")
            logging.info(
                "Detector active (%s): alias=%s id=%s type=%s class=%s protocol=%s backend=%s",
                source,
                alias,
                det_id,
                det_type,
                cls_name,
                det_protocol,
                detector_backend,
            )
            self._append_hw_log(
                f"Detector {alias}: {det_type} via {det_protocol} ({cls_name})"
            )

    def _ensure_hardware_client(self):
        if getattr(self, "hardware_client", None) is None:
            from hardware.difra.hardware.hardware_client import create_hardware_client

            self.hardware_client = create_hardware_client(self.config)
        return self.hardware_client

    def _selected_stage_config(self) -> Dict:
        cfg = self.config if hasattr(self, "config") and self.config else {}
        stages = cfg.get("translation_stages", [])
        selected_ids = (
            cfg.get("dev_active_stages", [])
            if cfg.get("DEV", False)
            else cfg.get("active_translation_stages", [])
        )
        for stage in stages:
            if stage.get("id") in selected_ids:
                return stage
        return stages[0] if stages else {}

    def _get_stage_limits(self) -> Optional[Dict[str, Tuple[float, float]]]:
        if hasattr(self, "stage_controller") and self.stage_controller is not None:
            try:
                if hasattr(self.stage_controller, "get_limits"):
                    return self.stage_controller.get_limits()
            except Exception:
                pass

        stage_cfg = self._selected_stage_config()
        limits_cfg = stage_cfg.get("settings", {}).get("limits_mm", {})
        try:
            x_limits = limits_cfg.get("x", [-14.0, 14.0])
            y_limits = limits_cfg.get("y", [-14.0, 14.0])
            return {
                "x": (float(x_limits[0]), float(x_limits[1])),
                "y": (float(y_limits[0]), float(y_limits[1])),
            }
        except Exception:
            return None

    def _get_home_load_positions(self) -> Dict[str, Tuple[float, float]]:
        defaults = {"home": (9.25, 6.0), "load": (-13.9, -6.0)}
        if hasattr(self, "stage_controller") and self.stage_controller is not None:
            try:
                if hasattr(self.stage_controller, "get_home_load_positions"):
                    return self.stage_controller.get_home_load_positions()
            except Exception:
                pass

        stage_cfg = self._selected_stage_config()
        settings = stage_cfg.get("settings", {})

        def _parse_pair(value, fallback):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (float(value[0]), float(value[1]))
            return fallback

        return {
            "home": _parse_pair(settings.get("home"), defaults["home"]),
            "load": _parse_pair(settings.get("load"), defaults["load"]),
        }

    def _apply_readiness_to_controls(self, hardware_ok: bool) -> None:
        sidecar_ok = not bool(getattr(self, "_sidecar_locked", False))
        effective_ok = bool(hardware_ok and sidecar_ok)
        move_ready = hardware_ok
        home_ready = hardware_ok
        exposure_ready = hardware_ok

        try:
            readiness = self._ensure_hardware_client().get_command_readiness()
            move_ready = readiness.get(("Motion", "MoveTo"), None)
            move_ready = move_ready.ready if move_ready is not None else hardware_ok
            home_ready = readiness.get(("Motion", "Home"), None)
            home_ready = home_ready.ready if home_ready is not None else hardware_ok
            exposure_ready = readiness.get(("Acquisition", "StartExposure"), None)
            exposure_ready = (
                exposure_ready.ready if exposure_ready is not None else hardware_ok
            )
        except Exception as exc:
            logging.debug("Failed to fetch command readiness: %s", exc)

        self.start_btn.setEnabled(effective_ok and exposure_ready and move_ready)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        if hasattr(self, "skip_btn") and self.skip_btn is not None:
            self.skip_btn.setEnabled(False)
        self.xPosSpin.setEnabled(effective_ok and move_ready)
        self.yPosSpin.setEnabled(effective_ok and move_ready)
        self.gotoBtn.setEnabled(effective_ok and move_ready)
        if hasattr(self, "homeBtn"):
            self.homeBtn.setEnabled(effective_ok and home_ready)
        if hasattr(self, "loadPosBtn"):
            self.loadPosBtn.setEnabled(effective_ok and move_ready)

    def sync_hardware_state_from_backend(self) -> None:
        """Mirror backend initialization state in UI if stack is already running."""
        self.refresh_sidecar_status(show_message=False)
        try:
            client = self._ensure_hardware_client()
            readiness = client.get_command_readiness()
        except Exception as exc:
            logging.debug("Backend state sync skipped: %s", exc)
            return

        motion_item = readiness.get(("DeviceInitialization", "InitializeMotion"))
        detector_item = readiness.get(("DeviceInitialization", "InitializeDetector"))
        stage_initialized = bool(motion_item is not None and not motion_item.ready)
        detector_initialized = bool(detector_item is not None and not detector_item.ready)
        hardware_ok = bool(stage_initialized and detector_initialized)

        self._apply_readiness_to_controls(hardware_ok)
        if not hardware_ok:
            return

        self.hardware_controller = client.hardware_controller
        self.stage_controller = client.stage_controller
        self.detector_controller = client.detector_controllers
        self.xyStageIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        self.cameraIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        self.initializeBtn.setText("Deinitialize Hardware")
        self._log_hardware_init_details(
            client,
            stage_ok=stage_initialized,
            det_ok=detector_initialized,
            source="sync",
        )
        if not getattr(self, "hardware_initialized", False):
            self.hardware_initialized = True
            if hasattr(self, "hardware_state_changed"):
                self.hardware_state_changed.emit(True)
            try:
                self.refresh_detector_tabs_for_mode_switch()
            except Exception as exc:
                logging.debug("Detector tab refresh during backend sync failed: %s", exc)
        self.update_xy_pos()

    def toggle_hardware(self):
        """
        Toggle hardware initialization state and keep GUI routed through a dual-path
        hardware client (gRPC primary, direct fallback).
        """
        if not getattr(self, "hardware_initialized", False):
            from PyQt5.QtWidgets import QMessageBox

            if not self.refresh_sidecar_status(show_message=True):
                self._append_hw_log("Initialize blocked: A2K sidecar heartbeat unavailable")
                return

            try:
                client = self._ensure_hardware_client()
                self._append_hw_log("Initializing hardware...")
                res_xystage = client.initialize_motion()
                res_det = client.initialize_detector()
            except Exception as exc:
                logging.exception("Hardware initialization failed")
                self._append_hw_log(f"Initialization failed: {exc}")
                QMessageBox.warning(
                    self,
                    "Hardware Initialization Failed",
                    f"Could not initialize hardware:\n{exc}",
                )
                return

            self.hardware_controller = client.hardware_controller
            self.stage_controller = client.stage_controller
            self.detector_controller = client.detector_controllers

            self.xyStageIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
                if res_xystage
                else "background-color: red; border-radius: 10px;"
            )
            self.cameraIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
                if res_det
                else "background-color: red; border-radius: 10px;"
            )

            ok = bool(res_xystage and res_det)
            self._apply_readiness_to_controls(ok)
            self._log_hardware_init_details(
                client,
                stage_ok=res_xystage,
                det_ok=res_det,
                source="user-init",
            )

            if ok:
                self.refresh_detector_tabs_for_mode_switch()
                self.initializeBtn.setText("Deinitialize Hardware")
                self.hardware_initialized = True
                if hasattr(self, "hardware_state_changed"):
                    self.hardware_state_changed.emit(True)
                self._append_hw_log("Initialization complete")
            else:
                self._append_hw_log("Initialization incomplete (stage/detector failed)")
                QMessageBox.warning(
                    self,
                    "Hardware Initialization Failed",
                    "Detector and/or motion initialization did not complete. "
                    "Check hardware connections and logs.",
                )
        else:
            try:
                self._append_hw_log("Deinitializing hardware...")
                self._ensure_hardware_client().deinitialize()
            except Exception as exc:
                logging.warning("Error deinitializing hardware: %s", exc)
                self._append_hw_log(f"Deinitialize warning: {exc}")

            self.clear_detector_param_tabs()
            self.xyStageIndicator.setStyleSheet(
                "background-color: gray; border-radius: 10px;"
            )
            self.cameraIndicator.setStyleSheet(
                "background-color: gray; border-radius: 10px;"
            )
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            if hasattr(self, "skip_btn") and self.skip_btn is not None:
                self.skip_btn.setEnabled(False)
            self.xPosSpin.setEnabled(False)
            self.yPosSpin.setEnabled(False)
            self.gotoBtn.setEnabled(False)
            if hasattr(self, "homeBtn"):
                self.homeBtn.setEnabled(False)
            if hasattr(self, "loadPosBtn"):
                self.loadPosBtn.setEnabled(False)
            self.initializeBtn.setText("Initialize Hardware")
            self.hardware_initialized = False
            if hasattr(self, "hardware_state_changed"):
                self.hardware_state_changed.emit(False)
            self._append_hw_log("Deinitialized")

    def update_xy_pos(self):
        """
        Updates the current XY position display and beam cross overlay on the scene.
        Note: Does NOT update the Stage X/Y spinboxes - those are for user input only.
        """
        if getattr(self, "hardware_initialized", False):
            try:
                if getattr(self, "hardware_client", None) is not None:
                    x, y = self.hardware_client.get_xy_position()
                elif hasattr(self, "stage_controller") and self.stage_controller is not None:
                    x, y = self.stage_controller.get_xy_position()
                else:
                    x, y = 0.0, 0.0
                    if not getattr(self, "_xy_pos_stage_unavailable_logged", False):
                        logging.debug("Stage controller unavailable while hardware_initialized=True")
                        self._xy_pos_stage_unavailable_logged = True
                if getattr(self, "_xy_pos_stage_unavailable_logged", False):
                    self._xy_pos_stage_unavailable_logged = False

                position_text = f"Current XY: ({x:.3f}, {y:.3f}) mm"
                if hasattr(self, "currentPositionLabel"):
                    self.currentPositionLabel.setText(position_text)
                if hasattr(self, "zoneCurrentPositionLabel"):
                    self.zoneCurrentPositionLabel.setText(position_text)
            except Exception as exc:
                if not getattr(self, "_xy_pos_error_logged", False):
                    logging.warning("Error reading stage position: %s", exc)
                    self._xy_pos_error_logged = True
                x, y = 0, 0
                error_text = "Current XY: (Error reading position)"
                if hasattr(self, "currentPositionLabel"):
                    self.currentPositionLabel.setText(error_text)
                if hasattr(self, "zoneCurrentPositionLabel"):
                    self.zoneCurrentPositionLabel.setText(error_text)
        else:
            x, y = 0, 0
            if getattr(self, "_xy_pos_error_logged", False):
                self._xy_pos_error_logged = False
            not_init_text = "Current XY: (Not initialized)"
            if hasattr(self, "currentPositionLabel"):
                self.currentPositionLabel.setText(not_init_text)
            if hasattr(self, "zoneCurrentPositionLabel"):
                self.zoneCurrentPositionLabel.setText(not_init_text)

        old = self.image_view.points_dict.get("beam", [])
        try:
            for itm in old:
                self.image_view.scene.removeItem(itm)
        except Exception as exc:
            print("Error removing old beam cross:", exc)

        x_pix, y_pix = self.mm_to_pixels(x, y)

        if x_pix >= 0 and y_pix >= 0:
            size = 15
            from PyQt5.QtGui import QPen

            pen = QPen(Qt.black, 5)
            hl = self._add_beam_line(x_pix - size, y_pix, x_pix + size, y_pix, pen)
            vl = self._add_beam_line(x_pix, y_pix - size, x_pix, y_pix + size, pen)
            self.image_view.points_dict["beam"] = [hl, vl]
        else:
            self.image_view.points_dict["beam"] = []
