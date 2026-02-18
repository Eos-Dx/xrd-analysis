# zone_measurements/logic/stage_control_mixin.py

import logging
from typing import Dict, Optional, Tuple

from PyQt5.QtCore import Qt


class StageControlMixin:
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

        self.start_btn.setEnabled(hardware_ok and exposure_ready and move_ready)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.xPosSpin.setEnabled(hardware_ok and move_ready)
        self.yPosSpin.setEnabled(hardware_ok and move_ready)
        self.gotoBtn.setEnabled(hardware_ok and move_ready)
        if hasattr(self, "homeBtn"):
            self.homeBtn.setEnabled(hardware_ok and home_ready)
        if hasattr(self, "loadPosBtn"):
            self.loadPosBtn.setEnabled(hardware_ok and move_ready)

    def toggle_hardware(self):
        """
        Toggle hardware initialization state and keep GUI routed through a dual-path
        hardware client (gRPC primary, direct fallback).
        """
        if not getattr(self, "hardware_initialized", False):
            from PyQt5.QtWidgets import QMessageBox

            try:
                client = self._ensure_hardware_client()
                res_xystage = client.initialize_motion()
                res_det = client.initialize_detector()
            except Exception as exc:
                logging.exception("Hardware initialization failed")
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

            if ok:
                self.refresh_detector_tabs_for_mode_switch()
                self.initializeBtn.setText("Deinitialize Hardware")
                self.hardware_initialized = True
                if hasattr(self, "hardware_state_changed"):
                    self.hardware_state_changed.emit(True)
            else:
                QMessageBox.warning(
                    self,
                    "Hardware Initialization Failed",
                    "Detector and/or motion initialization did not complete. "
                    "Check hardware connections and logs.",
                )
        else:
            try:
                self._ensure_hardware_client().deinitialize()
            except Exception as exc:
                logging.warning("Error deinitializing hardware: %s", exc)

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

    def goto_stage_position(self):
        """
        Moves the stage to the user-specified X/Y coordinates.
        Updates X/Y spin boxes and calls the client.
        """
        from PyQt5.QtWidgets import QMessageBox

        if not getattr(self, "hardware_initialized", False):
            QMessageBox.warning(
                self, "Stage Not Ready", "Stage not initialized; cannot GoTo."
            )
            return

        x = self.xPosSpin.value()
        y = self.yPosSpin.value()
        logging.info("Stage goto operation started: target position (%.3f, %.3f)", x, y)
        try:
            client = self._ensure_hardware_client()
            client.move_to(x, axis="x", timeout_s=25)
            new_x, new_y = client.move_to(y, axis="y", timeout_s=25)
            self.update_xy_pos()
            logging.info("Successfully moved to goto position: (%.3f, %.3f)", new_x, new_y)
        except TimeoutError:
            logging.error("Stage movement timeout occurred during goto operation")
            QMessageBox.warning(
                self,
                "Stage Timeout",
                "Stage movement timed out. Please check the hardware and try again.",
            )
        except Exception as exc:
            try:
                from hardware.difra.hardware.xystages import StageAxisLimitError

                if isinstance(exc, StageAxisLimitError):
                    limits = self._get_stage_limits()
                    if limits:
                        x_min, x_max = limits.get("x", (None, None))
                        y_min, y_max = limits.get("y", (None, None))
                        QMessageBox.warning(
                            self,
                            "Stage Move Error",
                            f"Requested position ({x:.3f}, {y:.3f}) is outside limits:\n"
                            f"X[{x_min:.1f}, {x_max:.1f}] mm, Y[{y_min:.1f}, {y_max:.1f}] mm",
                        )
                    else:
                        QMessageBox.warning(self, "Stage Move Error", str(exc))
                else:
                    QMessageBox.warning(self, "Stage Move Error", str(exc))
            except Exception:
                QMessageBox.warning(self, "Stage Move Error", str(exc))

    def home_stage_button_clicked(self):
        """
        Moves the XY stage to the configured home position.
        """
        from PyQt5.QtWidgets import QMessageBox

        logging.info("Stage home operation started")
        if not getattr(self, "hardware_initialized", False):
            logging.warning("Home operation failed: Stage not initialized")
            print("Stage not initialized.")
            return

        try:
            positions = self._get_home_load_positions()
            home_x, home_y = positions["home"]
            logging.info(
                "Moving to configured home position: (%.3f, %.3f)", home_x, home_y
            )
            client = self._ensure_hardware_client()
            client.move_to(home_x, axis="x", timeout_s=25)
            new_x, new_y = client.move_to(home_y, axis="y", timeout_s=25)
            logging.info(
                "Successfully moved to home position: (%.3f, %.3f)", new_x, new_y
            )
            self.update_xy_pos()
        except TimeoutError:
            logging.error("Stage movement timeout occurred during home operation")
            QMessageBox.warning(
                self,
                "Stage Timeout",
                "Stage movement timed out. Please check the hardware and try again.",
            )
        except Exception as exc:
            logging.error("Error during home operation: %s", exc)
            QMessageBox.warning(
                self, "Stage Error", f"Error moving to home position: {str(exc)}"
            )

    def load_position_button_clicked(self):
        """
        Moves the XY stage to the configured load position.
        """
        from PyQt5.QtWidgets import QMessageBox

        logging.info("Stage load position operation started")
        if not getattr(self, "hardware_initialized", False):
            logging.warning("Load operation failed: Stage not initialized")
            print("Stage not initialized.")
            return

        try:
            positions = self._get_home_load_positions()
            load_x, load_y = positions["load"]
            logging.info(
                "Moving to configured load position: (%.3f, %.3f)", load_x, load_y
            )
            client = self._ensure_hardware_client()
            client.move_to(load_x, axis="x", timeout_s=25)
            new_x, new_y = client.move_to(load_y, axis="y", timeout_s=25)
            logging.info(
                "Successfully moved to load position: (%.3f, %.3f)", new_x, new_y
            )
            self.update_xy_pos()
        except TimeoutError:
            logging.error("Stage movement timeout occurred during load operation")
            QMessageBox.warning(
                self,
                "Stage Timeout",
                "Stage movement timed out. Please check the hardware and try again. That's SAD",
            )
        except Exception as exc:
            logging.error("Error during load operation: %s", exc)
            QMessageBox.warning(
                self, "Stage Error", f"Error moving to load position: {str(exc)}"
            )
