"""Manual stage motion helpers extracted from StageControlMixin."""

import logging
import threading


class StageManualMotionMixin:
    """Async/manual movement controls (goto/home/load)."""

    def _set_manual_motion_controls_enabled(self, enabled: bool) -> None:
        controls = ("gotoBtn", "homeBtn", "loadPosBtn", "initializeBtn")
        for name in controls:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.setEnabled(bool(enabled))
            except Exception:
                pass

    def _start_manual_motion_async(
        self,
        *,
        label: str,
        worker_fn,
        on_success,
        on_error,
    ) -> None:
        from PyQt5.QtCore import QTimer

        if getattr(self, "_manual_motion_in_progress", False):
            return
        self._manual_motion_in_progress = True
        self._manual_motion_done = False
        self._manual_motion_error = None
        self._manual_motion_result = None
        self._manual_motion_label = str(label)
        self._set_manual_motion_controls_enabled(False)

        def _worker():
            try:
                self._manual_motion_result = worker_fn()
            except Exception as exc:
                self._manual_motion_error = exc
            finally:
                self._manual_motion_done = True

        self._manual_motion_thread = threading.Thread(
            target=_worker,
            name=f"manual-motion-{label}",
            daemon=True,
        )
        self._manual_motion_thread.start()

        self._manual_motion_poll_timer = QTimer(self)
        self._manual_motion_poll_timer.setInterval(500)

        def _poll():
            try:
                self.update_xy_pos()
            except Exception:
                pass

            if not getattr(self, "_manual_motion_done", False):
                return

            self._manual_motion_poll_timer.stop()
            self._manual_motion_poll_timer.deleteLater()
            self._manual_motion_poll_timer = None
            self._manual_motion_in_progress = False
            self._set_manual_motion_controls_enabled(True)

            err = getattr(self, "_manual_motion_error", None)
            if err is not None:
                on_error(err)
                return

            res = getattr(self, "_manual_motion_result", None)
            on_success(res)

        self._manual_motion_poll_timer.timeout.connect(_poll)
        self._manual_motion_poll_timer.start()

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

        def _worker():
            client = self._ensure_hardware_client()
            client.move_to(x, axis="x", timeout_s=25)
            return client.move_to(y, axis="y", timeout_s=25)

        def _on_success(res):
            try:
                new_x, new_y = res
            except Exception:
                new_x, new_y = self.hardware_client.get_xy_position()
            self.update_xy_pos()
            logging.info(
                "Successfully moved to goto position: (%.3f, %.3f)", new_x, new_y
            )
            self._append_hw_log(f"MoveTo complete: ({new_x:.3f}, {new_y:.3f}) mm")

        def _on_error(exc):
            if isinstance(exc, TimeoutError):
                logging.error("Stage movement timeout occurred during goto operation")
                self._append_hw_log("MoveTo timeout")
                QMessageBox.warning(
                    self,
                    "Stage Timeout",
                    "Stage movement timed out. Please check the hardware and try again.",
                )
                return
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
            self._append_hw_log(f"MoveTo error: {exc}")

        self._start_manual_motion_async(
            label="goto",
            worker_fn=_worker,
            on_success=_on_success,
            on_error=_on_error,
        )

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

            def _worker():
                client = self._ensure_hardware_client()
                client.move_to(home_x, axis="x", timeout_s=25)
                return client.move_to(home_y, axis="y", timeout_s=25)

            def _on_success(res):
                try:
                    new_x, new_y = res
                except Exception:
                    new_x, new_y = self.hardware_client.get_xy_position()
                logging.info(
                    "Successfully moved to home position: (%.3f, %.3f)", new_x, new_y
                )
                self.update_xy_pos()
                self._append_hw_log(f"Home complete: ({new_x:.3f}, {new_y:.3f}) mm")

            def _on_error(exc):
                if isinstance(exc, TimeoutError):
                    logging.error("Stage movement timeout occurred during home operation")
                    self._append_hw_log("Home timeout")
                    QMessageBox.warning(
                        self,
                        "Stage Timeout",
                        "Stage movement timed out. Please check the hardware and try again.",
                    )
                    return
                logging.error("Error during home operation: %s", exc)
                self._append_hw_log(f"Home error: {exc}")
                QMessageBox.warning(
                    self, "Stage Error", f"Error moving to home position: {str(exc)}"
                )

            self._start_manual_motion_async(
                label="home",
                worker_fn=_worker,
                on_success=_on_success,
                on_error=_on_error,
            )
        except Exception as exc:
            logging.error("Error preparing home operation: %s", exc)
            self._append_hw_log(f"Home error: {exc}")
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

            def _worker():
                client = self._ensure_hardware_client()
                client.move_to(load_x, axis="x", timeout_s=25)
                return client.move_to(load_y, axis="y", timeout_s=25)

            def _on_success(res):
                try:
                    new_x, new_y = res
                except Exception:
                    new_x, new_y = self.hardware_client.get_xy_position()
                logging.info(
                    "Successfully moved to load position: (%.3f, %.3f)", new_x, new_y
                )
                self.update_xy_pos()
                self._append_hw_log(f"Load complete: ({new_x:.3f}, {new_y:.3f}) mm")

            def _on_error(exc):
                if isinstance(exc, TimeoutError):
                    logging.error("Stage movement timeout occurred during load operation")
                    self._append_hw_log("Load timeout")
                    QMessageBox.warning(
                        self,
                        "Stage Timeout",
                        "Stage movement timed out. Please check the hardware and try again.",
                    )
                    return
                logging.error("Error during load operation: %s", exc)
                self._append_hw_log(f"Load error: {exc}")
                QMessageBox.warning(
                    self, "Stage Error", f"Error moving to load position: {str(exc)}"
                )

            self._start_manual_motion_async(
                label="load",
                worker_fn=_worker,
                on_success=_on_success,
                on_error=_on_error,
            )
        except Exception as exc:
            logging.error("Error preparing load operation: %s", exc)
            self._append_hw_log(f"Load error: {exc}")
            QMessageBox.warning(
                self, "Stage Error", f"Error moving to load position: {str(exc)}"
            )
