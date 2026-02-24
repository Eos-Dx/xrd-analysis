import logging
import os
import subprocess
import sys
import tempfile
import time

from hardware.difra.gui.container_api import get_container_version

logger = logging.getLogger(__name__)


def _tm():
    from hardware.difra.gui.main_window_ext import technical_measurements as tm

    return tm


class TechnicalCaptureMixin:
    def _start_capture(self, typ: str):
        tm = _tm()
        if not self._technical_imports_available():
            error_msg = (
                f"Cannot start {typ} capture - technical measurement modules failed to import. "
                "Check application logs for detailed error information. "
                "Common causes: missing pyFAI or fabio dependencies."
            )
            self._log_technical_event(error_msg)
            logger.error(error_msg)
            tm.QMessageBox.warning(
                self,
                "Import Error",
                error_msg + "\n\nPlease check the application log file for detailed traceback.",
            )
            return

        counter_attr = f"{typ.lower()}_counter"
        count = getattr(self, counter_attr, 0) + 1
        setattr(self, counter_attr, count)

        validate_folder = self._get_technical_module("validate_folder")
        folder = validate_folder(self.folderLE.text())
        base = self._file_base(typ)
        base_with_count = f"{base}_{count:03d}"
        ts = time.strftime("%Y%m%d_%H%M%S")
        integration_time_s = float(self.integrationTimeSpin.value())
        frames = int(self.captureFramesSpin.value())
        t_token = f"{integration_time_s:.6f}s"
        txt_filename_base = os.path.join(
            folder,
            f"{base_with_count}_{ts}_{t_token}_{frames}frames",
        )

        stage_controller = None
        if hasattr(self, "hardware_controller") and self.hardware_controller:
            stage_controller = self.hardware_controller.stage_controller
        elif hasattr(self, "stage_controller"):
            stage_controller = self.stage_controller
        elif hasattr(self, "hardware_client") and self.hardware_client:
            stage_controller = self.hardware_client.stage_controller

        enable_continuous_movement = (
            getattr(self, "moveContinuousCheck", None) is not None
            and self.moveContinuousCheck.isChecked()
        )
        movement_radius = (
            self.movementRadiusSpin.value()
            if getattr(self, "movementRadiusSpin", None) is not None
            else 2.0
        )

        logger.debug(
            f"Starting {typ} capture: integration_time={integration_time_s}s, frames={frames}, "
            f"continuous_movement={enable_continuous_movement}, radius={movement_radius}mm"
        )
        self._pending_aux_capture_metadata = {
            "integration_time_ms": integration_time_s * 1000.0,
            "n_frames": frames,
        }

        container_version = get_container_version(
            self.config if hasattr(self, "config") else None
        )
        CaptureWorker = self._get_technical_module("CaptureWorker")
        worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=integration_time_s,
            txt_filename_base=txt_filename_base,
            frames=frames,
            naming_mode="normal",
            continuous_movement_controller=self.continuous_movement_controller,
            stage_controller=stage_controller,
            enable_continuous_movement=enable_continuous_movement,
            movement_radius=movement_radius,
            container_version=container_version,
            hardware_client=getattr(self, "hardware_client", None),
        )
        thread = tm.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _cleanup(success, result_files, t=typ):
            try:
                self._on_capture_done(
                    success,
                    result_files,
                    t,
                    error_messages=getattr(worker, "error_messages", None),
                )
            except Exception as e:
                logger.error(f"Error in _on_capture_done for {t}: {e}", exc_info=True)
            finally:
                worker.deleteLater()
                thread.quit()
                thread.deleteLater()
                self._capture_workers.remove(worker)

        worker.finished.connect(_cleanup)
        thread.start()

        if not hasattr(self, "_capture_workers"):
            self._capture_workers = []
        self._capture_workers.append(worker)

    def _on_capture_done(
        self,
        success: bool,
        result_files: dict,
        typ: str,
        error_messages=None,
    ):
        if not success:
            details = [
                str(msg).strip()
                for msg in (error_messages or [])
                if str(msg).strip()
            ]
            if details:
                joined = " | ".join(details[:3])
                self._log_technical_event(f"{typ} capture failed: {joined}")
            else:
                self._log_technical_event(f"{typ} capture failed")
            logger.warning(
                "[%s] capture failed; details=%s; result_files=%s",
                typ,
                details,
                result_files,
            )
            self._aux_timer.stop()
            self._aux_status.setText("")
            return

        self._log_technical_event(f"{typ} capture successful: {len(result_files)} files")
        logger.info(f"[{typ}] capture successful: {list(result_files.keys())}")
        self._aux_timer.stop()
        self._aux_status.setText("Processing...")

        if not self._technical_imports_available():
            error_msg = "Cannot process files - technical imports not available"
            self._log_technical_event(error_msg)
            logger.error(error_msg)
            self._aux_status.setText("Import error")
            return

        self._log_technical_event("Processing measurement files...")
        MeasurementWorker = self._get_technical_module("MeasurementWorker")
        worker = MeasurementWorker(
            filenames=result_files,
            frames=1,
            average_frames=False,
        )
        worker.add_aux_item.connect(self._add_aux_item_to_list)
        try:
            worker.run()
            if hasattr(self, "_sync_active_technical_container_from_table"):
                self._sync_active_technical_container_from_table(show_errors=True)
        finally:
            self._pending_aux_capture_metadata = None

    def measure_aux(self):
        tm = _tm()
        if not self._technical_imports_available():
            self._log_technical_event("Cannot start Aux measurement - technical imports not available")
            print("Cannot start Aux measurement - technical measurements disabled due to import errors")
            tm.QMessageBox.warning(
                self,
                "Technical Measurements Unavailable",
                "Technical measurements are disabled due to import errors.\n\nCheck the console for details.",
            )
            return

        if hasattr(self, "_ensure_active_technical_container_available"):
            ready = self._ensure_active_technical_container_available(
                for_edit=True,
                prompt_on_locked=True,
            )
            if not ready:
                self._log_technical_event(
                    "Aux measurement cancelled: technical container is not editable"
                )
                return

        self._log_technical_event("Starting auxiliary measurement...")
        self._aux_start = time.time()
        self._aux_spinner_state = 0
        self._aux_status.setText("0 s ⁑")
        self._aux_timer.start()
        self._start_capture("Aux")

    def _open_measurement_from_table(self, row: int, _col: int):
        tm = _tm()
        file_item = self.auxTable.item(row, self.AUX_COL_FILE)
        if not file_item:
            return
        file_path = file_item.data(tm.Qt.UserRole)
        resolved_path = str(file_path or "").strip()
        is_h5_ref = resolved_path.startswith("h5ref://")
        if resolved_path and not is_h5_ref and not os.path.exists(resolved_path):
            folder = (self.folderLE.text() or "").strip()
            candidate = os.path.join(folder, os.path.basename(resolved_path)) if folder else ""
            if candidate and os.path.exists(candidate):
                resolved_path = candidate

        self._log_technical_event(
            f"Opening measurement file: {os.path.basename(resolved_path) if resolved_path else 'Unknown'}"
        )

        if not resolved_path or (not is_h5_ref and not os.path.exists(resolved_path)):
            tm.QMessageBox.warning(
                self,
                "File Not Found",
                f"Measurement file is missing:\n{resolved_path or str(file_path)}",
            )
            self._log_technical_event(
                f"Cannot open measurement: missing file {resolved_path or str(file_path)}"
            )
            return

        alias_cb = self.auxTable.cellWidget(row, self.AUX_COL_ALIAS)
        alias = None
        if isinstance(alias_cb, tm.QComboBox):
            a = alias_cb.currentText().strip()
            if a and a != self.NO_SELECTION_LABEL:
                alias = a

        if not alias:
            disp = file_item.text()
            if ":" in disp:
                alias = disp.split(":", 1)[0].strip()

        if not alias:
            try:
                alias = next(iter(self.detector_controller))
            except Exception:
                alias = None

        if not self._technical_imports_available():
            self._log_technical_event("Cannot open measurement window - technical imports not available")
            return

        show_measurement_window = self._get_technical_module("show_measurement_window")
        try:
            show_measurement_window(
                resolved_path, self.masks.get(alias), self.ponis.get(alias), self
            )
        except Exception as exc:
            self._log_technical_event(f"Failed to open measurement window: {exc}")
            tm.QMessageBox.warning(
                self,
                "Open Measurement Failed",
                f"Could not open measurement file:\n{resolved_path}\n\nError: {exc}",
            )

    def run_pyfai(self):
        self._log_technical_event("Starting PyFAI calibration...")
        env = self.config.get("conda")
        if not env:
            self._log_technical_event("Error: No conda environment configured")
            print("No conda env set in self.config['conda']")
            return

        validate_folder = self._get_technical_module("validate_folder")
        folder = validate_folder(self.folderLE.text())

        if os.name == "nt":
            cmd = f"CALL conda activate {env} " f'&& cd /d "{folder}" ' f"&& pyfai-calib2"
            start_cmd = f'start cmd /K "{cmd}"'
            try:
                subprocess.Popen(start_cmd, shell=True)
                self._log_technical_event("PyFAI calibration launched in new window")
                print("Launched PyFai in new cmd window.")
            except Exception as e:
                self._log_technical_event(f"Failed to launch PyFAI on Windows: {e}")
                print("Failed to launch PyFai on Windows:", e)
            return

        try:
            if sys.platform == "darwin":
                script_content = f"""#!/bin/bash
cd "{folder}"
echo "Starting PyFAI calibration in conda environment: {env}"
echo "Folder: {folder}"
echo ""
conda run -n {env} pyfai-calib2
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to launch PyFAI. Check that:"
    echo "  1. Conda environment '{env}' exists (run: conda env list)"
    echo "  2. pyfai-calib2 is installed (run: conda run -n {env} which pyfai-calib2)"
    echo ""
    echo "Press any key to close..."
    read -n 1
fi
"""
                with tempfile.NamedTemporaryFile(mode="w", suffix=".command", delete=False) as f:
                    f.write(script_content)
                    script_path = f.name
                os.chmod(script_path, 0o755)
                subprocess.Popen(["open", "-a", "Terminal", script_path])
                self._log_technical_event(f"PyFAI calibration script created: {script_path}")
            else:
                bash_cmd = (
                    f'cd "{folder}" && '
                    f'echo "Starting PyFAI in environment: {env}" && '
                    f'conda run -n {env} pyfai-calib2 || '
                    f'(echo "\\nError: Failed to launch PyFAI"; read -p "Press Enter to close...")'
                )
                for terminal in ["gnome-terminal", "konsole", "xterm"]:
                    try:
                        subprocess.Popen([terminal, "--", "bash", "-c", bash_cmd])
                        break
                    except FileNotFoundError:
                        continue
            self._log_technical_event("PyFAI calibration launched in new terminal window")
            print("Launched PyFai in new terminal window.")
        except Exception as e:
            self._log_technical_event(f"Failed to launch PyFAI on Unix: {e}")
            print("Failed to launch PyFai on Unix:", e)

    def _update_aux_status(self):
        elapsed = int(time.time() - self._aux_start)
        spinner = ["⁑", "⁙", "⁹", "⁸", "‼", "‴", "…", "‧", " ", "‏"]
        ch = spinner[self._aux_spinner_state % len(spinner)]
        self._aux_spinner_state += 1
        self._aux_status.setText(f"{elapsed} s {ch}")

        if elapsed > 0 and elapsed % 10 == 0 and self._aux_spinner_state % len(spinner) == 0:
            self._log_technical_event(f"Auxiliary measurement in progress: {elapsed} seconds")
