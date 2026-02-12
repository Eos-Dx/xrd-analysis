# zone_measurements/logic/process_mixin.py

import hashlib
import json
import time
import uuid
from copy import copy
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

# Defer all technical imports to avoid pyFAI crashes on startup
# These will be imported only when actually needed
_ZONE_TECHNICAL_IMPORTS_AVAILABLE = None  # None = not yet tested
_zone_technical_modules = {}

def _get_zone_technical_imports():
    """Lazy import of technical modules to avoid startup crashes."""
    global _ZONE_TECHNICAL_IMPORTS_AVAILABLE, _zone_technical_modules
    
    if _ZONE_TECHNICAL_IMPORTS_AVAILABLE is not None:
        return _ZONE_TECHNICAL_IMPORTS_AVAILABLE
    
    try:
        from hardware.difra.gui.technical.capture import (
            CaptureWorker,
            validate_folder,
        )
        from hardware.difra.gui.technical.measurement_worker import MeasurementWorker
        from hardware.difra.gui.technical.widgets import MeasurementHistoryWidget
        
        _zone_technical_modules.update({
            'CaptureWorker': CaptureWorker,
            'validate_folder': validate_folder,
            'MeasurementWorker': MeasurementWorker,
            'MeasurementHistoryWidget': MeasurementHistoryWidget,
        })
        _ZONE_TECHNICAL_IMPORTS_AVAILABLE = True
        return True
    except Exception as e:
        print(f"Warning: Zone technical measurement imports failed: {e}")
        print("Zone measurements will be disabled.")
        _ZONE_TECHNICAL_IMPORTS_AVAILABLE = False
        return False

def _get_zone_technical_module(name):
    """Get a zone technical module by name, with fallback stubs."""
    if _get_zone_technical_imports():
        return _zone_technical_modules.get(name)
    else:
        # Return stub implementations
        stubs = {
            'CaptureWorker': type('CaptureWorker', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'moveToThread': lambda self, thread: None,
                'finished': type('Signal', (), {'connect': lambda self, f: None})()
            }),
            'validate_folder': lambda path: str(path) if path else "",
            'MeasurementWorker': type('MeasurementWorker', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'run': lambda self: None,
                'add_aux_item': type('Signal', (), {'connect': lambda self, f: None})()
            }),
            'MeasurementHistoryWidget': type('MeasurementHistoryWidget', (), {
                '__init__': lambda self, *args, **kwargs: None
            })
        }
        return stubs.get(name)
from hardware.difra.utils.logger import get_module_logger

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

        # ===== PRE-MEASUREMENT PONI UPDATE CONFIRMATION =====
        if not self._confirm_poni_settings_before_measurement():
            return  # User chose to update PONI settings first
        # ==================================

        # ===== PRE-FLIGHT MANDATORY CHECKLIST =====
        try:
            from .preflight_dialog import PreflightDialog

            # Pass session_manager for technical container validation
            d = PreflightDialog(
                self,
                session_manager=getattr(self, 'session_manager', None),
            )
            if d.exec_() != d.Accepted:
                return
        except Exception as e:
            logger.warning(
                "Preflight dialog failed; proceeding without it", error=str(e)
            )
        # ==================================

        # Ensure a session-level calibration group hash exists and is placed in the state before copying
        group_hash = getattr(self, "calibration_group_hash", None)
        if not group_hash:
            try:
                group_hash = uuid.uuid4().hex[:16]
            except Exception:
                group_hash = None
            setattr(self, "calibration_group_hash", group_hash)
        if group_hash:
            try:
                # Store in current state so it propagates everywhere
                if isinstance(getattr(self, "state", None), dict):
                    self.state["CALIBRATION_GROUP_HASH"] = group_hash
            except Exception:
                pass

        try:
            self.state_measurements = copy(self.state)
        except Exception as e:
            logger.error("Error copying state for measurements", error=str(e))
            QMessageBox.warning(self, "No state", "Save it.")
            return  # Exit the function early

        try:
            from hardware.difra.hardware.auxiliary import encode_image_to_base64

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

        # If attenuation is enabled, capture background (without sample) ONCE for this run
        try:
            if (
                hasattr(self, "attenuationCheckBox")
                and self.attenuationCheckBox.isChecked()
            ):
                self._capture_attenuation_background()
        except Exception as e:
            logger.warning(
                "Failed to capture attenuation background; will continue without it",
                error=str(e),
            )

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
        try:
            self._append_measurement_log(
                f"Start: {self.total_points} points, T={self.integration_time:.2f}s"
            )
        except Exception:
            pass

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
        # Ensure CALIBRATION_GROUP_HASH is present in state_measurements
        gh = getattr(self, "calibration_group_hash", None)
        if gh:
            self.state_measurements["CALIBRATION_GROUP_HASH"] = gh
        self.manual_save_state()
        
        # Add points to session container if session is active
        if hasattr(self, 'session_manager') and self.session_manager.is_session_active():
            try:
                # Convert measurement_points to session container format
                points_for_session = []
                for pt in measurement_points:
                    # Get pixel coordinates from point item
                    pt_idx = pt['point_index']
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
                    
                    points_for_session.append({
                        "pixel_coordinates": [float(pixel_x), float(pixel_y)],
                        "physical_coordinates_mm": [pt['x'], pt['y']],
                    })
                
                # Add all points to session
                logger.info("=== SESSION CONTAINER POPULATION ===")
                logger.info(f"Adding {len(points_for_session)} points to session container...")
                self.session_manager.add_points(points_for_session)
                logger.info(f"✓ Added {len(points_for_session)} points to session container")
                
                # Add zones to session if available
                if hasattr(self, '_add_zones_to_session'):
                    logger.info("Adding zones to session container...")
                    num_shapes = len(self.state.get('shapes', []))
                    logger.info(f"Found {num_shapes} shapes in state")
                    self._add_zones_to_session()
                    logger.info("✓ Zones processing complete")
                else:
                    logger.warning("⚠ _add_zones_to_session method not found")
                
                # Add image mapping to session
                if hasattr(self, '_add_mapping_to_session'):
                    logger.info("Adding mapping to session container...")
                    self._add_mapping_to_session()
                    logger.info("✓ Mapping added")
                else:
                    logger.warning("⚠ _add_mapping_to_session method not found")
                
                logger.info("=== SESSION CONTAINER INITIALIZED ===")
                
                # Note: Attenuation linking happens per-point during automatic attenuation workflow
                
            except Exception as e:
                logger.error(
                    f"Failed to add points to session container: {e}",
                    exc_info=True,
                )
        
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

        try:
            self._append_measurement_log(
                f"Point {self.current_measurement_sorted_index + 1}/{self.total_points}: move to ({self._x_mm:.3f}, {self._y_mm:.3f}) mm"
            )
        except Exception:
            pass

        # Move the stage using the controller
        import os
        import time

        self._timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._base_name = self.fileNameLineEdit.text().strip()
        txt_filename_base = os.path.join(
            self.measurement_folder,
            f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}",
        )

        # If attenuation is enabled, run attenuation sequence first, then normal capture
        attenuation_enabled = getattr(self, "attenuationCheckBox", None)
        if attenuation_enabled and self.attenuationCheckBox.isChecked():
            self._start_attenuation_then_normal(txt_filename_base)
            return

        # Otherwise, move stage and run normal capture directly
        try:
            new_x, new_y = self.stage_controller.move_stage(
                self._x_mm, self._y_mm, move_timeout=15
            )
        except TimeoutError:
            QMessageBox.warning(
                self,
                "Stage Timeout",
                "Stage movement timed out. Please check the hardware and try again. That's SAD",
            )
            return
        except Exception as e:
            QMessageBox.warning(
                self,
                "Stage Error",
                f"Stage movement failed: {str(e)}",
            )
            return

        self._start_normal_capture(txt_filename_base)

    def _start_normal_capture(self, txt_filename_base: str):
        # Launch the dual-capture worker in its own thread (normal mode)
        if not _get_zone_technical_imports():
            logger.error("Cannot start normal capture - technical imports not available")
            try:
                self._append_measurement_log("ERROR: Technical imports not available")
            except Exception:
                pass
            return
            
        try:
            self._append_measurement_log("Normal: capture")
        except Exception:
            pass
        
        # Get container version from config
        container_version = self.config.get('container_version', '0.1') if hasattr(self, 'config') else '0.1'
        
        CaptureWorker = _get_zone_technical_module('CaptureWorker')
        self.capture_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integration_time,
            txt_filename_base=txt_filename_base,
            frames=1,
            naming_mode="normal",
            container_version=container_version,
        )
        self.capture_thread = QThread()
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        self.capture_thread.start()

    def _get_loading_position(self):
        try:
            att = self.config.get("attenuation", {})
            pos = att.get("loading_position")
            if pos and isinstance(pos, dict):
                return float(pos.get("x")), float(pos.get("y"))
        except Exception:
            pass
        # Fallback to controller-provided load position if available
        try:
            positions = self.stage_controller.get_home_load_positions()
            return positions.get("load", (None, None))
        except Exception:
            return (None, None)

    def _capture_attenuation_background(self):
        """Capture attenuation WITHOUT sample once at loading position for this run."""
        frames = (
            int(getattr(self, "attenFramesSpin", None).value())
            if hasattr(self, "attenFramesSpin")
            else 100
        )
        short_t = (
            float(getattr(self, "attenTimeSpin", None).value())
            if hasattr(self, "attenTimeSpin")
            else 0.00005
        )

        load_x, load_y = self._get_loading_position()
        if load_x is None or load_y is None:
            logger.warning(
                "Loading position not configured; skipping attenuation background capture"
            )
            self._attenuation_bg_files = None
            return

        try:
            self.stage_controller.move_stage(load_x, load_y, move_timeout=20)
        except Exception as e:
            logger.warning(
                "Failed to move to loading position; skipping attenuation background capture",
                error=str(e),
            )
            self._attenuation_bg_files = None
            return

        import os

        try:
            self._append_measurement_log("Attenuation: move to loading position")
            self._append_measurement_log(
                f"Attenuation: capture WITHOUT sample (frames={frames}, t={short_t:.6f}s)"
            )
        except Exception:
            pass

        group_ts = time.strftime("%Y%m%d_%H%M%S")
        base_name = self.fileNameLineEdit.text().strip()
        group_base = os.path.join(self.measurement_folder, f"{base_name}_{group_ts}")

        # Get container version from config
        container_version = self.config.get('container_version', '0.1') if hasattr(self, 'config') else '0.1'
        
        results = {}
        for alias, controller in self.detector_controller.items():
            try:
                per_alias_base = f"{group_base}__{alias}_ATTENUATION0"
                ok = controller.capture_point(
                    Nframes=frames, Nseconds=short_t, filename_base=per_alias_base
                )
                txt_path = per_alias_base + ".txt" if ok else None
                if txt_path and os.path.exists(txt_path):
                    # Detector converts raw file to container format
                    npy_path = controller.convert_to_container_format(
                        txt_path, container_version
                    )
                    results[alias] = npy_path
                else:
                    results[alias] = None
            except Exception as e:
                logger.warning(
                    "Error capturing attenuation background",
                    detector=alias,
                    error=str(e),
                )
                results[alias] = None

        self._attenuation_bg_files = results
        try:
            n_ok = sum(1 for v in results.values() if v)
            self._append_measurement_log(
                f"Attenuation: background saved for {n_ok} detector(s)"
            )
        except Exception:
            pass
        
        # Add I₀ (without sample) to session container
        if hasattr(self, 'session_manager') and self.session_manager.is_session_active():
            try:
                all_data = {}
                for alias, npy_file in results.items():
                    if npy_file:
                        import numpy as np
                        all_data[alias] = np.load(npy_file)
                
                if all_data:
                    metadata = {
                        "n_frames": frames,
                        "integration_time_s": short_t,
                        "timestamp": group_ts,
                        "loading_position_mm": [load_x, load_y],
                    }
                    
                    # Get PONI map
                    pony_map = {}
                    for alias in all_data.keys():
                        if hasattr(self, 'get_poni_file'):
                            poni_file = self.get_poni_file(alias)
                            if poni_file:
                                pony_map[alias] = poni_file
                    
                    self.session_manager.add_attenuation_measurement(
                        data=all_data,
                        metadata=metadata,
                        pony_map=pony_map,
                        mode="without",
                    )
                    
                    logger.info(
                        "Added I₀ (without sample) to session container",
                        detectors=list(all_data.keys()),
                    )
            except Exception as e:
                logger.error(
                    f"Failed to add I₀ to session container: {e}",
                    exc_info=True,
                )

    def _record_attenuation_files(self, key: str, files: dict):
        """Record attenuation files in the measurement state under current point unique_id.
        key: "without_sample" | "with_sample"
        files: dict alias->filepath
        """
        try:
            mp = self.state_measurements.get("measurement_points", [])
            idx = self.current_measurement_sorted_index
            if 0 <= idx < len(mp):
                uid = mp[idx].get("unique_id")
            else:
                uid = None
        except Exception:
            uid = None
        if uid is None:
            return
        try:
            att = self.state_measurements.setdefault("attenuation_files", {})
            entry = att.setdefault(uid, {})
            entry[key] = files or {}
            # Persist to state file if available
            if (
                hasattr(self, "state_path_measurements")
                and self.state_path_measurements
            ):
                import json

                with open(self.state_path_measurements, "w") as f:
                    json.dump(self.state_measurements, f, indent=4)
        except Exception as e:
            print(f"Warning: failed to record attenuation files: {e}")

    def _start_attenuation_then_normal(self, txt_filename_base: str):
        # Read attenuation params from UI
        frames = (
            int(getattr(self, "attenFramesSpin", None).value())
            if hasattr(self, "attenFramesSpin")
            else 100
        )
        short_t = (
            float(getattr(self, "attenTimeSpin", None).value())
            if hasattr(self, "attenTimeSpin")
            else 0.00005
        )

        # Duplicate WITHOUT sample mapping from background (if available)
        if getattr(self, "_attenuation_bg_files", None):
            try:
                self._record_attenuation_files(
                    "without_sample", self._attenuation_bg_files
                )
            except Exception:
                pass
        else:
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Attenuation Background Missing",
                "Background attenuation (without sample) was not captured; proceeding with with-sample and normal measurements.",
            )

        # Move to point and capture WITH sample
        try:
            self.stage_controller.move_stage(self._x_mm, self._y_mm, move_timeout=15)
        except Exception:
            pass

        # Start attenuation capture (with sample) in a thread
        if not _get_zone_technical_imports():
            logger.error("Cannot start attenuation capture - technical imports not available")
            try:
                self._append_measurement_log("ERROR: Technical imports not available")
            except Exception:
                pass
            return
            
        try:
            self._append_measurement_log(
                f"Attenuation: capture WITH sample (frames={frames}, t={short_t:.6f}s)"
            )
        except Exception:
            pass
        
        # Get container version from config
        container_version = self.config.get('container_version', '0.1') if hasattr(self, 'config') else '0.1'
        
        CaptureWorker = _get_zone_technical_module('CaptureWorker')
        self._attn2_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=short_t,
            txt_filename_base=txt_filename_base,
            frames=frames,
            naming_mode="attenuation_with",
            container_version=container_version,
        )
        self._attn2_thread = QThread()
        self._attn2_worker.moveToThread(self._attn2_thread)
        self._attn2_thread.started.connect(self._attn2_worker.run)

        def _after_attn_with(success2, result_files2):
            # Files are already converted by detector - just record paths
            try:
                self._append_measurement_log("Attenuation: with-sample files saved")
            except Exception:
                pass
            
            # result_files2 already contains .npy paths from detector conversion
            moved_map = result_files2 or {}
            
            try:
                self._record_attenuation_files("with_sample", moved_map)
            except Exception:
                pass
            
            # Add I (with sample) to session container
            if hasattr(self, 'session_manager') and self.session_manager.is_session_active():
                try:
                    all_data = {}
                    for alias, npy_file in moved_map.items():
                        if npy_file:
                            import numpy as np
                            all_data[alias] = np.load(npy_file)
                    
                    if all_data:
                        metadata = {
                            "n_frames": frames,
                            "integration_time_s": short_t,
                            "timestamp": self._timestamp,
                            "point_position_mm": [self._x_mm, self._y_mm],
                        }
                        
                        # Get PONI map
                        pony_map = {}
                        for alias in all_data.keys():
                            if hasattr(self, 'get_poni_file'):
                                poni_file = self.get_poni_file(alias)
                                if poni_file:
                                    pony_map[alias] = poni_file
                        
                        self.session_manager.add_attenuation_measurement(
                            data=all_data,
                            metadata=metadata,
                            pony_map=pony_map,
                            mode="with",
                        )
                        
                        # Now link this point to the attenuation measurements
                        # (Link I₀ and I to current point)
                        try:
                            self.session_manager.link_attenuation_to_points(
                                num_points=1,
                                start_point_idx=self.current_measurement_sorted_index,
                            )
                            logger.info(
                                f"Linked attenuation to point {self.current_measurement_sorted_index}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to link attenuation to point: {e}",
                                exc_info=True,
                            )
                        
                        logger.info(
                            f"Added I (with sample) to session container at point {self.current_measurement_sorted_index}",
                            detectors=list(all_data.keys()),
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to add I to session container: {e}",
                        exc_info=True,
                    )
            
            # Proceed with normal capture
            self._start_normal_capture(txt_filename_base)

        self._attn2_worker.finished.connect(_after_attn_with)
        self._attn2_worker.finished.connect(self._attn2_thread.quit)
        self._attn2_worker.finished.connect(self._attn2_worker.deleteLater)
        self._attn2_thread.finished.connect(self._attn2_thread.deleteLater)
        self._attn2_thread.start()

    def on_capture_finished(self, success: bool, result_files: dict):
        """
        Callback after detector(s) finish capturing.
        Handles errors, triggers post-processing, colors UI.
        Adds detector meta to measurements_meta for each measurement file.
        """
        from pathlib import Path
        import numpy as np
        
        print(f"\n\n>>> CRITICAL DEBUG: on_capture_finished ENTERED, success={success}\n\n")
        logger.info(f">>> on_capture_finished called: success={success}, files={list(result_files.keys()) if result_files else None}")
        
        if not success:
            logger.error("Measurement capture failed")
            try:
                self._append_measurement_log("Normal: capture failed")
            except Exception:
                pass
            return
        
        logger.info("Measurement capture successful", files=list(result_files.keys()))
        print(">>> CHECKPOINT 1: Before measurement log")
        try:
            self._append_measurement_log("Normal: capture finished")
        except Exception as e:
            print(f">>> ERROR in _append_measurement_log: {e}")
        
        print(">>> CHECKPOINT 2: After measurement log")
        
        try:
            self._append_measurement_log("[DEBUG] Post-processing started")
        except Exception:
            pass
        
        print(">>> CHECKPOINT 3: Starting post-processing")
        logger.info("Starting post-capture processing...")

        # Build detector meta as before
        print(">>> CHECKPOINT 4: Building detector lookup")
        detector_lookup = {d["alias"]: d for d in self.config["detectors"]}

        measurements = self.state_measurements.get("measurements_meta", {})
        measurement_points = self.state_measurements["measurement_points"]
        current_index = self.current_measurement_sorted_index
        x = self._x_mm
        y = self._y_mm
        point_unique_id = measurement_points[current_index]["unique_id"]

        for alias, npy_filename in result_files.items():
            detector_meta = detector_lookup.get(alias, {})
            entry = {
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
            # Attach calibration group hash if available
            gh = getattr(self, "calibration_group_hash", None)
            if gh:
                entry["CALIBRATION_GROUP_HASH"] = gh
            # result_files now contains .npy files (converted by detector)
            measurements[Path(npy_filename).name] = entry

        self.state_measurements["measurements_meta"] = measurements

        try:
            self._append_measurement_log("[DEBUG] Saving state file")
        except Exception:
            pass
        
        # Save updated state
        with open(self.state_path_measurements, "w") as f:
            json.dump(self.state_measurements, f, indent=4)
        
        try:
            self._append_measurement_log("[DEBUG] State saved")
        except Exception:
            pass
        
        # Add to session container if session is active
        logger.info(f"Checking session manager: has_attr={hasattr(self, 'session_manager')}, active={self.session_manager.is_session_active() if hasattr(self, 'session_manager') else False}")
        
        try:
            self._append_measurement_log("[DEBUG] Checking H5 session")
        except Exception:
            pass
        
        if hasattr(self, 'session_manager') and self.session_manager.is_session_active():
            try:
                self._append_measurement_log("[DEBUG] Writing to H5")
            except Exception:
                pass
            try:
                logger.info(f"=== ADDING MEASUREMENT TO H5 (Point {current_index + 1}) ===")
                logger.info(f"Session path: {self.session_manager.session_path}")
                # Files are already .npy (converted by detector)
                
                all_data = {}
                raw_files_data = {}
                
                detector_lookup = {d["alias"]: d for d in self.config["detectors"]}
                pony_alias_map = {}
                for alias, npy_file in result_files.items():
                    detector_meta = detector_lookup.get(alias, {})
                    detector_id = detector_meta.get("id", alias)
                    pony_alias_map[alias] = detector_id
                    # Load data directly
                    logger.info(f"Loading {alias} data from: {Path(npy_file).name}")
                    all_data[detector_id] = np.load(npy_file)
                    logger.info(f"  Data shape: {all_data[detector_id].shape}")
                    
                    # Find and read raw files for blob storage
                    npy_path = Path(npy_file)
                    base_name = npy_path.stem  # filename without .npy
                    folder = npy_path.parent
                    
                    # Get raw file patterns from detector
                    detector_controller = self.detector_controller.get(alias)
                    if detector_controller and hasattr(detector_controller, 'get_raw_file_patterns'):
                        patterns = detector_controller.get_raw_file_patterns()
                    else:
                        # Fallback patterns if detector doesn't specify
                        patterns = ['*.txt', '*.dsc', '*.t3pa']
                        logger.warning(f"Detector {alias} has no get_raw_file_patterns(), using default patterns")
                    
                    # Collect raw files based on detector patterns
                    raw_files = {}
                    for pattern in patterns:
                        ext = pattern[1:] if pattern.startswith('*') else pattern
                        raw_file = folder / f"{base_name}{ext}"
                        if raw_file.exists():
                            try:
                                with open(raw_file, 'rb') as f:
                                    # Store with key format: raw_<ext> (e.g., raw_txt, raw_dsc)
                                    # This matches technical container blob naming convention
                                    file_format = ext[1:] if ext.startswith('.') else ext
                                    blob_key = f"raw_{file_format}"
                                    raw_files[blob_key] = f.read()
                                logger.debug(f"Read raw file for blob: {raw_file.name} -> {blob_key}")
                            except Exception as e:
                                logger.warning(f"Failed to read raw file {raw_file}: {e}")
                    
                    if raw_files:
                        raw_files_data[detector_id] = raw_files
                        logger.info(f"  Found {len(raw_files)} raw files for {alias}: {list(raw_files.keys())}")
                    else:
                        logger.warning(f"  No raw files found for {alias} using patterns {patterns}")
                
                logger.info(f"Loaded data from {len(all_data)} detectors")
                
                # Build detector metadata (per-detector)
                detector_metadata = {}
                for detector_id in all_data.keys():
                    detector_metadata[detector_id] = {
                        "integration_time_ms": self.integration_time * 1000,  # Convert to ms
                        "detector_id": detector_id,
                        "x_mm": x,
                        "y_mm": y,
                        "timestamp": self._timestamp,
                        "unique_id": point_unique_id,
                    }
                
                raw_files_by_detector_id = raw_files_data
                
                # Add measurement to session container with raw file blobs
                # current_index is 0-based, but session container uses 1-based indices
                point_index_1based = current_index + 1
                logger.info(f"Writing to H5: /measurements/pt_{point_index_1based:03d}/meas_NNNNNNNNN")
                logger.info(f"  Detectors: {list(all_data.keys())}")
                logger.info(f"  Raw files: {len(raw_files_by_detector_id)} detector(s) with blobs")
                
                self.session_manager.add_measurement(
                    point_index=point_index_1based,
                    measurement_data=all_data,
                    detector_metadata=detector_metadata,
                    pony_alias_map=pony_alias_map,
                    raw_files=raw_files_by_detector_id if raw_files_by_detector_id else None,
                )
                
                logger.info(f"✓ Measurement added to H5 container for point {point_index_1based}")
                
                try:
                    self._append_measurement_log("[DEBUG] H5 write complete")
                except Exception:
                    pass
                
            except Exception as e:
                logger.error("="*60)
                logger.error(f"✗ CRITICAL ERROR: Failed to add measurement to H5")
                logger.error("="*60)
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {e}")
                logger.error(f"Point index: {current_index + 1}")
                logger.error(f"Detectors: {list(result_files.keys())}")
                logger.error(f"Session path: {self.session_manager.session_path if hasattr(self, 'session_manager') else 'N/A'}")
                logger.error("="*60, exc_info=True)
                # Don't return - let the workflow continue even if H5 write fails
                logger.warning("Continuing measurement workflow despite H5 write failure...")
        else:
            logger.warning("⚠ Session manager not active - measurements will NOT be saved to H5!")
            try:
                self._append_measurement_log("[DEBUG] No H5 session active")
            except Exception:
                pass

        # === The rest is unchanged (your logic) ===
        try:
            self._append_measurement_log("[DEBUG] Spawning worker thread")
        except Exception:
            pass
        
        logger.info("Spawning measurement thread for post-processing...")
        current_row = self.sorted_indices[self.current_measurement_sorted_index]
        self.spawn_measurement_thread(current_row, result_files)

        # Visual feedback
        try:
            self._append_measurement_log("[DEBUG] Updating UI colors")
        except Exception:
            pass
        
        logger.info("Updating UI visual feedback...")
        green_brush = QColor(0, 255, 0)
        self._point_item.setBrush(green_brush)
        try:
            if self._zone_item:
                green_zone = QColor(0, 255, 0)
                green_zone.setAlphaF(0.2)
                self._zone_item.setBrush(green_zone)
        except Exception as e:
            logger.warning("Error updating zone item color", error=str(e))
        
        try:
            self._append_measurement_log("[DEBUG] Scheduling next point")
        except Exception:
            pass
        
        print(">>> CHECKPOINT FINAL: About to schedule QTimer")
        logger.info("Scheduling measurement_finished in 1000ms...")
        QTimer.singleShot(1000, self.measurement_finished)
        print(">>> CHECKPOINT EXIT: on_capture_finished complete")
        logger.info("<<< on_capture_finished complete")

    def spawn_measurement_thread(self, row, file_map):
        """
        Spawns a MeasurementWorker in a new thread for post-processing measurement files.
        Connects signals for result handling and thread cleanup.
        """
        if not _get_zone_technical_imports():
            logger.error("Cannot spawn measurement thread - technical imports not available")
            return
            
        thread = QThread(self)
        MeasurementWorker = _get_zone_technical_module('MeasurementWorker')
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
        logger.info(f">>> measurement_finished called (point {self.current_measurement_sorted_index + 1}/{self.total_points})")
        
        if self.stopped:
            logger.debug("Measurement stopped in measurement_finished")
            return

        logger.info("Advancing to next point...")
        self.current_measurement_sorted_index += 1
        self.progressBar.setValue(self.current_measurement_sorted_index)
        logger.info(f"Progress: {self.current_measurement_sorted_index}/{self.total_points}")
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
            logger.info(f"Moving to next point ({self.current_measurement_sorted_index + 1}/{self.total_points})")
            self.measure_next_point()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                logger.info("=== ALL MEASUREMENT POINTS COMPLETED ===")
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                self.start_btn.setEnabled(True)
            else:
                logger.warning(f"Measurement stopped: paused={self.paused}, stopped={self.stopped}")
        
        logger.info("<<< measurement_finished complete")

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
    ) -> Optional["MeasurementHistoryWidget"]:
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
        if not _get_zone_technical_imports():
            logger.error("Cannot create measurement widget - technical imports not available")
            return None
            
        MeasurementHistoryWidget = _get_zone_technical_module('MeasurementHistoryWidget')
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

    def _confirm_poni_settings_before_measurement(self):
        """Show PONI settings confirmation dialog before starting measurements.
        Returns True if user wants to proceed, False if they want to update PONI settings first.
        """
        from PyQt5.QtWidgets import QMessageBox

        try:
            active_aliases = self.hardware_controller.active_detector_aliases
        except Exception:
            dev_mode = self.config.get("DEV", False)
            ids = (
                self.config.get("dev_active_detectors", [])
                if dev_mode
                else self.config.get("active_detectors", [])
            )
            active_aliases = [
                d.get("alias")
                for d in self.config.get("detectors", [])
                if d.get("id") in ids
            ]

        ponis = getattr(self, "ponis", {}) or {}
        poni_files = getattr(self, "poni_files", {}) or {}

        # Check for missing PONI calibrations
        missing = [a for a in active_aliases if not ponis.get(a)]
        if missing:
            QMessageBox.warning(
                self,
                "Missing PONI Calibration",
                "PONI calibration must be set for detectors: "
                + ", ".join(missing)
                + "\nOpen the detector tabs and set PONI files before starting measurements.",
            )
            return False

        # Build PONI status summary
        # If there are no active detectors, skip dialog and proceed
        if not active_aliases:
            return True

        poni_status = []
        for alias in active_aliases:
            meta = poni_files.get(alias, {})
            path = meta.get("path")
            name = meta.get("name") or "Default/Embedded PONI"

            if path:
                from pathlib import Path

                if Path(path).exists():
                    status = "✓ File exists"
                else:
                    status = "⚠ File missing"
                poni_status.append(f"• {alias}: {name}\n  {status}: {path}")
            else:
                poni_status.append(f"• {alias}: {name}\n  ✓ Using embedded data")

        status_text = "\n\n".join(poni_status)

        # Show confirmation dialog
        try:
            parent = (
                self
                if hasattr(self, "isWidgetType")
                and callable(getattr(self, "isWidgetType"))
                and self.isWidgetType()
                else None
            )
        except Exception:
            parent = None
        msg = QMessageBox(parent)
        msg.setWindowTitle("Confirm PONI Settings")
        msg.setIcon(QMessageBox.Question)
        msg.setText(
            "Current PONI calibration settings:\n\n"
            f"{status_text}\n\n"
            "Do you want to start measurements with these settings?"
        )

        # Add custom buttons
        start_button = msg.addButton("Start Measurements", QMessageBox.AcceptRole)
        update_button = msg.addButton("Update PONI Settings", QMessageBox.RejectRole)
        cancel_button = msg.addButton("Cancel", QMessageBox.RejectRole)

        msg.setDefaultButton(start_button)
        msg.exec_()

        clicked = msg.clickedButton()
        if clicked == start_button:
            return True  # Proceed with measurements
        elif clicked == update_button:
            # Switch to first detector tab to allow user to update PONI settings
            if hasattr(self, "tabs") and hasattr(self, "detector_tabs"):
                # Find first detector tab and switch to it
                first_detector_tab = None
                min_index = float("inf")
                for alias, tab_info in self.detector_tabs.items():
                    if tab_info["index"] < min_index:
                        min_index = tab_info["index"]
                        first_detector_tab = tab_info["index"]
                if first_detector_tab is not None:
                    self.tabs.setCurrentIndex(first_detector_tab)
            return False  # Don't start measurements
        else:
            return False  # Cancel
