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
    def start_measurements(self):
        pm = _pm()

        self.manual_save_state()
        self.measurement_folder = Path(self.folderLineEdit.text().strip())
        self.state_path_measurements = self.measurement_folder / f"{self.fileNameLineEdit.text()}_state.json"

        if not self.measurement_folder.exists():
            pm.QMessageBox.warning(
                self,
                "Folder Error",
                "Selected folder does not exist. Please select the correct folder.",
            )
            return

        if not self._confirm_poni_settings_before_measurement():
            return

        try:
            from .preflight_dialog import PreflightDialog

            d = PreflightDialog(
                self,
                session_manager=getattr(self, "session_manager", None),
            )
            if d.exec_() != d.Accepted:
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
        try:
            self._append_measurement_log(f"Start: {self.total_points} points, T={self.integration_time:.2f}s")
        except Exception:
            pass

        try:
            limits = self.stage_controller.get_limits() if hasattr(self, "stage_controller") else None
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

        if not measurement_points:
            pm.logger.error("No valid measurement points within axis limits")
            pm.QMessageBox.warning(
                self,
                "No Valid Points",
                f"All measurement points exceed the axis limits of X[{x_min:.1f},{x_max:.1f}] and Y[{y_min:.1f},{y_max:.1f}] mm. "
                "Please adjust your measurement grid.",
            )
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
                self.session_manager.add_points(points_for_session)
                pm.logger.info(f"✓ Added {len(points_for_session)} points to session container")

                if hasattr(self, "_add_zones_to_session"):
                    pm.logger.info("Adding zones to session container...")
                    num_shapes = len(self.state.get("shapes", []))
                    pm.logger.info(f"Found {num_shapes} shapes in state")
                    self._add_zones_to_session()
                    pm.logger.info("✓ Zones processing complete")
                else:
                    pm.logger.warning("⚠ _add_zones_to_session method not found")

                if hasattr(self, "_add_mapping_to_session"):
                    pm.logger.info("Adding mapping to session container...")
                    self._add_mapping_to_session()
                    pm.logger.info("✓ Mapping added")
                else:
                    pm.logger.warning("⚠ _add_mapping_to_session method not found")

                pm.logger.info("=== SESSION CONTAINER INITIALIZED ===")
            except Exception as e:
                pm.logger.error(f"Failed to add points to session container: {e}", exc_info=True)

        self.measure_next_point()
