import hashlib
import json
import os
import shutil
import string
from pathlib import Path

from PyQt5.QtCore import QRectF, QTimer
from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

from hardware.Ulster.gui.image_view_ext.point_editing_extension import null_dict


class StateSaverMixin:
    @staticmethod
    def _get_autosave_drive():
        drives = [
            f"{d}:/"
            for d in string.ascii_uppercase
            if os.path.exists(f"{d}:/") and d.lower() not in ("a", "b")
        ]
        return (
            Path(
                drives[
                    (1 if len(drives) > 1 and drives[0].lower().startswith("c") else 0)
                ]
            )
            if drives
            else Path.cwd()
        )

    _AUTOSAVE_DRIVE = _get_autosave_drive.__func__()
    AUTO_STATE_FILE = _AUTOSAVE_DRIVE / "autosave_state.json"
    PREV_STATE_FILE = _AUTOSAVE_DRIVE / "autosave_state_prev.json"

    # ---- Core Save/Restore API ----
    def restore_state(self, file_path=None):
        # Signal to UI code that we are restoring and should not create measurement widgets
        self._restoring_state = True
        self.measurement_widgets = {}
        state = self._load_state(file_path)
        self.state = state

        # Remove Measurement Points when restoring state and clear related UI
        if state:
            # Drop measurement-related entries so they are not carried over
            state.pop("measurement_points", None)
            state.pop("skipped_points", None)
            try:
                # Clear right-panel measurement widgets/tree if present
                if (
                    hasattr(self, "measurementsTree")
                    and self.measurementsTree is not None
                ):
                    self.measurementsTree.clear()
                if hasattr(self, "_measurement_items") and isinstance(
                    self._measurement_items, dict
                ):
                    # Remove any existing widgets safely
                    if hasattr(self, "remove_measurement_widget_from_panel"):
                        for pid in list(self._measurement_items.keys()):
                            try:
                                self.remove_measurement_widget_from_panel(pid)
                            except Exception:
                                pass
                    self._measurement_items.clear()
                # Reset mapping container
                if hasattr(self, "measurement_widgets") and isinstance(
                    self.measurement_widgets, dict
                ):
                    self.measurement_widgets.clear()
            except Exception as e:
                print(
                    f"Warning: failed to clear measurement widgets during restore: {e}"
                )

        if not state:
            self._restoring_state = False
            return

        # Handle PONI file restoration with user confirmation
        self._handle_poni_restoration(state)

        self._restore_image(state.get("image"))
        self._restore_rotation(state.get("rotation_angle", 0))
        self._restore_crop_rect(state.get("crop_rect"))
        self._restore_shapes(state.get("shapes", []))
        self._restore_points(state.get("zone_points", []))
        self._refresh_id_counter()
        # Update UI while suppressing widget creation in the points table
        try:
            if hasattr(self, "update_points_table"):
                print(
                    "Attempting to update points table after restore (no measurement widgets)..."
                )
                self.update_points_table()
            else:
                print("update_points_table method not available, skipping")
        except Exception as e:
            print(f"Skipping points table update due to error: {e}")

        try:
            if hasattr(self, "update_shape_table"):
                self.update_shape_table()
        except Exception as e:
            print(f"Error updating shape table: {e}")

        try:
            if hasattr(self, "update_coordinates"):
                self.update_coordinates()
        except Exception as e:
            print(f"Error updating coordinates: {e}")
        finally:
            # Re-enable widget creation for subsequent operations
            self._restoring_state = False

    def auto_save_state(self):
        self._save_state(self.AUTO_STATE_FILE, True)

    def manual_save_state(self):
        self._save_state(self.PREV_STATE_FILE, False)

    def setup_auto_save(self, interval=2000):
        self.autoSaveTimer = QTimer(self)
        self.autoSaveTimer.timeout.connect(self.auto_save_state)
        self.autoSaveTimer.start(interval)

    # ---- Internal Helpers ----
    def _load_state(self, file_path):
        for path in [file_path, self.PREV_STATE_FILE, self.AUTO_STATE_FILE]:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    try:
                        return json.load(f)
                    except Exception as e:
                        print("Error loading state:", e)
        print("No saved state file found. Nothing to restore.")
        return None

    def _save_state(self, target_file, is_auto):
        state = {
            "measurement_points": self.generate_measurement_points(),
            "image": getattr(self.image_view, "current_image_path", None),
            "rotation_angle": getattr(self.image_view, "rotation_angle", 0),
            "crop_rect": self._get_crop_rect(),
            "shapes": self._get_shapes(),
            "zone_points": self._get_zone_points(),
        }
        if not is_auto:
            rx = getattr(self, "real_x_pos_mm", None)
            ry = getattr(self, "real_y_pos_mm", None)
            state["real_center"] = (
                rx.value() if rx is not None else None,
                ry.value() if ry is not None else None,
            )
            state["pixel_to_mm_ratio"] = getattr(self, "pixel_to_mm_ratio", 1)

        # --- Include active detectors and their PONI meta in the state ---
        try:
            active_aliases = self.hardware_controller.active_detector_aliases
        except Exception:
            # Fallback based on config
            dev_mode = self.config.get("DEV", False)
            active_ids = (
                self.config.get("dev_active_detectors", [])
                if dev_mode
                else self.config.get("active_detectors", [])
            )
            aliases = []
            for det_cfg in self.config.get("detectors", []):
                if det_cfg.get("id") in active_ids:
                    aliases.append(det_cfg.get("alias"))
            active_aliases = aliases

        state["active_detectors_aliases"] = active_aliases

        ponis = getattr(self, "ponis", {}) or {}
        # Try to get current settings from UI if available
        try:
            current_settings = self.get_current_poni_settings()
        except Exception:
            current_settings = {}

        det_info = {}
        for alias in active_aliases:
            # Prefer current UI settings, fallback to stored settings
            if alias in current_settings:
                settings = current_settings[alias]
                det_info[alias] = {
                    "poni_filename": settings.get("name"),
                    "poni_path": settings.get("path"),
                    "poni_value": settings.get("value", ""),
                }
            else:
                # Fallback to stored poni_files data
                poni_files = getattr(self, "poni_files", {}) or {}
                meta = poni_files.get(alias, {})
                pth = meta.get("path")
                det_info[alias] = {
                    "poni_filename": meta.get("name"),
                    "poni_path": str(pth) if pth is not None else None,
                    "poni_value": ponis.get(alias, ""),
                }
        state["detector_poni"] = det_info

        self.state = state
        if is_auto and os.path.exists(self.AUTO_STATE_FILE):
            try:
                shutil.copyfile(self.AUTO_STATE_FILE, self.PREV_STATE_FILE)
            except Exception as e:
                print("Error copying autosave file:", e)
        try:
            with open(target_file, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print("Error saving state:", e)

    def _get_crop_rect(self):
        r = getattr(self.image_view, "crop_rect", None)
        return (
            {"x": r.x(), "y": r.y(), "width": r.width(), "height": r.height()}
            if r
            else None
        )

    def _get_shapes(self):
        result = []
        for s in getattr(self.image_view, "shapes", []):
            item = s.get("item")
            if item:
                rect = item.sceneBoundingRect()
                result.append(
                    {
                        "id": s.get("id"),
                        "type": s.get("type"),
                        "role": s.get("role", "include"),
                        "geometry": {
                            "x": rect.x(),
                            "y": rect.y(),
                            "width": rect.width(),
                            "height": rect.height(),
                        },
                    }
                )
        return result

    def _get_zone_points(self):
        out = []
        for t in ("generated", "user"):
            for pt in self.image_view.points_dict[t]["points"]:
                center = pt.sceneBoundingRect().center()
                out.append(
                    {
                        "x": center.x(),
                        "y": center.y(),
                        "type": t,
                        "id": pt.data(1),
                    }
                )
        return out

    def _restore_image(self, image_path):
        if image_path:
            self.image_view.set_image(QPixmap(image_path), image_path=image_path)
        else:
            print("No image to restore.")

    def _restore_rotation(self, angle):
        self.image_view.rotation_angle = angle
        if getattr(self.image_view, "image_item", None):
            self.image_view.image_item.setRotation(angle)

    def _restore_crop_rect(self, rect):
        if rect:
            self.image_view.crop_rect = QRectF(
                rect["x"], rect["y"], rect["width"], rect["height"]
            )
        else:
            self.image_view.crop_rect = None

    def _restore_shapes(self, shapes):
        self.image_view.shapes = []
        for i, shape in enumerate(shapes):
            shape_id = shape.get("id", 0) + 100 + i
            s_type, role, geo = (
                shape.get("type"),
                shape.get("role", "include"),
                shape.get("geometry"),
            )
            x, y, w, h = (
                geo.get("x"),
                geo.get("y"),
                geo.get("width"),
                geo.get("height"),
            )
            item = (
                QGraphicsEllipseItem(x, y, w, h)
                if s_type.lower() in ["ellipse", "circle"]
                else QGraphicsRectItem(x, y, w, h)
            )
            pen = (
                QPen(
                    QColor(
                        "green"
                        if role == "include"
                        else "red" if role == "exclude" else "blue"
                    ),
                    3,
                )
                if role in ["include", "exclude", "sample holder"]
                else QPen(QColor("black"), 1)
            )
            item.setPen(pen)
            self.image_view.scene.addItem(item)
            self.image_view.shapes.append(
                {
                    "id": shape_id,
                    "type": s_type,
                    "role": role,
                    "item": item,
                    "active": (
                        True
                        if role in ["include", "exclude", "sample holder"]
                        else False
                    ),
                }
            )

    # --- state_saver_extension.py ---

    def _restore_points(self, points):
        import copy

        self.image_view.points_dict = copy.deepcopy(null_dict)

        # Initialize next_point_id from incoming points
        # If there are no ids, start at 1
        existing_ids = [pt.get("id") for pt in points if pt.get("id") is not None]
        try:
            self.next_point_id = (
                max(int(x) for x in existing_ids) + 1 if existing_ids else 1
            )
        except Exception:
            self.next_point_id = 1

        # Import ZonePointsRenderer if available (new system)
        try:
            from .points.zone_points_renderer import ZonePointsRenderer

            use_new_system = True
            print("Using new ZonePointsRenderer system for point restoration")
        except ImportError:
            use_new_system = False
            print(
                "Using legacy system for point restoration (ZonePointsRenderer not available)"
            )

        for pt in points:
            x, y, pt_type, pt_id = pt["x"], pt["y"], pt["type"], pt.get("id")

            # assign an id if missing
            if pt_id is None:
                pt_id = self.next_point_id
                self.next_point_id += 1

            if use_new_system:
                # Use new ZonePointsRenderer system
                # Create point using the new renderer
                if pt_type == "user":
                    radius = 10  # Default radius for user points
                else:
                    radius = 5  # Default radius for generated points

                point_item = ZonePointsRenderer.create_point_item(x, y, pt_id, pt_type)
                zone_item = ZonePointsRenderer.create_zone_item(x, y, radius)

                self.image_view.scene.addItem(zone_item)
                self.image_view.scene.addItem(point_item)

                self.image_view.points_dict[pt_type]["points"].append(point_item)
                self.image_view.points_dict[pt_type]["zones"].append(zone_item)
            else:
                # Fallback to old system
                if pt_type == "user":
                    rad = 10
                    marker = QGraphicsEllipseItem(-rad, -rad, 2 * rad, 2 * rad)
                    marker.setBrush(QColor("blue"))
                    marker.setPen(QPen())
                    marker.setFlags(
                        QGraphicsEllipseItem.ItemIsSelectable
                        | QGraphicsEllipseItem.ItemIsMovable
                    )
                    marker.setData(0, "user")
                    marker.setData(1, pt_id)  # <-- always set id
                    marker.setPos(x, y)
                    self.image_view.scene.addItem(marker)
                    self.image_view.points_dict["user"]["points"].append(marker)

                    zone = QGraphicsEllipseItem(x - rad, y - rad, 2 * rad, 2 * rad)
                    zc = QColor("blue")
                    zc.setAlphaF(0.2)
                    zone.setBrush(zc)
                    zone.setPen(QPen())
                    self.image_view.scene.addItem(zone)
                    self.image_view.points_dict["user"]["zones"].append(zone)
                else:
                    marker = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
                    marker.setBrush(QColor("red"))
                    marker.setPen(QPen())
                    marker.setFlags(
                        QGraphicsEllipseItem.ItemIsSelectable
                        | QGraphicsEllipseItem.ItemIsMovable
                    )
                    marker.setData(0, "generated")
                    marker.setData(1, pt_id)  # <-- always set id
                    self.image_view.scene.addItem(marker)
                    self.image_view.points_dict["generated"]["points"].append(marker)

                    zone = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
                    zc = QColor("cyan")
                    zc.setAlphaF(0.2)
                    zone.setBrush(zc)
                    zone.setPen(QPen())
                    self.image_view.scene.addItem(zone)
                    self.image_view.points_dict["generated"]["zones"].append(zone)

        self.shapes = self.image_view.shapes
        self.generated_points = self.image_view.points_dict["generated"]["points"]
        self.user_defined_points = self.image_view.points_dict["user"]["points"]
        self.measurement_widgets = {}

    def _refresh_id_counter(self):
        all_ids = [
            pt.data(1)
            for pt in (
                self.image_view.points_dict["generated"]["points"]
                + self.image_view.points_dict["user"]["points"]
            )
            if pt.data(1) is not None
        ]
        try:
            self.next_point_id = max(int(i) for i in all_ids) + 1 if all_ids else 1
        except Exception:
            self.next_point_id = 1

    # ---- As before ----
    def generate_measurement_points(self):
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
        measurement_points = []
        for idx, (pt_idx, x_mm, y_mm) in enumerate(all_points_sorted):
            id_str = f"{idx}:{pt_idx}:{x_mm:.6f}:{y_mm:.6f}"
            unique_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()[:16]
            measurement_points.append(
                {
                    "unique_id": unique_id,
                    "index": idx,
                    "point_index": pt_idx,
                    "x": x_mm,
                    "y": y_mm,
                }
            )
        return measurement_points

    def _handle_poni_restoration(self, state):
        """Handle PONI file restoration with user confirmation dialog."""
        detector_poni = state.get("detector_poni", {})
        if not detector_poni:
            return  # No PONI data in state

        from PyQt5.QtWidgets import QMessageBox

        # Create confirmation dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Restore PONI Files")
        msg.setIcon(QMessageBox.Question)

        # Build message showing what PONI files are in the state
        poni_info = []
        for alias, poni_data in detector_poni.items():
            filename = poni_data.get("poni_filename", "N/A")
            path = poni_data.get("poni_path", "N/A")
            poni_info.append(f"• {alias}: {filename}")
            if path and path != "N/A":
                poni_info.append(f"  Path: {path}")

        poni_list = "\n".join(poni_info)
        msg.setText(
            "The state file contains PONI calibration files.\n\n"
            "Do you want to restore PONI file paths from the state?\n\n"
            f"PONI files in state:\n{poni_list}"
        )

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        result = msg.exec_()

        if result == QMessageBox.Yes:
            # User wants to restore PONI files
            self._restore_poni_files_from_state(detector_poni)
        else:
            print("User chose to keep existing PONI file settings")

    def _restore_poni_files_from_state(self, detector_poni):
        """Restore PONI file paths from state, checking if files exist."""
        from pathlib import Path

        restored_count = 0
        missing_files = []

        for alias, poni_data in detector_poni.items():
            path = poni_data.get("poni_path")
            value = poni_data.get("poni_value", "")

            # Update the internal PONI value
            if not hasattr(self, "ponis"):
                self.ponis = {}
            self.ponis[alias] = value

            # Update the PONI file metadata
            if not hasattr(self, "poni_files"):
                self.poni_files = {}

            if path and Path(path).exists():
                # File exists - restore the path
                self.poni_files[alias] = {
                    "path": path,
                    "name": poni_data.get("poni_filename"),
                }
                # Update UI line edit if it exists
                poni_lineedit = getattr(self, f"{alias.lower()}_poni_lineedit", None)
                if poni_lineedit:
                    poni_lineedit.setText(path)
                restored_count += 1
                print(f"Restored PONI path for {alias}: {path}")
            else:
                # File doesn't exist - clear the path
                self.poni_files[alias] = {"path": None, "name": None}
                # Clear UI line edit if it exists
                poni_lineedit = getattr(self, f"{alias.lower()}_poni_lineedit", None)
                if poni_lineedit:
                    poni_lineedit.setText("")
                missing_files.append(f"{alias}: {path or 'N/A'}")
                print(f"PONI file missing for {alias}: {path}")

        # Show summary
        from PyQt5.QtWidgets import QMessageBox

        summary_msg = f"PONI file restoration complete:\n\n• {restored_count} files restored successfully"
        if missing_files:
            summary_msg += (
                f"\n• {len(missing_files)} files were missing and paths were cleared:\n"
            )
            for missing in missing_files:
                summary_msg += f"  - {missing}\n"

        QMessageBox.information(self, "PONI Restoration Complete", summary_msg)
