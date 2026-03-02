"""State saver restoration and reconstruction responsibilities."""

from . import state_saver_extension as _module

base64 = _module.base64
json = _module.json
os = _module.os
shutil = _module.shutil
string = _module.string
Path = _module.Path
unquote = _module.unquote
urlparse = _module.urlparse

QRectF = _module.QRectF
QTimer = _module.QTimer
QColor = _module.QColor
QPen = _module.QPen
QPixmap = _module.QPixmap
QGraphicsEllipseItem = _module.QGraphicsEllipseItem
QGraphicsRectItem = _module.QGraphicsRectItem

null_dict = _module.null_dict


class StateSaverRestoreMixin:
    def _restore_image(self, image_path, state_dir: Path = None):
        # Normalize/convert path-like inputs, handle file:// URIs
        try:
            if image_path:
                image_path = str(image_path)
                if image_path.lower().startswith("file:"):
                    parsed = urlparse(image_path)
                    local = unquote(parsed.path)
                    # On Windows, parsed.path can start with "/C:/..."
                    if (
                        os.name == "nt"
                        and local.startswith("/")
                        and len(local) > 3
                        and local[2] == ":"
                    ):
                        local = local.lstrip("/")
                    image_path = local
        except Exception:
            pass

        def try_load(candidate_path) -> bool:
            try:
                cp = str(candidate_path)
                if os.path.exists(cp):
                    pm = QPixmap(cp)
                    if not pm.isNull():
                        self.image_view.set_image(pm, image_path=cp)
                        return True
            except Exception:
                return False
            return False

        # 1) Absolute path or directly loadable
        if image_path:
            # If relative but exists in CWD, this will also succeed
            if try_load(image_path):
                return

        # 2) Try relative to state file directory
        try:
            if image_path and state_dir is None:
                state_dir = getattr(self, "_last_state_path", None)
                state_dir = state_dir.parent if state_dir else None
            if image_path and state_dir:
                candidate = (Path(state_dir) / image_path).resolve()
                if try_load(candidate):
                    return
        except Exception:
            pass

        # 3) Try folderLineEdit (user-selected root) if available
        try:
            if (
                image_path
                and hasattr(self, "folderLineEdit")
                and self.folderLineEdit
                and self.folderLineEdit.text()
            ):
                root = Path(self.folderLineEdit.text())
                candidate = (root / Path(image_path).name).resolve()
                if try_load(candidate):
                    return
        except Exception:
            pass

        # 4) Try config default_image_folder/default_folder
        try:
            default_folder = None
            if hasattr(self, "config") and isinstance(self.config, dict):
                default_folder = self.config.get(
                    "default_image_folder"
                ) or self.config.get("default_folder")
            if image_path and default_folder:
                candidate = (Path(default_folder) / Path(image_path).name).resolve()
                if try_load(candidate):
                    return
        except Exception:
            pass

        # 5) Try basename search in likely roots (state_dir, folderLineEdit, default_folder, cwd)
        base = None
        try:
            base = Path(image_path).name if image_path else None
        except Exception:
            base = None
        roots = []
        try:
            if state_dir:
                roots.append(Path(state_dir))
        except Exception:
            pass
        try:
            if (
                hasattr(self, "folderLineEdit")
                and self.folderLineEdit
                and self.folderLineEdit.text()
            ):
                roots.append(Path(self.folderLineEdit.text()))
        except Exception:
            pass
        try:
            default_folder = None
            if hasattr(self, "config") and isinstance(self.config, dict):
                default_folder = self.config.get(
                    "default_image_folder"
                ) or self.config.get("default_folder")
            if default_folder:
                roots.append(Path(default_folder))
        except Exception:
            pass
        roots.append(Path.cwd())
        if base:
            try:
                for root in roots:
                    try:
                        candidates = list(root.rglob(base))
                    except Exception:
                        candidates = []
                    if not candidates:
                        # Try same stem any image extension
                        stem = Path(base).stem
                        try:
                            candidates = [
                                p
                                for p in root.rglob(stem + ".*")
                                if p.suffix.lower()
                                in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                            ]
                        except Exception:
                            candidates = []
                    if len(candidates) == 1:
                        if try_load(candidates[0]):
                            return
                    elif len(candidates) > 1:
                        # Let user choose
                        from PyQt5.QtWidgets import QFileDialog

                        chosen, _ = QFileDialog.getOpenFileName(
                            self,
                            "Select image",
                            str(root),
                            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
                        )
                        if chosen and try_load(chosen):
                            return
            except Exception:
                pass

        # 6) Fallback to base64 embedded image if available
        try:
            b64 = None
            if isinstance(getattr(self, "state", None), dict):
                b64 = self.state.get("image_base64") or self.state.get("image_b64")
            if b64:
                data = base64.b64decode(b64)
                pm = QPixmap()
                if pm.loadFromData(data):
                    self.image_view.set_image(pm, image_path=None)
                    return
        except Exception:
            pass

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
        used_ids = set()
        used_uids = set()
        for i, shape in enumerate(shapes):
            raw_id = shape.get("id", None)
            try:
                shape_id = int(raw_id) if raw_id is not None else i + 1
            except Exception:
                shape_id = i + 1
            while shape_id in used_ids:
                shape_id += 1
            used_ids.add(shape_id)
            raw_uid = str(shape.get("uid") or "").strip()
            if not raw_uid:
                raw_uid = f"sh_{os.urandom(16).hex()}"
            shape_uid = raw_uid
            while shape_uid in used_uids:
                shape_uid = f"sh_{os.urandom(16).hex()}"
            used_uids.add(shape_uid)
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
            item.setFlags(
                QGraphicsEllipseItem.ItemIsSelectable
                | QGraphicsEllipseItem.ItemIsMovable
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
                    "uid": shape_uid,
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
        try:
            if hasattr(self.image_view, "shape_counter"):
                self.image_view.shape_counter = (
                    max(used_ids) + 1 if used_ids else 1
                )
        except Exception:
            pass

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
            pt_uid = pt.get("uid")

            # assign an id if missing
            if pt_id is None:
                pt_id = self.next_point_id
                self.next_point_id += 1
            if not pt_uid:
                try:
                    pt_uid = f"{int(pt_id)}_{os.urandom(4).hex()}"
                except Exception:
                    pt_uid = f"0_{os.urandom(4).hex()}"

            if use_new_system:
                # Use new ZonePointsRenderer system
                # Create point using the new renderer
                # Restore saved radius if available, otherwise use default
                saved_radius = pt.get("radius")
                if saved_radius is not None and saved_radius > 0:
                    radius = saved_radius
                elif pt_type == "user":
                    radius = 10  # Default radius for user points
                else:
                    radius = 5  # Default radius for generated points

                point_item = ZonePointsRenderer.create_point_item(
                    x, y, pt_id, pt_type, point_uid=pt_uid
                )
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
                    marker.setData(2, pt_uid)
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
                    marker.setData(2, pt_uid)
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
            point_item = (
                generated_points[pt_idx]
                if pt_idx < len(generated_points)
                else user_points[pt_idx - len(generated_points)]
            )
            uid = None
            try:
                existing = point_item.data(2)
                if isinstance(existing, bytes):
                    existing = existing.decode("utf-8", errors="replace")
                if isinstance(existing, str) and existing.strip():
                    uid = existing.strip()
            except Exception:
                uid = None
            if not uid:
                uid = f"{idx + 1}_{os.urandom(4).hex()}"
                try:
                    point_item.setData(2, uid)
                except Exception:
                    pass
            measurement_points.append(
                {
                    "unique_id": uid,
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
