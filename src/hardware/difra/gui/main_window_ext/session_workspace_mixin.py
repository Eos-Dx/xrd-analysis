"""Workspace/session container synchronization helpers for SessionMixin."""

from . import session_mixin as _session_module

json = _session_module.json
Path = _session_module.Path
get_schema = _session_module.get_schema
get_writer = _session_module.get_writer
logger = _session_module.logger


class SessionWorkspaceMixin:
    @staticmethod
    def _normalize_xy_pair(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return (float(value[0]), float(value[1]))
            except Exception:
                return None
        return None

    def _set_spin_value_if_present(self, attr_name: str, value: float):
        widget = getattr(self, attr_name, None)
        if widget is None:
            return
        setter = getattr(widget, "setValue", None)
        if callable(setter):
            try:
                setter(float(value))
            except Exception:
                pass

    def _get_stage_reference_mm(self):
        ref_x = 0.0
        ref_y = 0.0

        x_widget = getattr(self, "real_x_pos_mm", None)
        y_widget = getattr(self, "real_y_pos_mm", None)
        x_value = getattr(x_widget, "value", None)
        y_value = getattr(y_widget, "value", None)
        if callable(x_value):
            try:
                ref_x = float(x_value())
            except Exception:
                ref_x = 0.0
        if callable(y_value):
            try:
                ref_y = float(y_value())
            except Exception:
                ref_y = 0.0

        return ref_x, ref_y

    def _get_include_center_px(self):
        center = self._normalize_xy_pair(getattr(self, "include_center", None))
        if center is not None:
            return center
        return (0.0, 0.0)

    def _pixel_to_physical_mm(self, x_px: float, y_px: float):
        try:
            ratio = float(getattr(self, "pixel_to_mm_ratio", 0.0))
        except Exception:
            ratio = 0.0
        if ratio == 0.0:
            return (0.0, 0.0)

        center_x, center_y = self._get_include_center_px()
        ref_x_mm, ref_y_mm = self._get_stage_reference_mm()
        x_mm = ref_x_mm - (float(x_px) - center_x) / ratio
        y_mm = ref_y_mm - (float(y_px) - center_y) / ratio
        return (float(x_mm), float(y_mm))

    def _build_mapping_conversion_payload(self):
        try:
            ratio = float(getattr(self, "pixel_to_mm_ratio", 0.0))
        except Exception:
            ratio = 0.0
        center_x, center_y = self._get_include_center_px()
        ref_x_mm, ref_y_mm = self._get_stage_reference_mm()

        return {
            "ratio": ratio,
            "units": "mm/pixel",
            "include_center_px": [float(center_x), float(center_y)],
            "stage_reference_mm": [float(ref_x_mm), float(ref_y_mm)],
            "formula": "x_mm = ref_x_mm - (x_px - center_x_px) / ratio",
        }

    def _image_file_signature(self):
        if not hasattr(self, "image_view"):
            return None
        image_path = getattr(self.image_view, "current_image_path", None)
        if not image_path:
            return None

        path = Path(str(image_path))
        if not path.exists():
            return {"path": str(path), "missing": True}

        try:
            stat = path.stat()
            return {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except Exception:
            return {"path": str(path)}

    @staticmethod
    def _sync_signature(payload):
        import hashlib

        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.md5(encoded.encode("utf-8")).hexdigest()

    def _load_image_array_from_path(self, image_path):
        """Load image from disk and normalize color channels to RGB/RGBA."""
        if not image_path:
            return None

        try:
            import cv2

            image_array = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image_array is not None:
                if image_array.ndim == 3:
                    channels = int(image_array.shape[2])
                    if channels == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    elif channels == 4:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)
                return image_array
        except Exception:
            pass

        try:
            import numpy as np
            from PIL import Image

            return np.array(Image.open(image_path))
        except Exception as exc:
            logger.warning(f"Failed to load image for session sync: {exc}")
            return None

    def _set_image_from_array(self, image_array):
        """Render numpy image array into image_view when available."""
        if not hasattr(self, "image_view"):
            return False

        try:
            import numpy as np
            from PyQt5.QtGui import QImage, QPixmap

            array = np.asarray(image_array)
            if array.ndim == 2:
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                height, width = array.shape
                qimage = QImage(
                    array.data, width, height, array.strides[0], QImage.Format_Grayscale8
                ).copy()
            elif array.ndim == 3 and array.shape[2] in (3, 4):
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                height, width, channels = array.shape
                fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
                qimage = QImage(
                    array.data, width, height, array.strides[0], fmt
                ).copy()
            else:
                return False

            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                return False

            self.image_view.set_image(pixmap, image_path=None)
            return True
        except Exception as exc:
            logger.warning(f"Failed to set image from session array: {exc}")
            return False

    def _restore_session_workspace_from_container(self, session_path: Path):
        """Restore image/zones/points from an existing session container into GUI."""
        if not hasattr(self, "state"):
            self.state = {}

        try:
            import h5py
            schema = get_schema(self.config if hasattr(self, "config") else None)

            restored_shapes = []
            restored_points = []
            restored_image = False
            restored_ratio = None
            restored_include_center = None
            restored_stage_reference = None

            with h5py.File(session_path, "r") as h5f:
                # Restore sample image (use first available image dataset)
                images_group = h5f.get(schema.GROUP_IMAGES)
                if images_group:
                    image_keys = sorted(
                        key for key in images_group.keys() if key.startswith("img_")
                    )
                    if image_keys:
                        image_group = images_group[image_keys[0]]
                        if "data" in image_group:
                            restored_image = self._set_image_from_array(image_group["data"][:])

                # Restore zones -> state shape structure
                zones_group = h5f.get(schema.GROUP_IMAGES_ZONES)
                if zones_group:
                    for index, zone_id in enumerate(sorted(zones_group.keys()), start=1):
                        zone_group = zones_group[zone_id]
                        zone_role = str(
                            self._decode_attr(
                                zone_group.attrs.get(schema.ATTR_ZONE_ROLE, "sample_holder")
                            )
                        ).lower()
                        shape_name = str(
                            self._decode_attr(zone_group.attrs.get(schema.ATTR_ZONE_SHAPE, "circle"))
                        ).lower()
                        geometry_value = None
                        if "geometry_px" in zone_group:
                            raw_geometry = zone_group["geometry_px"][()]
                            if isinstance(raw_geometry, bytes):
                                raw_geometry = raw_geometry.decode("utf-8", errors="replace")
                            geometry_value = raw_geometry

                        x = y = width = height = 0.0
                        if isinstance(geometry_value, str):
                            parsed = json.loads(geometry_value)
                            if isinstance(parsed, dict):
                                if "center" in parsed and "radius" in parsed:
                                    center_x, center_y = parsed.get("center", [0, 0])
                                    radius = float(parsed.get("radius", 0))
                                    x = float(center_x) - radius
                                    y = float(center_y) - radius
                                    width = height = radius * 2.0
                                else:
                                    x = float(parsed.get("x", 0))
                                    y = float(parsed.get("y", 0))
                                    width = float(parsed.get("width", 0))
                                    height = float(parsed.get("height", 0))
                            elif isinstance(parsed, list) and len(parsed) >= 4:
                                x, y, width, height = [float(parsed[i]) for i in range(4)]
                        elif geometry_value is not None:
                            values = list(geometry_value)
                            if len(values) >= 4:
                                x, y, width, height = [float(values[i]) for i in range(4)]

                        ui_role = "include" if zone_role == "sample_holder" else zone_role
                        restored_shapes.append(
                            {
                                "id": index,
                                "type": "circle" if shape_name == "circle" else "rectangle",
                                "role": ui_role,
                                "geometry": {
                                    "x": x,
                                    "y": y,
                                    "width": width,
                                    "height": height,
                                },
                            }
                        )

                # Restore points as generated points
                points_group = h5f.get(schema.GROUP_POINTS)
                if points_group:
                    for point_id in sorted(points_group.keys()):
                        point_group = points_group[point_id]
                        pixel_coords = point_group.attrs.get(schema.ATTR_PIXEL_COORDINATES, [])
                        if len(pixel_coords) < 2:
                            continue
                        point_index = int(point_id.split("_")[-1])
                        point_uid = self._decode_attr(
                            point_group.attrs.get("point_uid", "")
                        )
                        restored_points.append(
                            {
                                "id": point_index,
                                "x": float(pixel_coords[0]),
                                "y": float(pixel_coords[1]),
                                "type": "generated",
                                "radius": 5.0,
                                "uid": point_uid if point_uid else None,
                            }
                        )

                # Restore mapping ratio if available
                mapping_ds = h5f.get(f"{schema.GROUP_IMAGES_MAPPING}/mapping")
                if mapping_ds is not None:
                    mapping_raw = mapping_ds[()]
                    if isinstance(mapping_raw, bytes):
                        mapping_raw = mapping_raw.decode("utf-8", errors="replace")
                    mapping = json.loads(mapping_raw)
                    conversion = mapping.get("pixel_to_mm_conversion", {})
                    if isinstance(conversion, dict):
                        if "ratio" in conversion:
                            restored_ratio = float(conversion["ratio"])
                        restored_include_center = self._normalize_xy_pair(
                            conversion.get("include_center_px")
                            or conversion.get("center_px")
                            or conversion.get("image_center_px")
                        )
                        restored_stage_reference = self._normalize_xy_pair(
                            conversion.get("stage_reference_mm")
                            or conversion.get("reference_mm")
                            or conversion.get("stage_origin_mm")
                        )

            self.state["shapes"] = restored_shapes
            self.state["zone_points"] = restored_points

            if hasattr(self, "_restore_shapes"):
                self._restore_shapes(restored_shapes)
            if hasattr(self, "_restore_points"):
                self._restore_points(restored_points)
            if hasattr(self, "_refresh_id_counter"):
                self._refresh_id_counter()

            if hasattr(self, "update_points_table"):
                self.update_points_table()
            if hasattr(self, "update_shape_table"):
                self.update_shape_table()

            # Apply mapping fields after shape-table updates so inferred defaults
            # do not override container-restored calibration origin.
            if restored_ratio is not None:
                self.pixel_to_mm_ratio = restored_ratio
            if restored_include_center is not None:
                self.include_center = (
                    float(restored_include_center[0]),
                    float(restored_include_center[1]),
                )
            if restored_stage_reference is not None:
                self._set_spin_value_if_present("real_x_pos_mm", restored_stage_reference[0])
                self._set_spin_value_if_present("real_y_pos_mm", restored_stage_reference[1])
            if hasattr(self, "update_conversion_label"):
                self.update_conversion_label()
            if hasattr(self, "update_coordinates"):
                self.update_coordinates()

            # Rebuild per-point measurement history from session container datasets.
            self._restore_measurement_history_from_session(session_path)

            logger.info(
                f"Restored workspace from session container: session={session_path} "
                f"image_loaded={restored_image} shapes={len(restored_shapes)} "
                f"points={len(restored_points)}"
            )
            if hasattr(self, "_append_session_log"):
                self._append_session_log(
                    f"Workspace restored from session: shapes={len(restored_shapes)}, points={len(restored_points)}"
                )
        except Exception as exc:
            logger.warning(
                f"Session workspace restore failed for {session_path}: {exc}",
                exc_info=True,
            )
            if hasattr(self, "_append_session_log"):
                self._append_session_log(
                    f"Workspace restore failed: {type(exc).__name__}"
                )

    def _restore_measurement_history_from_session(self, session_path: Path):
        """Populate per-point measurement widgets from session container payloads."""
        if not hasattr(self, "measurement_widgets"):
            self.measurement_widgets = {}
        if not hasattr(self, "add_measurement_widget_to_panel"):
            return

        try:
            import h5py

            schema = get_schema(self.config if hasattr(self, "config") else None)
            detector_id_to_alias = {}
            for detector_cfg in (self.config or {}).get("detectors", []):
                detector_id = detector_cfg.get("id")
                detector_alias = detector_cfg.get("alias")
                if detector_id and detector_alias:
                    detector_id_to_alias[str(detector_id)] = str(detector_alias)

            with h5py.File(session_path, "r") as h5f:
                measurements_group = h5f.get(schema.GROUP_MEASUREMENTS)
                if measurements_group is None:
                    return

                for point_group_name in sorted(measurements_group.keys()):
                    if not str(point_group_name).startswith("pt_"):
                        continue
                    try:
                        point_index = int(str(point_group_name).split("_")[-1])
                    except Exception:
                        continue

                    self.add_measurement_widget_to_panel(point_index)
                    widget = self.measurement_widgets.get(point_index)
                    if widget is None:
                        continue

                    point_group = measurements_group[point_group_name]
                    for meas_name in sorted(point_group.keys()):
                        meas_group = point_group[meas_name]
                        timestamp = self._decode_attr(
                            meas_group.attrs.get(schema.ATTR_TIMESTAMP, "")
                        )
                        results = {}
                        for detector_group_name in sorted(meas_group.keys()):
                            if not str(detector_group_name).startswith("det_"):
                                continue
                            detector_group = meas_group[detector_group_name]
                            detector_alias = self._decode_attr(
                                detector_group.attrs.get(
                                    schema.ATTR_DETECTOR_ALIAS,
                                    "",
                                )
                            )
                            if not detector_alias:
                                detector_id = self._decode_attr(
                                    detector_group.attrs.get(schema.ATTR_DETECTOR_ID, "")
                                )
                                detector_alias = detector_id_to_alias.get(
                                    str(detector_id), str(detector_group_name).replace("det_", "").upper()
                                )

                            dataset_name = schema.DATASET_PROCESSED_SIGNAL
                            if dataset_name not in detector_group:
                                continue
                            dataset_path = (
                                f"{schema.GROUP_MEASUREMENTS}/{point_group_name}/"
                                f"{meas_name}/{detector_group_name}/{dataset_name}"
                            )
                            h5_ref = f"h5ref://{session_path}#{dataset_path}"
                            results[str(detector_alias)] = {
                                "filename": h5_ref,
                                "goodness": None,
                            }

                        if results:
                            widget.add_measurement(results, timestamp or "from container")
        except Exception as exc:
            logger.warning(
                f"Failed to restore measurement history from session container {session_path}: {exc}",
                exc_info=True,
            )

    def _extract_current_image_array(self):
        """Read current sample image into numpy array for session sync."""
        if not hasattr(self, "image_view"):
            return None

        image_path = getattr(self.image_view, "current_image_path", None)
        if not image_path:
            return None

        return self._load_image_array_from_path(image_path)

    def sync_workspace_to_session_container(self, state=None):
        """Persist image/zones/points snapshot into active unlocked session container."""
        if not hasattr(self, "session_manager"):
            return
        if not self.session_manager.is_session_active():
            return
        if self.session_manager.is_locked():
            return

        if state is None:
            state = getattr(self, "state", None) or {}

        try:
            import h5py
            schema = get_schema(self.config if hasattr(self, "config") else None)
            writer = get_writer(self.config if hasattr(self, "config") else None)

            session_path = self.session_manager.session_path
            session_path_str = str(session_path)
            if getattr(self, "_session_sync_cache_session_path", None) != session_path_str:
                self._session_sync_cache_session_path = session_path_str
                self._session_sync_shapes_sig = None
                self._session_sync_mapping_sig = None
                self._session_sync_points_sig = None
                self._session_sync_last_image_sig = None

            shapes = state.get("shapes", [])
            points = state.get("zone_points", [])
            mapping_conversion = self._build_mapping_conversion_payload()
            image_sig = self._image_file_signature()

            with h5py.File(session_path, "r") as h5f:
                image_exists = f"{schema.GROUP_IMAGES}/img_001/data" in h5f
                zones_exist = schema.GROUP_IMAGES_ZONES in h5f
                mapping_exists = f"{schema.GROUP_IMAGES_MAPPING}/mapping" in h5f
                points_exist = schema.GROUP_POINTS in h5f
                has_measurements = False
                measurements = h5f.get(schema.GROUP_MEASUREMENTS)
                if measurements:
                    for point_group in measurements.values():
                        if len(point_group.keys()) > 0:
                            has_measurements = True
                            break

            shapes_sig = self._sync_signature({"shapes": shapes})
            mapping_sig = self._sync_signature({"mapping": mapping_conversion})
            points_sig = self._sync_signature({"points": points})

            needs_shapes_sync = (not zones_exist) or (shapes_sig != getattr(self, "_session_sync_shapes_sig", None))
            needs_mapping_sync = (not mapping_exists) or (mapping_sig != getattr(self, "_session_sync_mapping_sig", None))
            effective_points_sig = points_sig if not has_measurements else "__points_locked_after_measurements__"
            needs_points_sync = (not has_measurements) and (
                (not points_exist) or (points_sig != getattr(self, "_session_sync_points_sig", None))
            )
            needs_image_sync = not image_exists

            overall_sig = self._sync_signature(
                {
                    "session": session_path_str,
                    "image_sig": image_sig,
                    "image_exists": image_exists,
                    "shapes_sig": shapes_sig,
                    "mapping_sig": mapping_sig,
                    "effective_points_sig": effective_points_sig,
                }
            )
            if overall_sig == getattr(self, "_session_sync_overall_sig", None):
                return

            did_write = False
            if needs_image_sync:
                image_array = self._extract_current_image_array()
                if image_array is not None:
                    writer.add_image(
                        file_path=session_path,
                        image_index=1,
                        image_data=image_array,
                        image_type="sample",
                    )
                    did_write = True
            else:
                last_image_sig = getattr(self, "_session_sync_last_image_sig", None)
                if (
                    image_sig is not None
                    and last_image_sig is not None
                    and image_sig != last_image_sig
                ):
                    logger.warning(
                        "Session image is immutable after first write; ignoring changed source image path"
                    )

            if needs_shapes_sync:
                with h5py.File(session_path, "a") as h5f:
                    if schema.GROUP_IMAGES_ZONES in h5f:
                        del h5f[schema.GROUP_IMAGES_ZONES]
                    h5f.create_group(schema.GROUP_IMAGES_ZONES)
                did_write = True

                for zone_index, shape in enumerate(shapes, start=1):
                    role = str(shape.get("role", "include")).lower()
                    zone_role = "exclude" if role == "exclude" else "sample_holder"
                    shape_type = str(shape.get("type", "circle")).lower()
                    geometry = shape.get("geometry", {})
                    geometry_px = [
                        float(geometry.get("x", 0)),
                        float(geometry.get("y", 0)),
                        float(geometry.get("width", 0)),
                        float(geometry.get("height", 0)),
                    ]
                    holder_diameter_mm = None
                    if zone_role == "sample_holder" and hasattr(self, "pixel_to_mm_ratio"):
                        diameter_px = max(geometry_px[2], geometry_px[3])
                        if getattr(self, "pixel_to_mm_ratio", 0):
                            holder_diameter_mm = diameter_px / float(self.pixel_to_mm_ratio)

                    writer.add_zone(
                        file_path=session_path,
                        zone_index=zone_index,
                        zone_role=zone_role,
                        geometry_px=geometry_px,
                        shape=shape_type,
                        holder_diameter_mm=holder_diameter_mm,
                    )

            if needs_mapping_sync and hasattr(self, "pixel_to_mm_ratio"):
                writer.add_image_mapping(
                    file_path=session_path,
                    sample_holder_zone_id="zone_001",
                    pixel_to_mm_conversion=mapping_conversion,
                    orientation="standard",
                    mapping_version=schema.SCHEMA_VERSION,
                )
                did_write = True

            if needs_points_sync:
                with h5py.File(session_path, "a") as h5f:
                    if schema.GROUP_POINTS in h5f:
                        del h5f[schema.GROUP_POINTS]
                    h5f.create_group(schema.GROUP_POINTS)
                did_write = True

                for point_index, point in enumerate(points, start=1):
                    x = float(point.get("x", 0))
                    y = float(point.get("y", 0))
                    x_mm, y_mm = self._pixel_to_physical_mm(x, y)
                    point_path = writer.add_point(
                        file_path=session_path,
                        point_index=point_index,
                        pixel_coordinates=[x, y],
                        physical_coordinates_mm=[x_mm, y_mm],
                        point_status=schema.POINT_STATUS_PENDING,
                    )
                    point_uid = str(point.get("uid") or "").strip()
                    if point_uid:
                        with h5py.File(session_path, "a") as h5f:
                            h5f[point_path].attrs["point_uid"] = point_uid

            self._session_sync_shapes_sig = shapes_sig
            self._session_sync_mapping_sig = mapping_sig
            self._session_sync_points_sig = effective_points_sig
            self._session_sync_last_image_sig = image_sig
            self._session_sync_overall_sig = overall_sig

            if did_write and hasattr(self, "_append_session_log"):
                self._append_session_log("Session workspace snapshot updated")

        except Exception as exc:
            logger.warning(
                f"Workspace snapshot sync to session failed: {exc}",
                exc_info=True,
            )
            if hasattr(self, "_append_session_log"):
                self._append_session_log(
                    f"Session workspace sync failed: {type(exc).__name__}"
                )
    
