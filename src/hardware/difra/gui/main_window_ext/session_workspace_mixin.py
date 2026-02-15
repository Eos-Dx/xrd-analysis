"""Workspace/session container synchronization helpers for SessionMixin."""

from . import session_mixin as _session_module

json = _session_module.json
Path = _session_module.Path
get_schema = _session_module.get_schema
get_writer = _session_module.get_writer
logger = _session_module.logger


class SessionWorkspaceMixin:
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
                        restored_points.append(
                            {
                                "id": point_index,
                                "x": float(pixel_coords[0]),
                                "y": float(pixel_coords[1]),
                                "type": "generated",
                                "radius": 5.0,
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
                    if "ratio" in conversion:
                        restored_ratio = float(conversion["ratio"])

            self.state["shapes"] = restored_shapes
            self.state["zone_points"] = restored_points
            if restored_ratio is not None:
                self.pixel_to_mm_ratio = restored_ratio

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
            if hasattr(self, "update_coordinates"):
                self.update_coordinates()

            logger.info(
                f"Restored workspace from session container: session={session_path} "
                f"image_loaded={restored_image} shapes={len(restored_shapes)} "
                f"points={len(restored_points)}"
            )
        except Exception as exc:
            logger.warning(
                f"Session workspace restore failed for {session_path}: {exc}",
                exc_info=True,
            )

    def _extract_current_image_array(self):
        """Read current sample image into numpy array for session sync."""
        if not hasattr(self, "image_view"):
            return None

        image_path = getattr(self.image_view, "current_image_path", None)
        if not image_path:
            return None

        try:
            import cv2

            image_array = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image_array is not None:
                return image_array
        except Exception:
            pass

        try:
            import numpy as np
            from PIL import Image

            return np.array(Image.open(image_path))
        except Exception as exc:
            logger.warning(f"Failed to load current image for session sync: {exc}")
            return None

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

            image_array = self._extract_current_image_array()
            if image_array is not None:
                with h5py.File(session_path, "a") as h5f:
                    if schema.GROUP_IMAGES in h5f and "img_001" in h5f[schema.GROUP_IMAGES]:
                        del h5f[f"{schema.GROUP_IMAGES}/img_001"]
                writer.add_image(
                    file_path=session_path,
                    image_index=1,
                    image_data=image_array,
                    image_type="sample",
                )

            shapes = state.get("shapes", [])
            with h5py.File(session_path, "a") as h5f:
                if schema.GROUP_IMAGES_ZONES in h5f:
                    del h5f[schema.GROUP_IMAGES_ZONES]
                h5f.create_group(schema.GROUP_IMAGES_ZONES)

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

            if hasattr(self, "pixel_to_mm_ratio"):
                writer.add_image_mapping(
                    file_path=session_path,
                    sample_holder_zone_id="zone_001",
                    pixel_to_mm_conversion={
                        "ratio": float(self.pixel_to_mm_ratio),
                        "units": "mm/pixel",
                    },
                    orientation="standard",
                    mapping_version=schema.SCHEMA_VERSION,
                )

            # Only rewrite points while there are no recorded measurements.
            has_measurements = False
            with h5py.File(session_path, "r") as h5f:
                measurements = h5f.get(schema.GROUP_MEASUREMENTS)
                if measurements:
                    for point_group in measurements.values():
                        if len(point_group.keys()) > 0:
                            has_measurements = True
                            break

            if not has_measurements:
                points = state.get("zone_points", [])
                with h5py.File(session_path, "a") as h5f:
                    if schema.GROUP_POINTS in h5f:
                        del h5f[schema.GROUP_POINTS]
                    h5f.create_group(schema.GROUP_POINTS)

                for point_index, point in enumerate(points, start=1):
                    x = float(point.get("x", 0))
                    y = float(point.get("y", 0))
                    writer.add_point(
                        file_path=session_path,
                        point_index=point_index,
                        pixel_coordinates=[x, y],
                        physical_coordinates_mm=[0.0, 0.0],
                        point_status=schema.POINT_STATUS_PENDING,
                    )

        except Exception as exc:
            logger.warning(
                f"Workspace snapshot sync to session failed: {exc}",
                exc_info=True,
            )
    
