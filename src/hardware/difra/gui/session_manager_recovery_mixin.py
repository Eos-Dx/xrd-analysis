"""Crash-recovery methods extracted from SessionManager."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionManagerRecoveryMixin:
    """Recovery-oriented helpers for incomplete point measurements."""

    def _extract_point_index_from_measurement_path(self, measurement_path: str) -> int:
        parts = str(measurement_path).strip("/").split("/")
        point_token = next((part for part in parts if part.startswith("pt_")), None)
        if point_token is None:
            raise ValueError(f"Cannot parse point index from measurement path: {measurement_path}")
        try:
            return int(point_token.split("_")[-1])
        except Exception as exc:
            raise ValueError(
                f"Invalid point token '{point_token}' in measurement path: {measurement_path}"
            ) from exc

    def _load_incomplete_measurements_from_container(self, session_file: Path) -> List[Dict]:
        import h5py
        import numpy as np

        incomplete: List[Dict] = []
        with h5py.File(session_file, "r") as f:
            points_group = f.get(self.schema.GROUP_POINTS, {})
            measurements_group = f.get(self.schema.GROUP_MEASUREMENTS, {})

            for point_id in measurements_group.keys():
                try:
                    point_index = int(point_id.split("_")[-1])
                except Exception:
                    continue

                point_group = measurements_group[point_id]
                point_info = points_group.get(point_id)
                pixel_coordinates: List[float] = []
                physical_coordinates_mm: List[float] = []
                point_status = ""

                if point_info is not None:
                    pixel_raw = point_info.attrs.get(self.schema.ATTR_PIXEL_COORDINATES, [])
                    phys_raw = point_info.attrs.get(self.schema.ATTR_PHYSICAL_COORDINATES_MM, [])
                    pixel_array = np.asarray(pixel_raw, dtype=float)
                    phys_array = np.asarray(phys_raw, dtype=float)
                    pixel_coordinates = pixel_array.tolist() if pixel_array.size else []
                    physical_coordinates_mm = phys_array.tolist() if phys_array.size else []
                    point_status = self._as_text(
                        point_info.attrs.get(self.schema.ATTR_POINT_STATUS, ""),
                        "",
                    )

                for measurement_id in point_group.keys():
                    measurement_group = point_group[measurement_id]
                    status = self._as_text(
                        measurement_group.attrs.get(self.schema.ATTR_MEASUREMENT_STATUS, ""),
                        "",
                    ).lower()
                    if status != self.schema.STATUS_IN_PROGRESS:
                        continue

                    measurement_path = f"{self.schema.GROUP_MEASUREMENTS}/{point_id}/{measurement_id}"
                    detector_roles = [
                        key for key in measurement_group.keys() if str(key).startswith("det_")
                    ]
                    incomplete.append(
                        {
                            "point_index": point_index,
                            "point_id": point_id,
                            "measurement_id": measurement_id,
                            "measurement_path": measurement_path,
                            "measurement_counter": int(
                                measurement_group.attrs.get(self.schema.ATTR_MEASUREMENT_COUNTER, 0)
                            ),
                            "timestamp_start": self._as_text(
                                measurement_group.attrs.get(self.schema.ATTR_TIMESTAMP_START, ""),
                                "",
                            ),
                            "timestamp_end": self._as_text(
                                measurement_group.attrs.get(self.schema.ATTR_TIMESTAMP_END, ""),
                                "",
                            ),
                            "measurement_status": status,
                            "point_status": point_status,
                            "pixel_coordinates": pixel_coordinates,
                            "physical_coordinates_mm": physical_coordinates_mm,
                            "detector_roles_present": detector_roles,
                        }
                    )

        incomplete.sort(key=lambda item: (item["point_index"], item["measurement_counter"]))
        return incomplete

    def list_incomplete_measurements(self) -> List[Dict]:
        """List in-progress measurements that need crash recovery decisions."""
        self._check_active()
        return self._load_incomplete_measurements_from_container(self.session_path)

    def _expected_detector_aliases(self) -> List[str]:
        detectors = self.config.get("detectors", [])
        if not isinstance(detectors, list) or not detectors:
            return []

        active_key = "dev_active_detectors" if self.config.get("DEV") else "active_detectors"
        active_ids = set(self.config.get(active_key, []) or [])

        aliases: List[str] = []
        for detector in detectors:
            alias = detector.get("alias")
            if not alias:
                continue
            detector_id = detector.get("id")
            if active_ids and detector_id not in active_ids:
                continue
            aliases.append(str(alias))

        if aliases:
            return aliases

        return [str(detector.get("alias")) for detector in detectors if detector.get("alias")]

    def _measurement_context(self, measurement_path: str) -> Dict:
        import h5py
        import numpy as np

        point_index = self._extract_point_index_from_measurement_path(measurement_path)
        point_id = self.schema.format_point_id(point_index)
        with h5py.File(self.session_path, "r") as f:
            if measurement_path not in f:
                raise KeyError(f"Measurement path not found in container: {measurement_path}")
            measurement_group = f[measurement_path]
            point_group = f.get(f"{self.schema.GROUP_POINTS}/{point_id}")

            physical_coordinates_mm = []
            pixel_coordinates = []
            if point_group is not None:
                phys_raw = point_group.attrs.get(self.schema.ATTR_PHYSICAL_COORDINATES_MM, [])
                pix_raw = point_group.attrs.get(self.schema.ATTR_PIXEL_COORDINATES, [])
                phys_array = np.asarray(phys_raw, dtype=float)
                pix_array = np.asarray(pix_raw, dtype=float)
                physical_coordinates_mm = phys_array.tolist() if phys_array.size else []
                pixel_coordinates = pix_array.tolist() if pix_array.size else []

            return {
                "point_index": point_index,
                "point_id": point_id,
                "measurement_path": measurement_path,
                "measurement_status": self._as_text(
                    measurement_group.attrs.get(self.schema.ATTR_MEASUREMENT_STATUS, ""),
                    "",
                ).lower(),
                "timestamp_start": self._as_text(
                    measurement_group.attrs.get(self.schema.ATTR_TIMESTAMP_START, ""),
                    "",
                ),
                "timestamp_end": self._as_text(
                    measurement_group.attrs.get(self.schema.ATTR_TIMESTAMP_END, ""),
                    "",
                ),
                "physical_coordinates_mm": physical_coordinates_mm,
                "pixel_coordinates": pixel_coordinates,
            }

    def scan_recovery_files_for_measurement(
        self,
        measurement_path: str,
        measurement_folder: Union[str, Path],
        expected_aliases: Optional[List[str]] = None,
    ) -> Dict:
        """Scan measurement folder for npy payload candidates for an in-progress point."""
        import numpy as np

        folder = Path(measurement_folder)
        context = self._measurement_context(measurement_path)

        if expected_aliases is None:
            expected_aliases = self._expected_detector_aliases()

        if not folder.exists():
            return {
                **context,
                "measurement_folder": str(folder),
                "expected_aliases": expected_aliases,
                "files_by_alias": {},
                "missing_aliases": expected_aliases,
                "unreadable_aliases": [],
                "is_complete": False,
            }

        all_npy_files = sorted(folder.glob("*.npy"), key=lambda item: item.stat().st_mtime, reverse=True)
        x_token = y_token = None
        if len(context["physical_coordinates_mm"]) >= 2:
            x_token = f"{float(context['physical_coordinates_mm'][0]):.2f}"
            y_token = f"{float(context['physical_coordinates_mm'][1]):.2f}"
        coordinate_token = f"_{x_token}_{y_token}_" if x_token is not None and y_token is not None else None

        timestamp_token = None
        timestamp_start = context.get("timestamp_start", "")
        if timestamp_start:
            try:
                timestamp_token = datetime.strptime(timestamp_start, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S")
            except Exception:
                timestamp_token = None

        def _select_best_candidate(alias: str) -> Optional[Path]:
            alias_upper = alias.upper()
            candidates = []
            for candidate in all_npy_files:
                stem_upper = candidate.stem.upper()
                if "_ATTENUATION" in stem_upper:
                    continue
                if not stem_upper.endswith(f"_{alias_upper}"):
                    continue
                candidates.append(candidate)
            if not candidates:
                return None

            strict = candidates
            if coordinate_token is not None:
                coord_filtered = [item for item in strict if coordinate_token in item.stem]
                if coord_filtered:
                    strict = coord_filtered
            if timestamp_token is not None:
                timestamp_filtered = [item for item in strict if timestamp_token in item.stem]
                if timestamp_filtered:
                    strict = timestamp_filtered

            return strict[0] if strict else None

        if not expected_aliases:
            inferred = []
            for item in all_npy_files:
                stem_upper = item.stem.upper()
                if "_ATTENUATION" in stem_upper:
                    continue
                tail = item.stem.rsplit("_", 1)[-1]
                if tail and tail not in inferred:
                    inferred.append(tail)
            expected_aliases = inferred

        files_by_alias: Dict[str, str] = {}
        missing_aliases: List[str] = []
        for alias in expected_aliases:
            candidate = _select_best_candidate(alias)
            if candidate is None:
                missing_aliases.append(alias)
            else:
                files_by_alias[alias] = str(candidate)

        unreadable_aliases: List[str] = []
        for alias, path_str in files_by_alias.items():
            try:
                np.load(path_str, mmap_mode="r")
            except Exception:
                unreadable_aliases.append(alias)

        return {
            **context,
            "measurement_folder": str(folder),
            "expected_aliases": expected_aliases,
            "files_by_alias": files_by_alias,
            "missing_aliases": missing_aliases,
            "unreadable_aliases": unreadable_aliases,
            "is_complete": not missing_aliases and not unreadable_aliases and bool(files_by_alias),
        }

    def finalize_incomplete_measurement_from_files(
        self,
        measurement_path: str,
        files_by_alias: Dict[str, Union[str, Path]],
        integration_time_ms: float = 0.0,
        timestamp_end: Optional[str] = None,
    ) -> str:
        """Finalize an in-progress measurement by loading detector arrays from npy files."""
        import numpy as np

        self._check_active()
        context = self._measurement_context(measurement_path)
        if context["measurement_status"] != self.schema.STATUS_IN_PROGRESS:
            raise ValueError(
                f"Measurement is not in-progress and cannot be recovered: {measurement_path}"
            )

        if timestamp_end is None:
            timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        detector_cfg_lookup = {
            str(detector.get("alias")): detector
            for detector in self.config.get("detectors", [])
            if detector.get("alias")
        }

        measurement_data: Dict[str, np.ndarray] = {}
        detector_metadata: Dict[str, Dict] = {}
        poni_alias_map: Dict[str, str] = {}
        raw_files_by_detector: Dict[str, Dict[str, bytes]] = {}

        for alias, file_ref in files_by_alias.items():
            file_path = Path(file_ref)
            if not file_path.exists():
                raise FileNotFoundError(f"Recovery file not found for detector {alias}: {file_path}")

            detector_signal = np.load(file_path)
            detector_cfg = detector_cfg_lookup.get(str(alias), {})
            detector_id = str(detector_cfg.get("id") or alias)

            measurement_data[detector_id] = detector_signal
            poni_alias_map[str(alias)] = detector_id
            detector_metadata[detector_id] = {
                "integration_time_ms": float(integration_time_ms),
                "detector_id": detector_id,
                "timestamp": timestamp_end,
                "recovered_from_file": str(file_path),
            }
            if len(context["physical_coordinates_mm"]) >= 2:
                detector_metadata[detector_id]["x_mm"] = float(context["physical_coordinates_mm"][0])
                detector_metadata[detector_id]["y_mm"] = float(context["physical_coordinates_mm"][1])

            raw_files = {}
            for extension in (".txt", ".dsc", ".t3pa"):
                raw_path = file_path.with_suffix(extension)
                if raw_path.exists():
                    try:
                        raw_files[f"raw_{extension[1:]}"] = raw_path.read_bytes()
                    except Exception:
                        continue
            if raw_files:
                raw_files_by_detector[detector_id] = raw_files

        if not measurement_data:
            raise ValueError(f"No detector payload found to recover measurement: {measurement_path}")

        self.writer.finalize_measurement(
            file_path=self.session_path,
            measurement_path=measurement_path,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            raw_files=raw_files_by_detector if raw_files_by_detector else None,
            timestamp_end=timestamp_end,
            measurement_status=self.schema.STATUS_COMPLETED,
        )
        self.writer.update_point_status(
            file_path=self.session_path,
            point_index=context["point_index"],
            point_status=self.schema.POINT_STATUS_MEASURED,
        )
        self._pending_measurements.pop(context["point_index"], None)
        self.log_event(
            message="Recovered point measurement from on-disk files",
            event_type="measurement_recovered_from_files",
            details={
                "point_index": context["point_index"],
                "measurement_path": measurement_path,
                "files_by_alias": {alias: str(path) for alias, path in files_by_alias.items()},
            },
        )
        return measurement_path

    def abort_incomplete_measurement(
        self,
        measurement_path: str,
        reason: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        measurement_status: Optional[str] = None,
    ) -> str:
        """Mark in-progress measurement as aborted (or failed) during recovery."""
        self._check_active()
        context = self._measurement_context(measurement_path)
        if context["measurement_status"] != self.schema.STATUS_IN_PROGRESS:
            return measurement_path

        terminal_status = measurement_status or self.schema.STATUS_ABORTED
        self.writer.fail_measurement(
            file_path=self.session_path,
            measurement_path=measurement_path,
            failure_reason=reason,
            timestamp_end=timestamp_end or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            measurement_status=terminal_status,
        )
        self.writer.update_point_status(
            file_path=self.session_path,
            point_index=context["point_index"],
            point_status=self.schema.POINT_STATUS_PENDING,
        )
        self._pending_measurements.pop(context["point_index"], None)
        self.log_event(
            message="Point measurement marked for re-measurement after restore",
            event_type="measurement_recovery_aborted",
            level="WARNING",
            details={
                "point_index": context["point_index"],
                "measurement_path": measurement_path,
                "status": terminal_status,
                "reason": reason or "",
            },
        )
        return measurement_path
