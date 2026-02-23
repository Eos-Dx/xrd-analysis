"""Session Manager for DIFRA GUI.

Handles HDF5 session container lifecycle:
- Creating new sessions
- Managing active session state
- Writing measurements to containers
- Tracking measurement counters
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

from hardware.difra.gui.container_api import get_container_module
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionManager:
    """Manages HDF5 session containers for DIFRA measurements."""

    @staticmethod
    def _resolve_machine_name(config: Dict) -> str:
        """Resolve machine name from explicit field or selected setup identity."""
        return (
            config.get("machine_name")
            or config.get("setup_name")
            or config.get("name")
            or config.get("default_setup")
            or "DIFRA-01"
        )

    @staticmethod
    def _as_text(value, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize SessionManager.
        
        Args:
            config: Optional configuration dict from global.json
                   If provided, beam_energy_kev will be read from config
        """
        self.session_path: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.sample_id: Optional[str] = None
        self.study_name: Optional[str] = None
        self.technical_container_path: Optional[Path] = None
        
        # Track counters for linking
        self.i0_counter: Optional[int] = None  # Attenuation without sample
        self.i_counter: Optional[int] = None   # Attenuation with sample
        # Track in-progress point measurements for crash recovery metadata.
        self._pending_measurements: Dict[int, str] = {}
        
        # Store config for later use
        self.config = config or {}
        self.container_module = get_container_module(self.config)
        self.schema = self.container_module.schema
        self.writer = self.container_module.writer
        self.container_manager = self.container_module.container_manager
        self.producer_software: str = str(
            self.config.get("producer_software")
            or self.config.get("app_name")
            or "difra"
        )
        self.producer_version: str = str(
            self.config.get("producer_version")
            or getattr(self.container_module, "__version__", "unknown")
        )
        
        # Configuration - read from config or use defaults
        if config:
            self.operator_id: str = config.get('operator_id', 'operator')
            self.site_id: str = config.get('site_id', 'DIFRA_LAB')
            self.machine_name: str = self._resolve_machine_name(config)
            self.beam_energy_kev: float = config.get('beam_energy_kev', 17.5)
        else:
            self.operator_id: str = "operator"
            self.site_id: str = "DIFRA_LAB"
            self.machine_name: str = "DIFRA-01"
            self.beam_energy_kev: float = 17.5

    def log_event(
        self,
        message: str,
        event_type: str = "event",
        level: str = "INFO",
        details: Optional[Dict] = None,
    ) -> None:
        """Append session runtime event to container log dataset."""
        if not self.is_session_active():
            return
        append_runtime_log = getattr(self.writer, "append_runtime_log", None)
        if not callable(append_runtime_log):
            return
        append_runtime_log(
            file_path=self.session_path,
            message=message,
            level=level,
            event_type=event_type,
            source=self.producer_software,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details=details or {},
        )
    
    def _get_technical_folder(self) -> Path:
        """Get technical container folder from config.
        
        Returns:
            Path to technical folder from config
        """
        # Try technical_folder from config first
        folder = self.config.get('technical_folder')
        if folder:
            return Path(folder)
        
        # Fall back to difra_base_folder/technical
        base = self.config.get('difra_base_folder')
        if base:
            return Path(base) / 'technical'
        
        # Last resort: home directory
        logger.warning("No technical folder in config, using default")
        return Path.home() / 'difra_technical'
    
    def is_session_active(self) -> bool:
        """Check if a session is currently active."""
        return self.session_path is not None and self.session_path.exists()
    
    def create_session(
        self,
        folder: Path,
        distance_cm: float,
        **session_attrs,
    ) -> Tuple[str, Path]:
        """Create a new session container.
        
        All required session attributes should be provided as keyword arguments.
        These will be passed to the container writer and validated against self.schema.
        
        Required session attributes (from schema):
            sample_id: str - Unique sample identifier
            study_name: str - Study name/identifier (optional, defaults to UNSPECIFIED)
            operator_id: str - Operator ID/name (optional, uses config default)
            site_id: str - Site identifier (optional, uses config default)
            machine_name: str - Machine name (optional, uses config default)
            beam_energy_keV: float - Beam energy (optional, uses config default)
            acquisition_date: str - Acquisition date (optional, auto-generated)
        
        Optional session attributes:
            patient_id: str - Patient identifier
            
        Args:
            folder: Directory for session container (measurements folder)
            distance_cm: Sample-detector distance (for technical container lookup)
            **session_attrs: All session attributes as keyword arguments
            
        Returns:
            Tuple of (session_id, session_path)
            
        Raises:
            RuntimeError: If no valid technical container found
            ValueError: If required session attributes are missing
        """
        schema = self.schema
        writer = self.writer
        find_active_technical_container = self.container_manager.find_active_technical_container
        is_container_locked = self.container_manager.is_container_locked

        # Get technical folder from config
        technical_folder = self._get_technical_folder()
        
        # Find active technical container for this distance in technical folder
        tech_path = find_active_technical_container(
            folder=technical_folder,
            distance_cm=distance_cm,
        )
        
        if not tech_path:
            raise RuntimeError(
                f"No technical container found for distance {distance_cm} cm. "
                "Please create technical measurements first."
            )
        
        if not is_container_locked(tech_path):
            raise RuntimeError(
                f"Technical container is not locked: {tech_path}\n"
                "Please lock the technical container before creating sessions."
            )
        
        # Build session attributes from provided kwargs and config defaults
        # Required attributes from schema
        container_attrs = {
            self.schema.ATTR_SAMPLE_ID: session_attrs.get(
                self.schema.ATTR_SAMPLE_ID,
                session_attrs.get('sample_id'),  # Support both snake_case and schema names
            ),
            self.schema.ATTR_STUDY_NAME: session_attrs.get(
                self.schema.ATTR_STUDY_NAME,
                session_attrs.get('study_name', "UNSPECIFIED"),
            ),
            self.schema.ATTR_OPERATOR_ID: session_attrs.get(
                self.schema.ATTR_OPERATOR_ID,
                session_attrs.get('operator_id', self.operator_id),
            ),
            self.schema.ATTR_SITE_ID: session_attrs.get(
                self.schema.ATTR_SITE_ID,
                session_attrs.get('site_id', self.site_id),
            ),
            self.schema.ATTR_MACHINE_NAME: session_attrs.get(
                self.schema.ATTR_MACHINE_NAME,
                session_attrs.get('machine_name', self.machine_name),
            ),
            self.schema.ATTR_BEAM_ENERGY_KEV: session_attrs.get(
                self.schema.ATTR_BEAM_ENERGY_KEV,
                session_attrs.get('beam_energy_keV', self.beam_energy_kev),
            ),
            self.schema.ATTR_ACQUISITION_DATE: session_attrs.get(
                self.schema.ATTR_ACQUISITION_DATE,
                session_attrs.get('acquisition_date', datetime.now().strftime("%Y-%m-%d")),
            ),
        }

        if hasattr(self.schema, "ATTR_PROJECT_ID"):
            project_attr = self.schema.ATTR_PROJECT_ID
            container_attrs[project_attr] = session_attrs.get(
                project_attr,
                session_attrs.get("project_id", container_attrs[self.schema.ATTR_STUDY_NAME]),
            )
        
        # Add optional attributes if provided
        if self.schema.ATTR_PATIENT_ID in session_attrs or 'patient_id' in session_attrs:
            container_attrs[self.schema.ATTR_PATIENT_ID] = session_attrs.get(
                self.schema.ATTR_PATIENT_ID,
                session_attrs.get('patient_id'),
            )
        
        # Validate required sample_id
        if not container_attrs[self.schema.ATTR_SAMPLE_ID]:
            raise ValueError("sample_id is required to create a session")
        
        sample_id = container_attrs[self.schema.ATTR_SAMPLE_ID]
        study_name = container_attrs[self.schema.ATTR_STUDY_NAME]
        
        logger.info(
            "Creating new session",
            sample_id=sample_id,
            distance_cm=distance_cm,
            technical_container=str(tech_path),
            study_name=study_name,
            operator_id=container_attrs.get(self.schema.ATTR_OPERATOR_ID),
            site_id=container_attrs.get(self.schema.ATTR_SITE_ID),
            machine_name=container_attrs.get(self.schema.ATTR_MACHINE_NAME),
        )
        
        # Create session container with schema-driven attributes
        self.session_id, session_path_str = self.writer.create_session_container(
            folder=folder,
            producer_software=self.producer_software,
            producer_version=self.producer_version,
            **container_attrs,
        )
        
        self.session_path = Path(session_path_str)
        self.sample_id = sample_id
        self.study_name = study_name
        self.technical_container_path = Path(tech_path)
        
        # Copy technical data to session
        self.writer.copy_technical_to_session(
            technical_file=tech_path,
            session_file=self.session_path,
        )
        self.log_event(
            message="Technical snapshot copied into session",
            event_type="technical_snapshot_copied",
            details={"technical_container": str(tech_path)},
        )
        
        logger.info(
            "Session created successfully",
            session_id=self.session_id,
            session_path=str(self.session_path),
        )
        
        # Reset counters
        self.i0_counter = None
        self.i_counter = None
        self._pending_measurements = {}
        
        return self.session_id, self.session_path
    
    def close_session(self):
        """Close the current session and clear state."""
        if self.session_path:
            logger.info(
                "Closing session",
                session_id=self.session_id,
                sample_id=self.sample_id,
            )
        
        self.session_path = None
        self.session_id = None
        self.sample_id = None
        self.study_name = None
        self.technical_container_path = None
        self.i0_counter = None
        self.i_counter = None
        self._pending_measurements = {}

    def open_existing_session(self, session_file: Path) -> Dict:
        """Load metadata from an existing session container into manager state."""
        import h5py

        session_file = Path(session_file)
        if not session_file.exists():
            raise FileNotFoundError(f"Session container not found: {session_file}")

        with h5py.File(session_file, "r") as f:
            self.session_path = session_file
            sample_group = f.get(self.schema.GROUP_SAMPLE)
            user_group = f.get(self.schema.GROUP_USER)
            calibration_snapshot = f.get(self.schema.GROUP_CALIBRATION_SNAPSHOT)

            self.sample_id = self._as_text(
                f.attrs.get(
                    self.schema.ATTR_SAMPLE_ID,
                    sample_group.attrs.get(self.schema.ATTR_SAMPLE_ID) if sample_group else None,
                ),
                "unknown",
            )
            self.study_name = self._as_text(
                f.attrs.get(
                    self.schema.ATTR_STUDY_NAME,
                    sample_group.attrs.get(self.schema.ATTR_STUDY_NAME) if sample_group else None,
                ),
                "UNSPECIFIED",
            )
            self.session_id = self._as_text(
                f.attrs.get(self.schema.ATTR_SESSION_ID),
                "unknown",
            )
            self.operator_id = self._as_text(
                f.attrs.get(
                    self.schema.ATTR_OPERATOR_ID,
                    user_group.attrs.get(self.schema.ATTR_OPERATOR_ID) if user_group else None,
                ),
                self.operator_id,
            )
            self.machine_name = self._as_text(
                f.attrs.get(
                    self.schema.ATTR_MACHINE_NAME,
                    user_group.attrs.get(self.schema.ATTR_MACHINE_NAME) if user_group else None,
                ),
                self.machine_name,
            )

            if calibration_snapshot is not None:
                source = calibration_snapshot.attrs.get("source_file")
                self.technical_container_path = Path(source) if source else None
            else:
                self.technical_container_path = None

        # Rebuild pending map from in-progress measurements for crash recovery.
        incomplete = self._load_incomplete_measurements_from_container(session_file)
        self._pending_measurements = {
            item["point_index"]: item["measurement_path"] for item in incomplete
        }

        return self.get_session_info()

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

    def replace_technical_container(
        self,
        technical_file: Path,
        auto_lock_source: bool = False,
    ) -> None:
        """Replace embedded calibration snapshot in an active unlocked session."""
        self._check_active()

        if self.is_locked():
            raise RuntimeError(
                "Cannot update technical data: session container is locked."
            )

        self.writer.copy_technical_to_session(
            technical_file=technical_file,
            session_file=self.session_path,
            auto_lock=auto_lock_source,
        )
        self.technical_container_path = Path(technical_file)
    
    def add_sample_image(
        self,
        image_data,
        image_index: int = 1,
        image_type: str = "sample",
    ) -> str:
        """Add sample image to session container.
        
        Args:
            image_data: 2D numpy array or path to image file
            image_index: Image index (default 1)
            image_type: Type of image (default "sample")
            
        Returns:
            Image group path
        """
        self._check_active()
        
        return self.writer.add_image(
            file_path=self.session_path,
            image_index=image_index,
            image_data=image_data,
            image_type=image_type,
        )
    
    def add_zone(
        self,
        zone_index: int,
        geometry_px,
        shape: str,
        zone_role: str = "sample_holder",
        holder_diameter_mm: Optional[float] = None,
    ) -> str:
        """Add zone definition to session container.
        
        Args:
            zone_index: Zone index (1-based)
            geometry_px: Geometry in pixels (dict or array)
            shape: Shape type ("circle", "rectangle", etc.)
            zone_role: Role of zone (default "sample_holder")
            holder_diameter_mm: Optional holder diameter in mm
            
        Returns:
            Zone group path
        """
        self._check_active()
        
        return self.writer.add_zone(
            file_path=self.session_path,
            zone_index=zone_index,
            geometry_px=geometry_px,
            shape=shape,
            zone_role=zone_role,
            holder_diameter_mm=holder_diameter_mm,
        )
    
    def add_points(
        self,
        points: List[Dict],
    ) -> List[str]:
        """Add multiple measurement points to session container.
        
        Args:
            points: List of point dicts with keys:
                - pixel_coordinates: [x_px, y_px]
                - physical_coordinates_mm: [x_mm, y_mm]
                - point_status: Optional status (default "pending")
                - thickness: Optional thickness value (default "unknown")
                
        Returns:
            List of point group paths
        """
        self._check_active()
        
        paths = []
        for idx, point in enumerate(points, start=1):
            path = self.writer.add_point(
                file_path=self.session_path,
                point_index=idx,
                pixel_coordinates=point["pixel_coordinates"],
                physical_coordinates_mm=point["physical_coordinates_mm"],
                point_status=point.get("point_status", "pending"),
                thickness=point.get("thickness", "unknown"),
            )
            paths.append(path)

        self.log_event(
            message=f"Generated {len(points)} measurement points",
            event_type="points_generated",
            details={"count": len(points)},
        )
        
        logger.info("Added points to session", num_points=len(points))
        return paths
    
    def add_attenuation_measurement(
        self,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        mode: str,  # "without" or "with"
    ) -> int:
        """Add attenuation measurement (I₀ or I) to session container.
        
        Args:
            measurement_data: Dict mapping detector_id to 2D array
            detector_metadata: Dict mapping detector_id to metadata dict
            poni_alias_map: Dict mapping detector_alias to detector_id
            mode: "without" for I₀, "with" for I
            
        Returns:
            Measurement counter for this analytical measurement
        """
        self._check_active()
        
        ana_path = self.writer.add_analytical_measurement(
            file_path=self.session_path,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            analysis_type="attenuation",
            analysis_role=(
                self.schema.ANALYSIS_ROLE_I0
                if mode == "without"
                else self.schema.ANALYSIS_ROLE_I
            ),
            timestamp_start=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        # Extract counter from path: /analytical_measurements/ana_000000001 → 1
        counter = int(ana_path.split("_")[-1])
        
        # Store counter
        if mode == "without":
            self.i0_counter = counter
            logger.info("Added I₀ attenuation measurement", counter=counter)
            self.log_event(
                message="Attenuation I0 recorded",
                event_type="attenuation_i0_recorded",
                details={"counter": counter},
            )
        else:  # mode == "with"
            self.i_counter = counter
            logger.info("Added I attenuation measurement", counter=counter)
            self.log_event(
                message="Attenuation I recorded",
                event_type="attenuation_i_recorded",
                details={"counter": counter},
            )
        
        return counter
    
    def link_attenuation_to_points(
        self,
        num_points: int,
        start_point_idx: int = 1,
    ):
        """Link both I₀ and I attenuation measurements to all points.
        
        Args:
            num_points: Total number of points to link
            start_point_idx: First point index to link (1-based)
        """
        self._check_active()
        if start_point_idx < 1:
            raise ValueError(
                f"start_point_idx must be >= 1, got {start_point_idx}"
            )
        
        if self.i0_counter is None or self.i_counter is None:
            raise RuntimeError(
                "Both I₀ and I measurements must be recorded before linking. "
                f"Current state: I₀={self.i0_counter}, I={self.i_counter}"
            )
        
        logger.info(
            "Linking attenuation to points",
            num_points=num_points,
            start_point_idx=start_point_idx,
            i0_counter=self.i0_counter,
            i_counter=self.i_counter,
        )
        
        end_point_idx = start_point_idx + num_points
        for point_idx in range(start_point_idx, end_point_idx):
            # Link I₀
            self.writer.link_analytical_measurement_to_point(
                file_path=self.session_path,
                point_index=point_idx,
                analytical_measurement_index=self.i0_counter,
            )
            
            # Link I
            self.writer.link_analytical_measurement_to_point(
                file_path=self.session_path,
                point_index=point_idx,
                analytical_measurement_index=self.i_counter,
            )
        
        logger.info("Attenuation linked to all points successfully")
        self.log_event(
            message="Linked attenuation analytical measurements to points",
            event_type="attenuation_linked",
            details={
                "start_point_idx": start_point_idx,
                "num_points": num_points,
                "i0_counter": self.i0_counter,
                "i_counter": self.i_counter,
            },
        )
    
    def begin_point_measurement(
        self,
        point_index: int,
        timestamp_start: Optional[str] = None,
    ) -> str:
        """Create an in-progress measurement record before detector capture starts."""
        self._check_active()

        existing = self._pending_measurements.get(point_index)
        if existing:
            return existing

        meas_path = self.writer.begin_measurement(
            file_path=self.session_path,
            point_index=point_index,
            timestamp_start=timestamp_start,
            measurement_status=self.schema.STATUS_IN_PROGRESS,
        )
        self._pending_measurements[point_index] = meas_path
        self.log_event(
            message="Point measurement started",
            event_type="measurement_started",
            details={
                "point_index": point_index,
                "measurement_path": meas_path,
            },
        )
        logger.info("Started point measurement", point_index=point_index, path=meas_path)
        return meas_path

    def complete_point_measurement(
        self,
        point_index: int,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        raw_files: Optional[Dict] = None,
        timestamp_end: Optional[str] = None,
        measurement_status: str = None,
    ) -> str:
        """Finalize point measurement and write detector payload."""
        self._check_active()
        if measurement_status is None:
            measurement_status = self.schema.STATUS_COMPLETED

        meas_path = self._pending_measurements.pop(point_index, None)
        if meas_path:
            meas_path = self.writer.finalize_measurement(
                file_path=self.session_path,
                measurement_path=meas_path,
                measurement_data=measurement_data,
                detector_metadata=detector_metadata,
                poni_alias_map=poni_alias_map,
                raw_files=raw_files,
                timestamp_end=timestamp_end,
                measurement_status=measurement_status,
            )
        else:
            meas_path = self.writer.add_measurement(
                file_path=self.session_path,
                point_index=point_index,
                measurement_data=measurement_data,
                detector_metadata=detector_metadata,
                poni_alias_map=poni_alias_map,
                raw_files=raw_files,
                timestamp_end=timestamp_end,
                measurement_status=measurement_status,
            )

        if measurement_status == self.schema.STATUS_COMPLETED:
            self.writer.update_point_status(
                file_path=self.session_path,
                point_index=point_index,
                point_status="measured",
            )
        self.log_event(
            message="Point measurement finalized",
            event_type="measurement_finalized",
            details={
                "point_index": point_index,
                "measurement_path": meas_path,
                "status": measurement_status,
                "detector_count": len(measurement_data or {}),
            },
        )

        logger.info(
            "Completed point measurement",
            point_index=point_index,
            status=measurement_status,
            path=meas_path,
        )
        return meas_path

    def fail_point_measurement(
        self,
        point_index: int,
        reason: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[str]:
        """Mark an in-progress point measurement as failed/aborted."""
        self._check_active()

        meas_path = self._pending_measurements.pop(point_index, None)
        if not meas_path:
            return None

        terminal_status = status or self.schema.STATUS_FAILED
        self.writer.fail_measurement(
            file_path=self.session_path,
            measurement_path=meas_path,
            failure_reason=reason,
            timestamp_end=timestamp_end,
            measurement_status=terminal_status,
        )
        self.log_event(
            message="Point measurement failed",
            event_type="measurement_failed",
            level="WARNING",
            details={
                "point_index": point_index,
                "measurement_path": meas_path,
                "status": terminal_status,
                "reason": reason or "",
            },
        )
        logger.warning(
            "Point measurement failed",
            point_index=point_index,
            status=terminal_status,
            reason=reason,
            path=meas_path,
        )
        return meas_path

    def add_measurement(
        self,
        point_index: int,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        raw_files: Optional[Dict] = None,
    ) -> str:
        """Backward-compatible wrapper: write completed point measurement."""
        return self.complete_point_measurement(
            point_index=point_index,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            raw_files=raw_files,
        )
    
    def _check_active(self):
        """Check if session is active, raise if not."""
        if not self.is_session_active():
            raise RuntimeError(
                "No active session. Please create a session first using create_session()."
            )
    
    def is_locked(self) -> bool:
        """Check if the current session container is locked.
        
        Returns:
            True if locked, False if unlocked or no active session
        """
        if not self.is_session_active():
            return False
        
        return self.container_manager.is_container_locked(self.session_path)
    
    def update_sample_id(self, new_sample_id: str) -> bool:
        """Update the sample ID in the session container.
        
        Can only update if container is unlocked.
        
        Args:
            new_sample_id: New sample identifier
            
        Returns:
            True if updated successfully, False if locked or failed
        """
        self._check_active()
        
        if self.is_locked():
            logger.warning(
                "Cannot update sample_id: container is locked",
                sample_id=new_sample_id,
            )
            return False
        
        try:
            import h5py
            
            with h5py.File(self.session_path, 'a') as f:
                old_sample_id = f.attrs.get(self.schema.ATTR_SAMPLE_ID, 'unknown')
                f.attrs[self.schema.ATTR_SAMPLE_ID] = new_sample_id
                if self.schema.GROUP_SAMPLE in f:
                    f[self.schema.GROUP_SAMPLE].attrs[self.schema.ATTR_SAMPLE_ID] = new_sample_id

            refresh_summary = getattr(self.writer, "refresh_human_summary", None)
            if callable(refresh_summary):
                refresh_summary(self.session_path)
                
            self.sample_id = new_sample_id
            
            logger.info(
                "Updated sample_id in session container",
                old_sample_id=old_sample_id,
                new_sample_id=new_sample_id,
                session_path=str(self.session_path),
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to update sample_id: {e}",
                exc_info=True,
            )
            return False
    
    def get_session_info(self) -> Dict:
        """Get current session information.
        
        Returns:
            Dict with session metadata
        """
        if not self.is_session_active():
            return {"active": False}
        
        return {
            "active": True,
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "sample_id": self.sample_id,
            "study_name": self.study_name,
            "operator_id": self.operator_id,
            "machine_name": self.machine_name,
            "beam_energy_kev": self.beam_energy_kev,
            "is_locked": self.is_locked(),
            "i0_recorded": self.i0_counter is not None,
            "i_recorded": self.i_counter is not None,
            "attenuation_complete": (
                self.i0_counter is not None and self.i_counter is not None
            ),
        }
