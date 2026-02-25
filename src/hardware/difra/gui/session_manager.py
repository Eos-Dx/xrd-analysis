"""Session Manager for DIFRA GUI.

Handles HDF5 session container lifecycle:
- Creating new sessions
- Managing active session state
- Writing measurements to containers
- Tracking measurement counters
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
from hardware.difra.gui.container_api import get_container_module
from hardware.difra.gui.session_manager_measurement_ops_mixin import (
    SessionManagerMeasurementOpsMixin,
)
from hardware.difra.gui.session_manager_recovery_mixin import (
    SessionManagerRecoveryMixin,
)
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionManager(SessionManagerRecoveryMixin, SessionManagerMeasurementOpsMixin):
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

    @staticmethod
    def _safe_int(value):
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _counter_from_measurement_name(name: str):
        try:
            return int(str(name).split("_")[-1])
        except Exception:
            return None

    def _restore_attenuation_counters_from_h5(self, h5f) -> None:
        """Restore attenuation counters from analytical measurements in an existing session."""
        self.i0_counter = None
        self.i_counter = None

        ana_group_path = getattr(
            self.schema, "GROUP_ANALYTICAL_MEASUREMENTS", "/analytical_measurements"
        )
        ana_group = h5f.get(ana_group_path)
        if ana_group is None:
            return

        role_attr_name = getattr(self.schema, "ATTR_ANALYSIS_ROLE", "analysis_role")
        type_attr_name = getattr(self.schema, "ATTR_ANALYSIS_TYPE", "analysis_type")
        counter_attr_name = getattr(
            self.schema, "ATTR_MEASUREMENT_COUNTER", "measurement_counter"
        )
        attenuation_type = str(
            getattr(self.schema, "ANALYSIS_TYPE_ATTENUATION", "attenuation")
        ).strip().lower()
        role_i0 = str(getattr(self.schema, "ANALYSIS_ROLE_I0", "i0")).strip().lower()
        role_i = str(getattr(self.schema, "ANALYSIS_ROLE_I", "i")).strip().lower()

        i0_candidates = []
        i_candidates = []

        for ana_id in sorted(ana_group.keys()):
            ana_group_item = ana_group[ana_id]
            counter = self._safe_int(ana_group_item.attrs.get(counter_attr_name))
            if counter is None:
                counter = self._counter_from_measurement_name(str(ana_id))
            if counter is None:
                continue

            analysis_type = self._as_text(
                ana_group_item.attrs.get(type_attr_name), ""
            ).strip().lower()
            analysis_role = self._as_text(
                ana_group_item.attrs.get(role_attr_name), ""
            ).strip().lower()

            is_attenuation = (
                analysis_type == attenuation_type
                or analysis_type.startswith("attenuation")
                or analysis_role in {role_i0, role_i}
            )
            if not is_attenuation:
                continue

            is_i0 = (
                analysis_role in {role_i0, "without", "without_sample"}
                or analysis_type
                in {
                    "attenuation_i0",
                    "attenuation_without",
                    "attenuation_without_sample",
                }
            )
            is_i = (
                analysis_role in {role_i, "with", "with_sample"}
                or analysis_type
                in {
                    "attenuation_i",
                    "attenuation_with",
                    "attenuation_with_sample",
                }
            )

            if is_i0:
                i0_candidates.append(counter)
                continue
            if is_i:
                i_candidates.append(counter)
                continue

            # Legacy fallback (no explicit role): first attenuation is I0, later ones are I.
            if not i0_candidates:
                i0_candidates.append(counter)
            else:
                i_candidates.append(counter)

        if i0_candidates:
            self.i0_counter = max(i0_candidates)
        if i_candidates:
            self.i_counter = max(i_candidates)

        if self.i0_counter is not None or self.i_counter is not None:
            logger.info(
                "Restored attenuation counters from existing session",
                session_path=str(self.session_path),
                i0_counter=self.i0_counter,
                i_counter=self.i_counter,
            )
    
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

    def _find_locked_technical_container_for_distance(
        self,
        folder: Path,
        distance_cm: float,
        tolerance_cm: float = 0.5,
    ) -> Optional[Path]:
        """Find newest locked technical container matching distance."""
        folder = Path(folder)
        if not folder.exists():
            return None

        candidates = []
        seen = set()
        for pattern in ("technical_*.nxs.h5", "technical_*.h5"):
            for tech_path in folder.glob(pattern):
                if "archive" in tech_path.parts:
                    continue
                if not tech_path.is_file():
                    continue

                key = str(tech_path.resolve())
                if key in seen:
                    continue
                seen.add(key)

                try:
                    with h5py.File(tech_path, "r") as h5f:
                        file_distance = float(h5f.attrs.get("distance_cm", float("nan")))
                except Exception:
                    continue

                if abs(file_distance - float(distance_cm)) > float(tolerance_cm):
                    continue
                if not self.container_manager.is_container_locked(tech_path):
                    continue

                candidates.append(tech_path)

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    
    def create_session(
        self,
        folder: Path,
        distance_cm: float,
        technical_container_path: Optional[str] = None,
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
            technical_container_path: Optional explicit technical container path
                (preferred when GUI has an active selected container)
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
        
        # Prefer explicit active technical container from UI when provided.
        explicit_tech_path = str(technical_container_path or "").strip()
        if explicit_tech_path:
            tech_path = Path(explicit_tech_path)
            if not tech_path.exists():
                raise RuntimeError(
                    f"Selected technical container was not found: {tech_path}\n"
                    "Please load or create a technical container and try again."
                )
        else:
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

        # If distance-based lookup returned an unlocked container while a locked one
        # exists for the same distance, automatically pick the locked candidate.
        if not explicit_tech_path and not is_container_locked(tech_path):
            locked_match = self._find_locked_technical_container_for_distance(
                folder=technical_folder,
                distance_cm=distance_cm,
            )
            if locked_match is not None and Path(locked_match) != Path(tech_path):
                logger.warning(
                    "Distance lookup returned unlocked technical container; using locked match instead",
                    requested_distance_cm=float(distance_cm),
                    unlocked_container=str(tech_path),
                    locked_container=str(locked_match),
                )
                tech_path = locked_match

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

            self._restore_attenuation_counters_from_h5(f)

        # Rebuild pending map from in-progress measurements for crash recovery.
        incomplete = self._load_incomplete_measurements_from_container(session_file)
        self._pending_measurements = {
            item["point_index"]: item["measurement_path"] for item in incomplete
        }

        return self.get_session_info()

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
