"""Session Manager for DIFRA GUI.

Handles HDF5 session container lifecycle:
- Creating new sessions
- Managing active session state
- Writing measurements to containers
- Tracking measurement counters
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from hardware.container.v0_1 import writer, schema
from hardware.container.v0_1.container_manager import (
    find_active_technical_container,
    is_container_locked,
)
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionManager:
    """Manages HDF5 session containers for DIFRA measurements."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize SessionManager.
        
        Args:
            config: Optional configuration dict from global.json
                   If provided, beam_energy_kev will be read from config
        """
        self.session_path: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.sample_id: Optional[str] = None
        
        # Track counters for linking
        self.i0_counter: Optional[int] = None  # Attenuation without sample
        self.i_counter: Optional[int] = None   # Attenuation with sample
        
        # Store config for later use
        self.config = config or {}
        
        # Configuration - read from config or use defaults
        if config:
            self.operator_id: str = config.get('operator_id', 'operator')
            self.site_id: str = config.get('site_id', 'DIFRA_LAB')
            self.machine_name: str = config.get('machine_name', 'DIFRA-01')
            self.beam_energy_kev: float = config.get('beam_energy_kev', 17.5)
        else:
            self.operator_id: str = "operator"
            self.site_id: str = "DIFRA_LAB"
            self.machine_name: str = "DIFRA-01"
            self.beam_energy_kev: float = 17.5
    
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
        These will be passed to the container writer and validated against schema.
        
        Required session attributes (from schema):
            sample_id: str - Unique sample identifier
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
            schema.ATTR_SAMPLE_ID: session_attrs.get(
                schema.ATTR_SAMPLE_ID,
                session_attrs.get('sample_id'),  # Support both snake_case and schema names
            ),
            schema.ATTR_OPERATOR_ID: session_attrs.get(
                schema.ATTR_OPERATOR_ID,
                session_attrs.get('operator_id', self.operator_id),
            ),
            schema.ATTR_SITE_ID: session_attrs.get(
                schema.ATTR_SITE_ID,
                session_attrs.get('site_id', self.site_id),
            ),
            schema.ATTR_MACHINE_NAME: session_attrs.get(
                schema.ATTR_MACHINE_NAME,
                session_attrs.get('machine_name', self.machine_name),
            ),
            schema.ATTR_BEAM_ENERGY_KEV: session_attrs.get(
                schema.ATTR_BEAM_ENERGY_KEV,
                session_attrs.get('beam_energy_keV', self.beam_energy_kev),
            ),
            schema.ATTR_ACQUISITION_DATE: session_attrs.get(
                schema.ATTR_ACQUISITION_DATE,
                session_attrs.get('acquisition_date', datetime.now().strftime("%Y-%m-%d")),
            ),
        }
        
        # Add optional attributes if provided
        if schema.ATTR_PATIENT_ID in session_attrs or 'patient_id' in session_attrs:
            container_attrs[schema.ATTR_PATIENT_ID] = session_attrs.get(
                schema.ATTR_PATIENT_ID,
                session_attrs.get('patient_id'),
            )
        
        # Validate required sample_id
        if not container_attrs[schema.ATTR_SAMPLE_ID]:
            raise ValueError("sample_id is required to create a session")
        
        sample_id = container_attrs[schema.ATTR_SAMPLE_ID]
        
        logger.info(
            "Creating new session",
            sample_id=sample_id,
            distance_cm=distance_cm,
            technical_container=str(tech_path),
            operator_id=container_attrs.get(schema.ATTR_OPERATOR_ID),
            site_id=container_attrs.get(schema.ATTR_SITE_ID),
            machine_name=container_attrs.get(schema.ATTR_MACHINE_NAME),
        )
        
        # Create session container with schema-driven attributes
        self.session_id, session_path_str = writer.create_session_container(
            folder=folder,
            **container_attrs,
        )
        
        self.session_path = Path(session_path_str)
        self.sample_id = sample_id
        
        # Copy technical data to session
        writer.copy_technical_to_session(
            technical_file=tech_path,
            session_file=self.session_path,
        )
        
        logger.info(
            "Session created successfully",
            session_id=self.session_id,
            session_path=str(self.session_path),
        )
        
        # Reset counters
        self.i0_counter = None
        self.i_counter = None
        
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
        self.i0_counter = None
        self.i_counter = None
    
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
        
        return writer.add_image(
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
        
        return writer.add_zone(
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
                
        Returns:
            List of point group paths
        """
        self._check_active()
        
        paths = []
        for idx, point in enumerate(points, start=1):
            path = writer.add_point(
                file_path=self.session_path,
                point_index=idx,
                pixel_coordinates=point["pixel_coordinates"],
                physical_coordinates_mm=point["physical_coordinates_mm"],
                point_status=point.get("point_status", "pending"),
            )
            paths.append(path)
        
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
        
        ana_path = writer.add_analytical_measurement(
            file_path=self.session_path,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            analysis_type="attenuation",
            timestamp_start=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        # Extract counter from path: /analytical_measurements/ana_000000001 → 1
        counter = int(ana_path.split("_")[-1])
        
        # Store counter
        if mode == "without":
            self.i0_counter = counter
            logger.info("Added I₀ attenuation measurement", counter=counter)
        else:  # mode == "with"
            self.i_counter = counter
            logger.info("Added I attenuation measurement", counter=counter)
        
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
            writer.link_analytical_measurement_to_point(
                file_path=self.session_path,
                point_index=point_idx,
                analytical_measurement_index=self.i0_counter,
            )
            
            # Link I
            writer.link_analytical_measurement_to_point(
                file_path=self.session_path,
                point_index=point_idx,
                analytical_measurement_index=self.i_counter,
            )
        
        logger.info("Attenuation linked to all points successfully")
    
    def add_measurement(
        self,
        point_index: int,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        raw_files: Optional[Dict] = None,
    ) -> str:
        """Add regular measurement at a point.
        
        Args:
            point_index: Point index (1-based)
            measurement_data: Dict mapping detector_id to 2D array
            detector_metadata: Dict mapping detector_id to metadata dict
            poni_alias_map: Dict mapping detector_alias to detector_id
            raw_files: Optional dict of {detector_id: {"file.txt": bytes, "file.dsc": bytes}}
            
        Returns:
            Measurement group path
        """
        self._check_active()
        
        meas_path = writer.add_measurement(
            file_path=self.session_path,
            point_index=point_index,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            raw_files=raw_files,
        )
        
        # Update point status to measured
        writer.update_point_status(
            file_path=self.session_path,
            point_index=point_index,
            point_status="measured",
        )
        
        logger.info("Added measurement", point_index=point_index, path=meas_path)
        return meas_path
    
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
        
        return is_container_locked(self.session_path)
    
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
                old_sample_id = f.attrs.get(schema.ATTR_SAMPLE_ID, 'unknown')
                f.attrs[schema.ATTR_SAMPLE_ID] = new_sample_id
                
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
