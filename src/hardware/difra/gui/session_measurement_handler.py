"""Session Measurement Handler

Integrates session container writing into the DIFRA GUI measurement workflow.
Handles:
- Session container creation on measurement start
- Copying technical data from technical container
- Writing measurements during acquisition
- Updating point status
- Managing measurement counter atomically
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from hardware.container.v0_1 import schema, session_container, measurement_counter

logger = logging.getLogger(__name__)


class SessionMeasurementHandler:
    """Manages session container operations during GUI measurements."""

    def __init__(
        self,
        session_folder: Union[str, Path],
        technical_container_file: Union[str, Path],
        **session_attrs,
    ):
        """Initialize session measurement handler.

        All session attributes should be provided as keyword arguments.
        These will be validated against the schema when creating the session.
        
        Required session attributes (from schema):
            sample_id: str - Sample identifier
            operator_id: str - Operator ID/name
            site_id: str - Site identifier
            machine_name: str - Acquisition machine name
            beam_energy_keV: float - Beam energy in keV
            acquisition_date: str - Acquisition date (auto-generated if not provided)
            
        Optional attributes:
            patient_id: str - Patient identifier

        Args:
            session_folder: Directory where session container will be created
            technical_container_file: Path to technical container with calibration data
            **session_attrs: All session attributes as keyword arguments
        """
        self.session_folder = Path(session_folder)
        self.technical_container_file = Path(technical_container_file)
        
        # Store session attributes for later use
        self.session_attrs = session_attrs.copy()
        
        # Auto-generate acquisition_date if not provided
        if schema.ATTR_ACQUISITION_DATE not in self.session_attrs and 'acquisition_date' not in self.session_attrs:
            self.session_attrs['acquisition_date'] = time.strftime("%Y-%m-%d")
        
        self.session_file = None
        self.session_id = None

    def create_session(self) -> str:
        """Create a new session container.

        Copies technical data from technical container on creation.

        Returns:
            Path to session container
        """
        if not self.technical_container_file.exists():
            raise FileNotFoundError(
                f"Technical container not found: {self.technical_container_file}"
            )

        self.session_folder.mkdir(parents=True, exist_ok=True)

        # Create session container with explicit required attributes
        sample_id = self.session_attrs.get(schema.ATTR_SAMPLE_ID, self.session_attrs.get("sample_id"))
        operator_id = self.session_attrs.get(schema.ATTR_OPERATOR_ID, self.session_attrs.get("operator_id"))
        site_id = self.session_attrs.get(schema.ATTR_SITE_ID, self.session_attrs.get("site_id"))
        machine_name = self.session_attrs.get(schema.ATTR_MACHINE_NAME, self.session_attrs.get("machine_name"))
        beam_energy_keV = self.session_attrs.get(schema.ATTR_BEAM_ENERGY_KEV, self.session_attrs.get("beam_energy_keV"))
        acquisition_date = self.session_attrs.get(schema.ATTR_ACQUISITION_DATE, self.session_attrs.get("acquisition_date"))
        patient_id = self.session_attrs.get(schema.ATTR_PATIENT_ID, self.session_attrs.get("patient_id"))
        container_id = self.session_attrs.get(schema.ATTR_SESSION_ID, self.session_attrs.get("session_id"))

        self.session_id, self.session_file = session_container.create_session_container(
            folder=self.session_folder,
            sample_id=sample_id,
            operator_id=operator_id,
            site_id=site_id,
            machine_name=machine_name,
            beam_energy_keV=beam_energy_keV,
            acquisition_date=acquisition_date,
            patient_id=patient_id,
            container_id=container_id,
        )
        self.sample_id = sample_id

        # Copy technical data
        session_container.copy_technical_to_session(
            technical_file=self.technical_container_file,
            session_file=self.session_file,
        )

        # Initialize counter handler
        self.counter = measurement_counter.MeasurementCounter(self.session_file)

        logger.info(
            f"Session container created",
            session_id=self.session_id,
            sample_id=self.sample_id,
            file=self.session_file,
        )

        return self.session_file

    def add_image(
        self,
        image_data: Union[np.ndarray, str],
        image_index: int = 1,
        image_type: str = schema.IMAGE_TYPE_SAMPLE,
    ) -> str:
        """Add sample image to session.

        Args:
            image_data: Image array or path to image file
            image_index: Image index (1-based)
            image_type: Type of image

        Returns:
            Image group path
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        return session_container.add_image(
            file_path=self.session_file,
            image_index=image_index,
            image_data=image_data,
            image_type=image_type,
        )

    def add_zone(
        self,
        zone_index: int,
        zone_role: str,
        geometry_px: Union[List, np.ndarray, str],
        shape: str = "polygon",
        holder_diameter_mm: Optional[float] = None,
    ) -> str:
        """Add zone to session.

        Args:
            zone_index: Zone index (1-based)
            zone_role: Role of zone (sample_holder, include, exclude)
            geometry_px: Pixel coordinates
            shape: Shape type
            holder_diameter_mm: Optional holder diameter

        Returns:
            Zone group path
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        return session_container.add_zone(
            file_path=self.session_file,
            zone_index=zone_index,
            zone_role=zone_role,
            geometry_px=geometry_px,
            shape=shape,
            holder_diameter_mm=holder_diameter_mm,
        )

    def add_image_mapping(
        self,
        sample_holder_zone_id: str,
        pixel_to_mm_conversion: Dict,
        orientation: str = "standard",
    ) -> str:
        """Add pixel-to-mm mapping.

        Args:
            sample_holder_zone_id: ID of sample holder zone
            pixel_to_mm_conversion: Conversion parameters dict
            orientation: Orientation label

        Returns:
            Mapping dataset path
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        return session_container.add_image_mapping(
            file_path=self.session_file,
            sample_holder_zone_id=sample_holder_zone_id,
            pixel_to_mm_conversion=pixel_to_mm_conversion,
            orientation=orientation,
        )

    def add_point(
        self,
        point_index: int,
        pixel_coordinates: List[float],
        physical_coordinates_mm: List[float],
        point_status: str = schema.POINT_STATUS_PENDING,
    ) -> str:
        """Add measurement point.

        Args:
            point_index: Point index (1-based)
            pixel_coordinates: [x_px, y_px]
            physical_coordinates_mm: [x_mm, y_mm]
            point_status: Point status

        Returns:
            Point group path
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        return session_container.add_point(
            file_path=self.session_file,
            point_index=point_index,
            pixel_coordinates=pixel_coordinates,
            physical_coordinates_mm=physical_coordinates_mm,
            point_status=point_status,
        )

    def update_point_status(
        self, point_index: int, point_status: str
    ) -> None:
        """Update point status.

        Args:
            point_index: Point index (1-based)
            point_status: New status
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        session_container.update_point_status(
            file_path=self.session_file,
            point_index=point_index,
            point_status=point_status,
        )

    def add_measurement(
        self,
        point_index: int,
        measurement_data: Dict[str, np.ndarray],
        detector_metadata: Dict[str, Dict],
        pony_alias_map: Dict[str, str],
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        measurement_status: str = schema.STATUS_COMPLETED,
    ) -> str:
        """Add measurement to point.

        Args:
            point_index: Point index (1-based)
            measurement_data: Dict mapping detector_id to processed signal array
array
            detector_metadata: Dict mapping detector_id to metadata
            pony_alias_map: Dict mapping alias to detector_id
            timestamp_start: Start timestamp
            timestamp_end: End timestamp
            measurement_status: Measurement status

        Returns:
            Measurement group path
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        return session_container.add_measurement(
            file_path=self.session_file,
            point_index=point_index,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            measurement_status=measurement_status,
        )

    def add_analytical_measurement(
        self,
        measurement_data: Dict[str, np.ndarray],
        detector_metadata: Dict[str, Dict],
        pony_alias_map: Dict[str, str],
        analysis_type: str,
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
    ) -> str:
        """Add analytical measurement.

        Args:
            measurement_data: Dict mapping detector_id to raw signal array
            detector_metadata: Dict mapping detector_id to metadata
            pony_alias_map: Dict mapping alias to detector_id
            analysis_type: Type of analysis (e.g., "attenuation")
            timestamp_start: Start timestamp
            timestamp_end: End timestamp

        Returns:
            Analytical measurement group path
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        return session_container.add_analytical_measurement(
            file_path=self.session_file,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type=analysis_type,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
        )

    def link_analytical_measurement_to_point(
        self, point_index: int, analytical_measurement_index: int
    ) -> None:
        """Link analytical measurement to a point.

        Args:
            point_index: Point index (1-based)
            analytical_measurement_index: Analytical measurement counter
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")

        session_container.link_analytical_measurement_to_point(
            file_path=self.session_file,
            point_index=point_index,
            analytical_measurement_index=analytical_measurement_index,
        )

    def get_current_counter(self) -> int:
        """Get current measurement counter value.

        Returns:
            Current counter value
        """
        if self.counter is None:
            raise RuntimeError("Session counter not initialized")
        return self.counter.get_current()

    def get_session_file(self) -> str:
        """Get path to session container.

        Returns:
            Path to session file
        """
        if self.session_file is None:
            raise RuntimeError("Session container not created")
        return self.session_file

    def get_session_id(self) -> str:
        """Get session ID.

        Returns:
            Session container ID
        """
        if self.session_id is None:
            raise RuntimeError("Session container not created")
        return self.session_id
