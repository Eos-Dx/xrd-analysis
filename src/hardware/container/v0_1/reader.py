"""HDF5 Container Reader for v0.1.

Provides classes for reading session and technical containers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np

from . import schema, utils


class BaseContainer:
    """Base class for HDF5 containers."""
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize container reader.
        
        Args:
            file_path: Path to HDF5 container file
        """
        self.file_path = str(file_path)
        self._validate_file()
    
    def _validate_file(self):
        """Validate that file exists and is readable HDF5."""
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"Container not found: {self.file_path}")
        
        if not utils.verify_container_readable(self.file_path):
            raise ValueError(f"Cannot read HDF5 file: {self.file_path}")
    
    def get_metadata(self) -> Dict:
        """Get container metadata (root attributes).
        
        Returns:
            Dict of metadata
        """
        return utils.get_container_info(self.file_path)
    
    @classmethod
    def open(cls, file_path: Union[str, Path], validate: bool = True):
        """Open and optionally validate container.
        
        Args:
            file_path: Path to container
            validate: Whether to validate structure
            
        Returns:
            Container instance
        """
        container = cls(file_path)
        if validate:
            container.validate()
        return container
    
    def validate(self):
        """Validate container structure. Override in subclasses."""
        pass


class SessionContainer(BaseContainer):
    """Reader for session HDF5 containers."""
    
    def get_points(self) -> Dict:
        """Get all measurement points.
        
        Returns:
            Dict mapping point_id to point data
        """
        with utils.open_h5_readonly(self.file_path) as f:
            if schema.GROUP_POINTS not in f:
                return {}
            
            points = {}
            points_group = f[schema.GROUP_POINTS]
            for pt_name in points_group.keys():
                pt = points_group[pt_name]
                points[pt_name] = {
                    "pixel_coordinates": pt.attrs.get(schema.ATTR_PIXEL_COORDINATES, []),
                    "physical_coordinates_mm": pt.attrs.get(schema.ATTR_PHYSICAL_COORDINATES_MM, []),
                    "point_status": pt.attrs.get(schema.ATTR_POINT_STATUS, "unknown"),
                }
            return points
    
    def get_measurements(self, point_index: Optional[int] = None) -> Dict:
        """Get measurements, optionally filtered by point.
        
        Args:
            point_index: Optional point index to filter
            
        Returns:
            Dict of measurements
        """
        with utils.open_h5_readonly(self.file_path) as f:
            if schema.GROUP_MEASUREMENTS not in f:
                return {}
            
            meas_group = f[schema.GROUP_MEASUREMENTS]
            measurements = {}
            
            for pt_name in meas_group.keys():
                if point_index is not None:
                    expected_pt = schema.format_point_id(point_index)
                    if pt_name != expected_pt:
                        continue
                
                pt_meas = meas_group[pt_name]
                measurements[pt_name] = {}
                
                for meas_name in pt_meas.keys():
                    meas = pt_meas[meas_name]
                    measurements[pt_name][meas_name] = {
                        "measurement_counter": meas.attrs.get(schema.ATTR_MEASUREMENT_COUNTER, 0),
                        "timestamp_start": meas.attrs.get(schema.ATTR_TIMESTAMP_START, ""),
                        "measurement_status": meas.attrs.get(schema.ATTR_MEASUREMENT_STATUS, ""),
                        "detectors": list(meas.keys()),
                    }
            
            return measurements
    
    def get_detector_data(self, point_index: int, measurement_counter: int, detector_id: str) -> Optional[np.ndarray]:
        """Get processed detector data for a specific measurement.
        
        Args:
            point_index: Point index
            measurement_counter: Measurement counter
            detector_id: Detector ID
            
        Returns:
            Processed signal array or None
        """
        pt_name = schema.format_point_id(point_index)
        meas_name = schema.format_measurement_id(measurement_counter)
        det_role = schema.format_detector_role(detector_id)
        
        signal_path = f"{schema.GROUP_MEASUREMENTS}/{pt_name}/{meas_name}/{det_role}/{schema.DATASET_PROCESSED_SIGNAL}"
        
        with utils.open_h5_readonly(self.file_path) as f:
            if signal_path in f:
                return f[signal_path][:]
        return None
    
    def get_images(self) -> Dict:
        """Get all sample images.
        
        Returns:
            Dict mapping image_id to image data and metadata
        """
        with utils.open_h5_readonly(self.file_path) as f:
            if schema.GROUP_IMAGES not in f:
                return {}
            
            images = {}
            img_group = f[schema.GROUP_IMAGES]
            for img_name in img_group.keys():
                if not img_name.startswith("img_"):
                    continue
                img = img_group[img_name]
                images[img_name] = {
                    "image_type": img.attrs.get(schema.ATTR_IMAGE_TYPE, ""),
                    "timestamp": img.attrs.get(schema.ATTR_TIMESTAMP, ""),
                    "data": img["data"][:] if "data" in img else None,
                }
            return images
    
    def validate(self):
        """Validate session container structure."""
        from . import validator
        is_valid, errors = validator.SessionContainerValidator(self.file_path).validate()
        if not is_valid:
            error_msgs = [e.message for e in errors if e.severity == "ERROR"]
            raise ValueError(f"Container validation failed: {error_msgs}")


class TechnicalContainer(BaseContainer):
    """Reader for technical HDF5 containers."""
    
    def get_technical_measurements(self) -> Dict:
        """Get all technical measurement events.
        
        Returns:
            Dict of technical measurements
        """
        with utils.open_h5_readonly(self.file_path) as f:
            if schema.GROUP_TECHNICAL not in f:
                return {}
            
            tech_group = f[schema.GROUP_TECHNICAL]
            measurements = {}
            
            for evt_name in tech_group.keys():
                if not evt_name.startswith("tech_evt_"):
                    continue
                evt = tech_group[evt_name]
                measurements[evt_name] = {
                    "type": evt.attrs.get("type", ""),
                    "timestamp": evt.attrs.get(schema.ATTR_TIMESTAMP, ""),
                    "detectors": list(evt.keys()),
                }
            
            return measurements
    
    def get_poni_data(self) -> Dict[str, str]:
        """Get PONI calibration data.
        
        Returns:
            Dict mapping detector dataset name to PONI content
        """
        with utils.open_h5_readonly(self.file_path) as f:
            if schema.GROUP_TECHNICAL_PONI not in f:
                return {}
            
            poni_group = f[schema.GROUP_TECHNICAL_PONI]
            poni_data: Dict[str, str] = {}
            for dataset_name in poni_group.keys():
                poni_data[dataset_name] = poni_group[dataset_name][()]
            return poni_data
