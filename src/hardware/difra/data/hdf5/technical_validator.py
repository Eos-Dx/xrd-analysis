"""Technical HDF5 Container Schema Validator v1.0

Validates technical containers against DIFRA HDF5 Data Model v1.0 specification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py

from . import schema_v1
from . import io

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when HDF5 container fails validation."""
    pass


class TechnicalContainerValidator:
    """Validator for technical HDF5 containers following schema v1.0."""
    
    def __init__(self, file_path: str, strict: bool = True):
        """Initialize validator.
        
        Args:
            file_path: Path to HDF5 container
            strict: If True, raise on first error; if False, collect all errors
        """
        self.file_path = Path(file_path)
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def _add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        if self.strict:
            raise ValidationError(message)
    
    def _add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validate the container.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors.clear()
        self.warnings.clear()
        
        if not self.file_path.exists():
            self._add_error(f"File not found: {self.file_path}")
            return False, self.errors, self.warnings
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                self._validate_root_attributes(f)
                self._validate_technical_group(f)
                self._validate_config_group(f)
                self._validate_pony_group(f)
                self._validate_technical_events(f)
        except Exception as e:
            self._add_error(f"Unexpected error during validation: {e}")
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_root_attributes(self, f: h5py.File):
        """Validate root-level attributes."""
        required = [
            schema_v1.ATTR_CONTAINER_ID,
            schema_v1.ATTR_CONTAINER_TYPE,
            schema_v1.ATTR_SCHEMA_VERSION,
            schema_v1.ATTR_CREATION_TIMESTAMP,
            schema_v1.ATTR_DISTANCE_CM,
        ]
        
        for attr_name in required:
            if attr_name not in f.attrs:
                self._add_error(f"Missing required root attribute: {attr_name}")
        
        # Validate container type
        if schema_v1.ATTR_CONTAINER_TYPE in f.attrs:
            container_type = f.attrs[schema_v1.ATTR_CONTAINER_TYPE]
            if container_type != schema_v1.CONTAINER_TYPE_TECHNICAL:
                self._add_error(
                    f"Invalid container_type: {container_type}, "
                    f"expected: {schema_v1.CONTAINER_TYPE_TECHNICAL}"
                )
        
        # Validate schema version
        if schema_v1.ATTR_SCHEMA_VERSION in f.attrs:
            version = f.attrs[schema_v1.ATTR_SCHEMA_VERSION]
            if version != schema_v1.SCHEMA_VERSION:
                self._add_warning(
                    f"Schema version mismatch: {version}, expected: {schema_v1.SCHEMA_VERSION}"
                )
        
        # Validate container ID format
        if schema_v1.ATTR_CONTAINER_ID in f.attrs:
            container_id = f.attrs[schema_v1.ATTR_CONTAINER_ID]
            if not schema_v1.validate_container_id(container_id):
                self._add_error(f"Invalid container_id format: {container_id}")
    
    def _validate_technical_group(self, f: h5py.File):
        """Validate /technical group exists."""
        if schema_v1.GROUP_TECHNICAL not in f:
            self._add_error(f"Missing required group: {schema_v1.GROUP_TECHNICAL}")
    
    def _validate_config_group(self, f: h5py.File):
        """Validate /technical/config group and detector_config dataset."""
        config_path = schema_v1.GROUP_TECHNICAL_CONFIG
        if config_path not in f:
            self._add_error(f"Missing required group: {config_path}")
            return
        
        detector_config_path = f"{config_path}/detector_config"
        if detector_config_path not in f:
            self._add_error(f"Missing detector_config dataset: {detector_config_path}")
    
    def _validate_pony_group(self, f: h5py.File):
        """Validate /technical/pony group."""
        pony_path = schema_v1.GROUP_TECHNICAL_PONY
        if pony_path not in f:
            self._add_warning(f"Missing PONY group: {pony_path} (optional but recommended)")
            return
        
        # Check for at least one PONY dataset
        pony_group = f[pony_path]
        pony_datasets = [k for k in pony_group.keys() if k.startswith('pony_')]
        if not pony_datasets:
            self._add_warning("PONY group exists but contains no pony_* datasets")
    
    def _validate_technical_events(self, f: h5py.File):
        """Validate technical events and ensure all required types are present."""
        tech_group = f.get(schema_v1.GROUP_TECHNICAL)
        if not tech_group:
            return  # Already reported as error
        
        # Collect all technical events
        event_ids = [k for k in tech_group.keys() if k.startswith('tech_evt_')]
        
        if not event_ids:
            self._add_error("No technical events found (tech_evt_###)")
            return
        
        # Track which required types are present
        found_types: Set[str] = set()
        
        for event_id in event_ids:
            event_path = f"{schema_v1.GROUP_TECHNICAL}/{event_id}"
            event_group = f[event_path]
            
            # Check required event attributes
            if 'type' not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'type' attribute")
                continue
            
            event_type = event_group.attrs['type']
            found_types.add(event_type)
            
            # Validate type is in allowed set
            if not schema_v1.validate_technical_type(event_type):
                self._add_error(
                    f"{event_path}: Invalid type '{event_type}', "
                    f"must be one of {schema_v1.ALL_TECHNICAL_TYPES}"
                )
            
            # Check for timestamp
            if 'timestamp_utc' not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'timestamp_utc' attribute")
            
            # Check for distance
            if schema_v1.ATTR_DISTANCE_CM not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'distance_cm' attribute")
            
            # Validate detector subgroups
            detector_groups = [k for k in event_group.keys() if k.startswith('det_')]
            if not detector_groups:
                self._add_error(f"{event_path}: No detector subgroups found (det_*)")
            
            for det_group_name in detector_groups:
                self._validate_detector_group(f, event_path, det_group_name, event_type)
        
        # Check that all REQUIRED types are present
        missing_types = set(schema_v1.REQUIRED_TECHNICAL_TYPES) - found_types
        if missing_types:
            self._add_error(
                f"Missing required technical measurement types: {sorted(missing_types)}. "
                f"Required: {schema_v1.REQUIRED_TECHNICAL_TYPES}"
            )
    
    def _validate_detector_group(
        self, f: h5py.File, event_path: str, det_group_name: str, event_type: str
    ):
        """Validate detector subgroup within a technical event."""
        det_path = f"{event_path}/{det_group_name}"
        det_group = f[det_path]
        
        # Check required datasets
        if schema_v1.DATASET_RAW_SIGNAL not in det_group:
            self._add_error(f"{det_path}: Missing 'raw_signal' dataset")
        
        # Check for raw data storage (optional but recommended)
        # New schema: blob/ group with raw_txt, raw_dsc, etc.
        # Legacy schema: raw_blob_txt, raw_blob_dsc, or raw_blob datasets
        has_blob_group = 'blob' in det_group
        has_legacy_blobs = any(k.startswith('raw_blob') for k in det_group.keys())
        
        if not has_blob_group and not has_legacy_blobs:
            self._add_warning(f"{det_path}: No raw data blobs found (blob/ group or raw_blob_* datasets) - optional")
        
        # Check required attributes
        required_attrs = [
            schema_v1.ATTR_TECHNICAL_TYPE,
            schema_v1.ATTR_DISTANCE_CM,
            schema_v1.ATTR_TIMESTAMP,
            schema_v1.ATTR_DETECTOR_ID,
        ]
        for attr_name in required_attrs:
            if attr_name not in det_group.attrs:
                self._add_error(f"{det_path}: Missing attribute '{attr_name}'")
        
        # For AGBH measurements, check for PONI reference
        if event_type == schema_v1.TECHNICAL_TYPE_AGBH:
            if 'poni_ref' not in det_group.attrs:
                self._add_warning(
                    f"{det_path}: AGBH measurement missing 'poni_ref' attribute (recommended)"
                )


def validate_technical_container(
    file_path: str, strict: bool = False
) -> Tuple[bool, List[str], List[str]]:
    """Validate a technical HDF5 container.
    
    Args:
        file_path: Path to HDF5 file
        strict: If True, stop on first error
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = TechnicalContainerValidator(file_path, strict=strict)
    return validator.validate()


def print_validation_report(
    file_path: str, is_valid: bool, errors: List[str], warnings: List[str]
):
    """Print a formatted validation report."""
    print(f"\n{'='*70}")
    print(f"Technical Container Validation Report")
    print(f"{'='*70}")
    print(f"File: {file_path}")
    print(f"Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"{'='*70}\n")
    
    if errors:
        print(f"❌ Errors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print()
    
    if warnings:
        print(f"⚠️  Warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print()
    
    if not errors and not warnings:
        print("✅ No issues found")
    
    print(f"{'='*70}\n")
