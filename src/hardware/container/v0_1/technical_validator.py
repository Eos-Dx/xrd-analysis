"""Technical HDF5 Container Schema Validator for container version v0_1."""

import logging
from pathlib import Path
from typing import List, Set, Tuple

import h5py

from . import schema

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when HDF5 container fails validation."""


class TechnicalContainerValidator:
    """Validator for technical HDF5 containers following schema v0_1."""

    def __init__(self, file_path: str, strict: bool = True):
        self.file_path = Path(file_path)
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def _add_error(self, message: str) -> None:
        self.errors.append(message)
        if self.strict:
            raise ValidationError(message)

    def _add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        self.errors.clear()
        self.warnings.clear()

        if not self.file_path.exists():
            self._add_error(f"File not found: {self.file_path}")
            return False, self.errors, self.warnings

        try:
            with h5py.File(self.file_path, "r") as file_handle:
                self._validate_root_attributes(file_handle)
                self._validate_technical_group(file_handle)
                self._validate_config_group(file_handle)
                self._validate_poni_group(file_handle)
                self._validate_technical_events(file_handle)
        except Exception as error:
            self._add_error(f"Unexpected error during validation: {error}")

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_root_attributes(self, file_handle: h5py.File) -> None:
        required = [
            schema.ATTR_CONTAINER_ID,
            schema.ATTR_CONTAINER_TYPE,
            schema.ATTR_SCHEMA_VERSION,
            schema.ATTR_CREATION_TIMESTAMP,
            schema.ATTR_DISTANCE_CM,
        ]
        for attr_name in required:
            if attr_name not in file_handle.attrs:
                self._add_error(f"Missing required root attribute: {attr_name}")

        if schema.ATTR_CONTAINER_TYPE in file_handle.attrs:
            container_type = file_handle.attrs[schema.ATTR_CONTAINER_TYPE]
            if container_type != schema.CONTAINER_TYPE_TECHNICAL:
                self._add_error(
                    f"Invalid container_type: {container_type}, "
                    f"expected: {schema.CONTAINER_TYPE_TECHNICAL}"
                )

        if schema.ATTR_SCHEMA_VERSION in file_handle.attrs:
            version = file_handle.attrs[schema.ATTR_SCHEMA_VERSION]
            if version != schema.SCHEMA_VERSION:
                self._add_warning(
                    f"Schema version mismatch: {version}, expected: {schema.SCHEMA_VERSION}"
                )

        if schema.ATTR_CONTAINER_ID in file_handle.attrs:
            container_id = file_handle.attrs[schema.ATTR_CONTAINER_ID]
            if not schema.validate_container_id(container_id):
                self._add_error(f"Invalid container_id format: {container_id}")

    def _validate_technical_group(self, file_handle: h5py.File) -> None:
        if schema.GROUP_TECHNICAL not in file_handle:
            self._add_error(f"Missing required group: {schema.GROUP_TECHNICAL}")

    def _validate_config_group(self, file_handle: h5py.File) -> None:
        config_path = schema.GROUP_TECHNICAL_CONFIG
        if config_path not in file_handle:
            self._add_error(f"Missing required group: {config_path}")
            return

        detector_config_path = f"{config_path}/detector_config"
        if detector_config_path not in file_handle:
            self._add_error(f"Missing detector_config dataset: {detector_config_path}")

    def _validate_poni_group(self, file_handle: h5py.File) -> None:
        poni_path = schema.GROUP_TECHNICAL_PONI
        if poni_path not in file_handle:
            self._add_warning(f"Missing PONI group: {poni_path} (optional but recommended)")
            return

        poni_group = file_handle[poni_path]
        has_poni_dataset = any(key.startswith("poni_") for key in poni_group.keys())
        if not has_poni_dataset:
            self._add_warning("PONI group exists but contains no poni_* datasets")

    def _validate_technical_events(self, file_handle: h5py.File) -> None:
        technical_group = file_handle.get(schema.GROUP_TECHNICAL)
        if technical_group is None:
            return

        event_ids = [key for key in technical_group.keys() if key.startswith("tech_evt_")]
        if not event_ids:
            self._add_error("No technical events found (tech_evt_###)")
            return

        found_types: Set[str] = set()
        for event_id in event_ids:
            event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
            event_group = file_handle[event_path]

            if "type" not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'type' attribute")
                continue

            event_type = event_group.attrs["type"]
            found_types.add(event_type)

            if not schema.validate_technical_type(event_type):
                self._add_error(
                    f"{event_path}: Invalid type '{event_type}', "
                    f"must be one of {schema.ALL_TECHNICAL_TYPES}"
                )

            if "timestamp_utc" not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'timestamp_utc' attribute")
            if schema.ATTR_DISTANCE_CM not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'distance_cm' attribute")

            detector_groups = [key for key in event_group.keys() if key.startswith("det_")]
            if not detector_groups:
                self._add_error(f"{event_path}: No detector subgroups found (det_*)")

            for detector_group_name in detector_groups:
                self._validate_detector_group(
                    file_handle,
                    event_path,
                    detector_group_name,
                    event_type,
                )

        missing_types = set(schema.REQUIRED_TECHNICAL_TYPES) - found_types
        if missing_types:
            self._add_error(
                f"Missing required technical measurement types: {sorted(missing_types)}. "
                f"Required: {schema.REQUIRED_TECHNICAL_TYPES}"
            )

    def _validate_detector_group(
        self,
        file_handle: h5py.File,
        event_path: str,
        detector_group_name: str,
        event_type: str,
    ) -> None:
        detector_path = f"{event_path}/{detector_group_name}"
        detector_group = file_handle[detector_path]

        if schema.DATASET_PROCESSED_SIGNAL not in detector_group:
            self._add_error(f"{detector_path}: Missing 'processed_signal' dataset")

        has_blob_group = "blob" in detector_group
        has_legacy_blobs = any(key.startswith("raw_blob") for key in detector_group.keys())
        if not has_blob_group and not has_legacy_blobs:
            self._add_warning(
                f"{detector_path}: No raw data blobs found (blob/ group or raw_blob_* datasets) - optional"
            )

        required_attrs = [
            schema.ATTR_TECHNICAL_TYPE,
            schema.ATTR_DISTANCE_CM,
            schema.ATTR_TIMESTAMP,
            schema.ATTR_DETECTOR_ID,
        ]
        for attr_name in required_attrs:
            if attr_name not in detector_group.attrs:
                self._add_error(f"{detector_path}: Missing attribute '{attr_name}'")

        if event_type == schema.TECHNICAL_TYPE_AGBH:
            if schema.ATTR_PONI_REF not in detector_group.attrs:
                self._add_warning(
                    f"{detector_path}: AGBH measurement missing 'poni_ref' attribute (recommended)"
                )


def validate_technical_container(
    file_path: str,
    strict: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """Validate a technical HDF5 container."""
    validator = TechnicalContainerValidator(file_path, strict=strict)
    return validator.validate()


def print_validation_report(
    file_path: str,
    is_valid: bool,
    errors: List[str],
    warnings: List[str],
) -> None:
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("Technical Container Validation Report")
    print("=" * 70)
    print(f"File: {file_path}")
    print(f"Status: {'VALID' if is_valid else 'INVALID'}")
    print("=" * 70 + "\n")

    if errors:
        print(f"Errors ({len(errors)}):")
        for index, error in enumerate(errors, 1):
            print(f"  {index}. {error}")
        print()

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for index, warning in enumerate(warnings, 1):
            print(f"  {index}. {warning}")
        print()

    if not errors and not warnings:
        print("No issues found")
    print("=" * 70 + "\n")

