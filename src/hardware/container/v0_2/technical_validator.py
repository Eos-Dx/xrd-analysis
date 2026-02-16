"""Technical NeXus container validator for v0.2."""

import logging
from pathlib import Path
from typing import List, Set, Tuple

import h5py

from . import schema

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when strict validation fails."""


class TechnicalContainerValidator:
    """Validator for technical NeXus/HDF5 containers."""

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
                self._validate_root(file_handle)
                self._validate_entry(file_handle)
                self._validate_technical_config(file_handle)
                self._validate_poni(file_handle)
                self._validate_calibration_events(file_handle)
        except Exception as error:
            self._add_error(f"Unexpected error during validation: {error}")

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_root(self, file_handle: h5py.File) -> None:
        required = [
            schema.ATTR_CONTAINER_ID,
            schema.ATTR_CONTAINER_TYPE,
            schema.ATTR_SCHEMA_VERSION,
            schema.ATTR_CREATION_TIMESTAMP,
            schema.ATTR_DISTANCE_CM,
            schema.ATTR_NX_CLASS,
        ]
        for attr_name in required:
            if attr_name not in file_handle.attrs:
                self._add_error(f"Missing required root attribute: {attr_name}")

        if file_handle.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_ROOT:
            self._add_error("Root NX_class must be NXroot")

        if file_handle.attrs.get(schema.ATTR_CONTAINER_TYPE) != schema.CONTAINER_TYPE_TECHNICAL:
            self._add_error("container_type must be technical")

        if str(file_handle.attrs.get(schema.ATTR_SCHEMA_VERSION, "")) != schema.SCHEMA_VERSION:
            self._add_error(
                f"schema_version must be {schema.SCHEMA_VERSION}, got {file_handle.attrs.get(schema.ATTR_SCHEMA_VERSION)}"
            )
        for producer_attr in (
            schema.ATTR_PRODUCER_SOFTWARE,
            schema.ATTR_PRODUCER_VERSION,
        ):
            if producer_attr not in file_handle.attrs:
                self._add_warning(f"Missing producer metadata: {producer_attr}")

    def _validate_entry(self, file_handle: h5py.File) -> None:
        if schema.GROUP_ENTRY not in file_handle:
            self._add_error(f"Missing required group: {schema.GROUP_ENTRY}")
            return

        entry = file_handle[schema.GROUP_ENTRY]
        if entry.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_ENTRY:
            self._add_error("/entry NX_class must be NXentry")

        def_path = f"{schema.GROUP_ENTRY}/{schema.ATTR_ENTRY_DEFINITION}"
        if def_path not in file_handle:
            self._add_error(f"Missing required dataset: {def_path}")
        else:
            value = file_handle[def_path][()]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if value != schema.APPDEF_TECHNICAL:
                self._add_error(
                    f"/entry/definition must be {schema.APPDEF_TECHNICAL}, got {value}"
                )

    def _validate_technical_config(self, file_handle: h5py.File) -> None:
        if schema.GROUP_TECHNICAL_CONFIG not in file_handle:
            self._add_error(f"Missing required group: {schema.GROUP_TECHNICAL_CONFIG}")
            return

        config_group = file_handle[schema.GROUP_TECHNICAL_CONFIG]
        if config_group.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_INSTRUMENT:
            self._add_error(
                f"{schema.GROUP_TECHNICAL_CONFIG} NX_class must be NXinstrument"
            )

        detector_config_path = f"{schema.GROUP_TECHNICAL_CONFIG}/detector_config"
        if detector_config_path not in file_handle:
            self._add_error(f"Missing detector_config dataset: {detector_config_path}")

    def _validate_poni(self, file_handle: h5py.File) -> None:
        if schema.GROUP_TECHNICAL_PONI not in file_handle:
            self._add_error(f"Missing required group: {schema.GROUP_TECHNICAL_PONI}")
            return

        poni_group = file_handle[schema.GROUP_TECHNICAL_PONI]
        has_poni_dataset = any(key.startswith("poni_") for key in poni_group.keys())
        if not has_poni_dataset:
            self._add_warning("No poni_* datasets found")

    def _get_poni_distance(self, file_handle: h5py.File, detector_suffix: str):
        dataset_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{detector_suffix}"
        if dataset_path not in file_handle:
            return None
        return file_handle[dataset_path].attrs.get(schema.ATTR_DISTANCE_CM)

    def _validate_calibration_events(self, file_handle: h5py.File) -> None:
        if schema.GROUP_TECHNICAL not in file_handle:
            self._add_error(f"Missing required group: {schema.GROUP_TECHNICAL}")
            return

        events_group = file_handle[schema.GROUP_TECHNICAL]
        event_ids = [key for key in events_group.keys() if key.startswith("tech_evt_")]
        if not event_ids:
            self._add_error("No technical events found (tech_evt_*)")
            return

        found_types: Set[str] = set()
        for event_id in event_ids:
            event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
            event_group = file_handle[event_path]

            if event_group.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_PROCESS:
                self._add_error(f"{event_path} NX_class must be NXprocess")

            if "type" not in event_group.attrs:
                self._add_error(f"{event_path}: Missing 'type' attribute")
                continue

            event_type = event_group.attrs["type"]
            if isinstance(event_type, bytes):
                event_type = event_type.decode("utf-8", errors="replace")
            found_types.add(event_type)

            if not schema.validate_technical_type(event_type):
                self._add_error(f"{event_path}: Invalid type '{event_type}'")

            if schema.ATTR_TIMESTAMP not in event_group.attrs:
                self._add_error(f"{event_path}: Missing '{schema.ATTR_TIMESTAMP}' attribute")
            if schema.ATTR_DISTANCE_CM not in event_group.attrs:
                self._add_error(f"{event_path}: Missing '{schema.ATTR_DISTANCE_CM}' attribute")

            detector_groups = [key for key in event_group.keys() if key.startswith("det_")]
            if not detector_groups:
                self._add_error(f"{event_path}: No detector subgroups found")

            for detector_group_name in detector_groups:
                detector_path = f"{event_path}/{detector_group_name}"
                detector_group = file_handle[detector_path]

                if detector_group.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_DETECTOR:
                    self._add_error(f"{detector_path} NX_class must be NXdetector")

                if schema.DATASET_PROCESSED_SIGNAL not in detector_group:
                    self._add_error(f"{detector_path}: Missing '{schema.DATASET_PROCESSED_SIGNAL}' dataset")

                for attr_name in [
                    schema.ATTR_TECHNICAL_TYPE,
                    schema.ATTR_DISTANCE_CM,
                    schema.ATTR_TIMESTAMP,
                    schema.ATTR_DETECTOR_ID,
                ]:
                    if attr_name not in detector_group.attrs:
                        self._add_error(f"{detector_path}: Missing attribute '{attr_name}'")

                if event_type == schema.TECHNICAL_TYPE_AGBH and "poni_ref" not in detector_group.attrs and "poni_path" not in detector_group.attrs:
                    self._add_warning(f"{detector_path}: AGBH measurement missing PONI link")

                # Check detector distance consistency with corresponding PONI dataset.
                detector_distance = detector_group.attrs.get(schema.ATTR_DISTANCE_CM)
                detector_suffix = detector_group_name.replace("det_", "")
                poni_distance = self._get_poni_distance(file_handle, detector_suffix)
                if poni_distance is not None and detector_distance is not None:
                    if abs(float(detector_distance) - float(poni_distance)) > 0.1:
                        self._add_error(
                            f"{detector_path}: distance_cm={detector_distance} does not match "
                            f"poni_{detector_suffix} distance_cm={poni_distance}"
                        )

        missing_types = set(schema.REQUIRED_TECHNICAL_TYPES) - found_types
        if missing_types:
            self._add_error(
                f"Missing required technical measurement types: {sorted(missing_types)}"
            )


def validate_technical_container(file_path: str, strict: bool = False) -> Tuple[bool, List[str], List[str]]:
    validator = TechnicalContainerValidator(file_path, strict=strict)
    return validator.validate()


def print_validation_report(file_path: str, is_valid: bool, errors: List[str], warnings: List[str]) -> None:
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
