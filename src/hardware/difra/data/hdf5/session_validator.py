"""Session Container Validator

Validates session HDF5 containers against the DIFRA_HDF5_Data_Model_FINAL specification.
Provides detailed error reporting and compliance checking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from hardware.difra.data.hdf5 import schema_v1

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a single validation error."""

    def __init__(self, severity: str, path: str, message: str):
        """Initialize validation error.

        Args:
            severity: "ERROR", "WARNING", or "INFO"
            path: HDF5 path where error occurred
            message: Descriptive error message
        """
        self.severity = severity
        self.path = path
        self.message = message

    def __repr__(self) -> str:
        return f"{self.severity} [{self.path}]: {self.message}"


class SessionContainerValidator:
    """Validates session HDF5 containers."""

    def __init__(self, session_file: Union[str, Path]):
        """Initialize validator.

        Args:
            session_file: Path to session container
        """
        self.session_file = Path(session_file)
        self.errors: List[ValidationError] = []

    def validate(self) -> Tuple[bool, List[ValidationError]]:
        """Run all validation checks.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.errors = []

        try:
            with h5py.File(self.session_file, "r") as f:
                self._validate_root_attributes(f)
                self._validate_technical_group(f)
                self._validate_images_group(f)
                self._validate_points_group(f)
                self._validate_measurements_group(f)
                self._validate_analytical_measurements_group(f)
                self._validate_references(f)
        except Exception as e:
            self.errors.append(
                ValidationError("ERROR", "/", f"Failed to open container: {e}")
            )

        has_errors = any(e.severity == "ERROR" for e in self.errors)
        return not has_errors, self.errors

    def _validate_root_attributes(self, f: h5py.File) -> None:
        """Validate root-level attributes."""
        required_attrs = [
            schema_v1.ATTR_CONTAINER_ID,
            schema_v1.ATTR_CONTAINER_TYPE,
            schema_v1.ATTR_SCHEMA_VERSION,
            schema_v1.ATTR_SAMPLE_ID,
            schema_v1.ATTR_SESSION_ID,
            schema_v1.ATTR_CREATION_TIMESTAMP,
            schema_v1.ATTR_ACQUISITION_DATE,
            schema_v1.ATTR_OPERATOR_ID,
            schema_v1.ATTR_SITE_ID,
            schema_v1.ATTR_MACHINE_NAME,
            schema_v1.ATTR_BEAM_ENERGY_KEV,
        ]

        for attr in required_attrs:
            if attr not in f.attrs:
                self.errors.append(
                    ValidationError("ERROR", "/", f"Missing required attribute: {attr}")
                )

        # Check container type
        if f.attrs.get(schema_v1.ATTR_CONTAINER_TYPE) != schema_v1.CONTAINER_TYPE_SESSION:
            self.errors.append(
                ValidationError(
                    "ERROR",
                    "/",
                    f"Invalid container type: expected {schema_v1.CONTAINER_TYPE_SESSION}",
                )
            )

        # Check measurement counter exists
        if "measurement_counter" not in f.attrs:
            self.errors.append(
                ValidationError("WARNING", "/", "Missing measurement_counter attribute")
            )

    def _validate_technical_group(self, f: h5py.File) -> None:
        """Validate /technical group."""
        if schema_v1.GROUP_TECHNICAL not in f:
            self.errors.append(
                ValidationError(
                    "ERROR",
                    schema_v1.GROUP_TECHNICAL,
                    "Technical group not found",
                )
            )
            return

        tech_group = f[schema_v1.GROUP_TECHNICAL]

        # Check for required subgroups
        required_subgroups = [schema_v1.GROUP_TECHNICAL_CONFIG, schema_v1.GROUP_TECHNICAL_PONY]
        for subgroup in required_subgroups:
            if subgroup not in f:
                self.errors.append(
                    ValidationError(
                        "ERROR",
                        schema_v1.GROUP_TECHNICAL,
                        f"Missing required subgroup: {subgroup}",
                    )
                )

        # Check for detector config
        config_path = f"{schema_v1.GROUP_TECHNICAL_CONFIG}/detector_config"
        if config_path not in f:
            self.errors.append(
                ValidationError(
                    "WARNING",
                    schema_v1.GROUP_TECHNICAL_CONFIG,
                    "Missing detector_config dataset",
                )
            )

        # Check for PONY datasets
        pony_group = f[schema_v1.GROUP_TECHNICAL_PONY]
        if len(pony_group) == 0:
            self.errors.append(
                ValidationError(
                    "WARNING",
                    schema_v1.GROUP_TECHNICAL_PONY,
                    "No PONY datasets found",
                )
            )

    def _validate_images_group(self, f: h5py.File) -> None:
        """Validate /images group."""
        if schema_v1.GROUP_IMAGES not in f:
            # Images are optional, just log
            return

        images_group = f[schema_v1.GROUP_IMAGES]

        # Check for zones subgroup
        if schema_v1.GROUP_IMAGES_ZONES not in f:
            self.errors.append(
                ValidationError(
                    "WARNING",
                    schema_v1.GROUP_IMAGES,
                    "Missing zones subgroup",
                )
            )

        # Check for mapping subgroup
        if schema_v1.GROUP_IMAGES_MAPPING not in f:
            self.errors.append(
                ValidationError(
                    "INFO",
                    schema_v1.GROUP_IMAGES,
                    "Missing mapping subgroup",
                )
            )

    def _validate_points_group(self, f: h5py.File) -> None:
        """Validate /points group."""
        if schema_v1.GROUP_POINTS not in f:
            return  # Points are optional

        points_group = f[schema_v1.GROUP_POINTS]

        for point_id in points_group:
            point_path = f"{schema_v1.GROUP_POINTS}/{point_id}"
            point = points_group[point_id]

            # Check required attributes
            required_attrs = [
                schema_v1.ATTR_PIXEL_COORDINATES,
                schema_v1.ATTR_PHYSICAL_COORDINATES_MM,
                schema_v1.ATTR_POINT_STATUS,
            ]

            for attr in required_attrs:
                if attr not in point.attrs:
                    self.errors.append(
                        ValidationError(
                            "WARNING",
                            point_path,
                            f"Missing attribute: {attr}",
                        )
                    )

    def _validate_measurements_group(self, f: h5py.File) -> None:
        """Validate /measurements group."""
        if schema_v1.GROUP_MEASUREMENTS not in f:
            return  # Measurements are optional

        meas_group = f[schema_v1.GROUP_MEASUREMENTS]

        for point_id in meas_group:
            point_path = f"{schema_v1.GROUP_MEASUREMENTS}/{point_id}"
            point_group = meas_group[point_id]

            for meas_id in point_group:
                meas_path = f"{point_path}/{meas_id}"
                meas = point_group[meas_id]

                # Check measurement attributes
                required_attrs = [
                    schema_v1.ATTR_MEASUREMENT_COUNTER,
                    schema_v1.ATTR_TIMESTAMP_START,
                    schema_v1.ATTR_MEASUREMENT_STATUS,
                ]

                for attr in required_attrs:
                    if attr not in meas.attrs:
                        self.errors.append(
                            ValidationError(
                                "WARNING",
                                meas_path,
                                f"Missing attribute: {attr}",
                            )
                        )

                # Check for detector data
                for detector_id in meas:
                    if detector_id.startswith("det_"):
                        det_path = f"{meas_path}/{detector_id}"
                        detector = meas[detector_id]

                        # Check for processed_signal dataset
                        if schema_v1.DATASET_PROCESSED_SIGNAL not in detector:
                            self.errors.append(
                                ValidationError(
                                    "ERROR",
                                    det_path,
                                    "Missing processed_signal dataset",
                                )
                            )
                        else:
                            # Validate processed_signal is 2D array
                            processed_signal = detector[schema_v1.DATASET_PROCESSED_SIGNAL]
                            if len(processed_signal.shape) != 2:
                                self.errors.append(
                                    ValidationError(
                                        "ERROR",
                                        f"{det_path}/{schema_v1.DATASET_PROCESSED_SIGNAL}",
                                        f"Expected 2D array, got shape {processed_signal.shape}",
                                    )
                                )

    def _validate_analytical_measurements_group(self, f: h5py.File) -> None:
        """Validate /analytical_measurements group."""
        if schema_v1.GROUP_ANALYTICAL_MEASUREMENTS not in f:
            return  # Analytical measurements are optional

        ana_group = f[schema_v1.GROUP_ANALYTICAL_MEASUREMENTS]

        for ana_id in ana_group:
            ana_path = f"{schema_v1.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"
            ana = ana_group[ana_id]

            # Check required attributes
            required_attrs = [
                schema_v1.ATTR_MEASUREMENT_COUNTER,
                schema_v1.ATTR_TIMESTAMP_START,
                schema_v1.ATTR_ANALYSIS_TYPE,
            ]

            for attr in required_attrs:
                if attr not in ana.attrs:
                    self.errors.append(
                        ValidationError(
                            "WARNING",
                            ana_path,
                            f"Missing attribute: {attr}",
                        )
                    )

    def _validate_references(self, f: h5py.File) -> None:
        """Validate HDF5 references are intact."""
        # Check point references in measurements
        if schema_v1.GROUP_MEASUREMENTS in f:
            meas_group = f[schema_v1.GROUP_MEASUREMENTS]
            for point_id in meas_group:
                point_group = meas_group[point_id]
                for meas_id in point_group:
                    meas = point_group[meas_id]

                    if schema_v1.ATTR_POINT_REF in meas.attrs:
                        try:
                            ref = meas.attrs[schema_v1.ATTR_POINT_REF]
                            _ = f[ref]  # Try to dereference
                        except Exception as e:
                            self.errors.append(
                                ValidationError(
                                    "WARNING",
                                    f"{schema_v1.GROUP_MEASUREMENTS}/{point_id}/{meas_id}",
                                    f"Invalid point reference: {e}",
                                )
                            )

        # Check analytical measurement references
        if schema_v1.GROUP_POINTS in f:
            points_group = f[schema_v1.GROUP_POINTS]
            for point_id in points_group:
                point = points_group[point_id]

                if schema_v1.ATTR_ANALYTICAL_MEASUREMENT_REFS in point.attrs:
                    try:
                        refs = point.attrs[schema_v1.ATTR_ANALYTICAL_MEASUREMENT_REFS]
                        if refs.size > 0:
                            for ref in refs:
                                _ = f[ref]  # Try to dereference
                    except Exception as e:
                        self.errors.append(
                            ValidationError(
                                "WARNING",
                                f"{schema_v1.GROUP_POINTS}/{point_id}",
                                f"Invalid analytical reference: {e}",
                            )
                        )

    def get_summary(self) -> str:
        """Get validation summary.

        Returns:
            Human-readable summary
        """
        if not self.errors:
            return "✓ Container is valid"

        errors = [e for e in self.errors if e.severity == "ERROR"]
        warnings = [e for e in self.errors if e.severity == "WARNING"]
        infos = [e for e in self.errors if e.severity == "INFO"]

        summary = f"Validation Results:\n"
        summary += f"  Errors: {len(errors)}\n"
        summary += f"  Warnings: {len(warnings)}\n"
        summary += f"  Info: {len(infos)}\n"

        if errors:
            summary += "\nErrors:\n"
            for e in errors:
                summary += f"  {e}\n"

        if warnings:
            summary += "\nWarnings:\n"
            for e in warnings:
                summary += f"  {e}\n"

        return summary


def validate_session_container(session_file: Union[str, Path]) -> Tuple[bool, str]:
    """Validate a session container.

    Args:
        session_file: Path to session container

    Returns:
        Tuple of (is_valid, summary_message)
    """
    validator = SessionContainerValidator(session_file)
    is_valid, errors = validator.validate()
    summary = validator.get_summary()
    return is_valid, summary
