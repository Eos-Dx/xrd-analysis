"""Comprehensive workflow tests for technical measurements.

Tests Phase 2, 3, 4 features:
- Auto-validation workflow
- Container locking with operator tracking
- Read-only file permissions
- Validation result handling
- Error cases (missing PONIs, invalid distances, failed validation)
- Session workflow enforcement
"""

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

# Add the project src root to the path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.container.v0_1 import (
    container_manager,
    schema,
    technical_container,
    technical_validator,
)
from hardware.difra.hardware.detectors import DummyDetectorController


# ==================== Fixtures ====================

@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_config():
    """Configuration for DEMO mode."""
    return {
        "DEV": True,
        "detectors": [
            {
                "id": "PRIMARY",
                "alias": "PRIMARY",
                "name": "SAXS Demo",
                "type": "dummy",
                "width": 256,
                "height": 256,
            },
            {
                "id": "SECONDARY",
                "alias": "SECONDARY",
                "name": "WAXS Demo",
                "type": "dummy",
                "width": 256,
                "height": 256,
            },
        ],
        "dev_active_detectors": ["PRIMARY", "SECONDARY"],
        "active_detectors": ["PRIMARY", "SECONDARY"],
    }


@pytest.fixture
def valid_poni_files(temp_dir):
    """Create valid PONI files for testing."""
    poni_dir = temp_dir / "poni"
    poni_dir.mkdir()

    poni_template = """# Detector: {detector}
# Pixel1: 7.500e-05
# Pixel2: 7.500e-05
PixelSize1: 7.500000e-05
PixelSize2: 7.500000e-05
Distance: {distance}
Poni1: 0.012345
Poni2: 0.023456
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
"""

    poni_files = {}
    for detector, distance in [("PRIMARY", 1.00), ("SECONDARY", 0.17)]:
        poni_path = poni_dir / f"{detector.lower()}.poni"
        poni_path.write_text(poni_template.format(detector=detector, distance=distance))
        poni_files[detector] = poni_path

    return poni_files


@pytest.fixture
def sample_measurements(temp_dir, demo_config):
    """Generate sample measurement data."""
    measurements = {}
    required_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH"]

    for meas_type in required_types:
        measurements[meas_type] = {}
        for detector_id in demo_config["dev_active_detectors"]:
            # Generate synthetic data
            data = np.random.poisson(50, size=(256, 256)).astype(np.float32)
            filename = temp_dir / f"{meas_type}_{detector_id}.npy"
            np.save(filename, data)
            measurements[meas_type][detector_id] = str(filename)

    return measurements


def create_valid_container(temp_dir, valid_poni_files, sample_measurements, demo_config):
    """Helper to create a valid technical container."""
    poni_data = {}
    for detector_id, poni_path in valid_poni_files.items():
        content = poni_path.read_text()
        poni_data[detector_id] = (content, poni_path.name)

    distances_cm = {"PRIMARY": 100.0, "SECONDARY": 17.0}
    poni_distances_cm = {"PRIMARY": 100.0, "SECONDARY": 17.0}

    container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(temp_dir),
        aux_measurements=sample_measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["dev_active_detectors"],
        distances_cm=distances_cm,
        poni_distances_cm=poni_distances_cm,
    )

    return Path(file_path)


# ==================== Phase 2: Auto-Validation Tests ====================

