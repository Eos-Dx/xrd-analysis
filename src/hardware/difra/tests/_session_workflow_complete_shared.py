"""Comprehensive Step-by-Step Session Container Workflow Tests.

Tests the complete end-to-end workflow from session creation to finalization,
including error handling, archiving, and replacement scenarios.

Test Structure:
1. Setup: Create technical container with calibration data
2. Session Creation: Create new session with sample image
3. Zone Definition: Add measurement zones (sample holder, include, exclude)
4. Point Generation: Add measurement points in zones
5. Measurements: Record detector measurements at each point
6. Attenuation: Record I₀ and I measurements
7. Validation: Verify HDF5 structure and data integrity
8. Error Handling: Test session replacement with error marking
9. Archiving: Test session archiving workflow
10. Finalization: Lock container for upload
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project root to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.container.v0_1 import (
    container_manager,
    schema,
    writer as session_container,
    technical_container,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_config():
    """Demo configuration with two detectors (SAXS, WAXS)."""
    return {
        "detectors": [
            {
                "id": "det_saxs",
                "alias": "SAXS",
                "name": "SAXS Detector",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [75.0, 75.0],
            },
            {
                "id": "det_waxs",
                "alias": "WAXS",
                "name": "WAXS Detector",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [55.0, 55.0],
            },
        ],
        "active_detectors": ["det_saxs", "det_waxs"],
    }


@pytest.fixture
def demo_poni_files(temp_dir):
    """Create demo PONI files for both detectors."""
    poni_dir = temp_dir / "poni"
    poni_dir.mkdir()

    poni_template = """# PONI calibration file
poni_version: 2.1
Detector: Detector
Distance: {distance}
Poni1: 0.012345
Poni2: 0.023456
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
PixelSize1: {pixel_size}
PixelSize2: {pixel_size}
"""

    poni_files = {}
    distances = {"SAXS": 1.00, "WAXS": 0.17}  # meters
    pixel_sizes = {"SAXS": 7.5e-05, "WAXS": 5.5e-05}

    for alias, distance in distances.items():
        poni_path = poni_dir / f"{alias.lower()}_demo.poni"
        content = poni_template.format(
            distance=distance, pixel_size=pixel_sizes[alias]
        )
        poni_path.write_text(content)
        poni_files[alias] = poni_path

    return poni_files


@pytest.fixture
def technical_container_path(temp_dir, demo_config, demo_poni_files):
    """Create technical container with all required calibration measurements."""
    print("\n=== STEP 0: Creating Technical Container ===")

    # Create technical measurements for all required types
    tech_measurements = {}
    for tech_type in ["DARK", "EMPTY", "BACKGROUND", "AGBH"]:
        tech_measurements[tech_type] = {}
        for det_config in demo_config["detectors"]:
            alias = det_config["alias"]
            size = det_config["size"]

            # Create synthetic data
            if tech_type == "DARK":
                data = np.random.poisson(5, size=(size["height"], size["width"]))
            elif tech_type == "EMPTY":
                data = np.random.poisson(10, size=(size["height"], size["width"]))
            elif tech_type == "BACKGROUND":
                data = np.random.poisson(20, size=(size["height"], size["width"]))
            else:  # AGBH
                # Create ring pattern for AgBH
                y, x = np.ogrid[: size["height"], : size["width"]]
                center_y, center_x = size["height"] // 2, size["width"] // 2
                r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # Add rings at specific radii
                rings = (
                    (np.abs(r - 30) < 2).astype(float) * 100
                    + (np.abs(r - 60) < 2).astype(float) * 80
                    + (np.abs(r - 90) < 2).astype(float) * 60
                )
                data = np.random.poisson(50 + rings)

            # Save as .npy file
            filename = temp_dir / f"{tech_type}_{alias}.npy"
            np.save(filename, data.astype(np.float32))
            tech_measurements[tech_type][alias] = str(filename)

    # Load PONI data
    poni_data = {}
    for alias, poni_path in demo_poni_files.items():
        content = poni_path.read_text()
        poni_data[alias] = (content, poni_path.name)

    # Generate technical container
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=temp_dir,
        aux_measurements=tech_measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["active_detectors"],
        distances_cm={"SAXS": 100.0, "WAXS": 17.0},
    )

    print(f"  ✓ Technical container created: {Path(file_path).name}")
    print(f"  ✓ Container ID: {container_id}")
    print(f"  ✓ Calibration types: {', '.join(tech_measurements.keys())}")

    # Lock the technical container
    container_manager.lock_technical_container(Path(file_path), locked_by="test_operator", notes="Test technical container")
    print(f"  ✓ Container locked and ready for use")

    return file_path


# ============================================================================
# Test 1: Session Creation
# ============================================================================


