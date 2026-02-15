"""Tests for analytical measurements (attenuation) in DIFRA HDF5 containers.

Tests verify:
- Adding analytical measurements with analysis_type="attenuation"
- Linking analytical measurements to points
- Per-detector metadata (integration time, beam energy)
- PONI references from analytical measurements
- Multiple analytical measurements per session
- Validation of analytical measurement structure
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project src to path
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.container.v0_1 import schema, validator, writer


@pytest.fixture
def session_container_with_technical(tmp_path):
    """Create session container with technical data copied."""
    from hardware.container.v0_1.technical_container import (
        create_technical_container,
        write_detector_config,
        write_poni_datasets,
    )
    temp_dir = Path(tmp_path)
    
    # Create minimal technical container
    tech_id, tech_path = create_technical_container(
        folder=temp_dir,
        distance_cm=17.0,
    )
    
    # Add detector config
    detector_config = [
        {
            "id": "DET1",
            "alias": "DET1",
            "type": "AdvaPIX",
            "size": [256, 256],
            "pixel_size_um": 55.0,
        },
        {
            "id": "DET2",
            "alias": "DET2",
            "type": "AdvaPIX",
            "size": [256, 256],
            "pixel_size_um": 55.0,
        },
    ]
    
    write_detector_config(tech_path, detector_config, ["DET1", "DET2"])
    
    # Add PONI data
    poni_content = """Detector: AdvaPIX
PixelSize1: 5.500e-05
PixelSize2: 5.500e-05
Distance: 0.170000
Poni1: 0.014025
Poni2: 0.014025
Rot1: 0.000000
Rot2: 0.000000
Rot3: 0.000000
Detector_config: {"pixel1": 5.5e-05, "pixel2": 5.5e-05, "max_shape": [256, 256]}
"""
    
    poni_data = {
        "DET1": (poni_content, "DET1_17cm.poni"),
        "DET2": (poni_content, "DET2_17cm.poni"),
    }
    
    write_poni_datasets(tech_path, poni_data, 17.0)
    
    # Lock technical container
    from hardware.container.v0_1.container_manager import lock_container
    lock_container(tech_path)
    
    # Create session container
    session_id, session_path = writer.create_session_container(
        folder=temp_dir,
        sample_id="SAMPLE_001",
        operator_id="test_operator",
        site_id="test_site",
        machine_name="DIFRA-01",
        beam_energy_keV=17.5,
        acquisition_date="2024-01-15",
    )
    
    # Copy technical data to session
    writer.copy_technical_to_session(
        technical_file=tech_path,
        session_file=session_path,
        auto_lock=False,  # Already locked
    )
    
    # Add a test point
    writer.add_point(
        file_path=session_path,
        point_index=1,
        pixel_coordinates=[100, 200],
        physical_coordinates_mm=[10.0, 20.0],
    )
    
    yield session_path

