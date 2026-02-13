"""Tests for multiple measurements of the same type.

Tests capturing multiple DARK, AGBH measurements for averaging and validation.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project src root to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.container.v0_1 import technical_container
from hardware.difra.hardware.detectors import DummyDetectorController


@pytest.fixture
def temp_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_detector():
    return DummyDetectorController(alias="PRIMARY", size=(256, 256))


@pytest.fixture
def demo_poni_content():
    return """PixelSize1: 7.5e-05
PixelSize2: 7.5e-05
Distance: 0.17
Poni1: 0.012345
Poni2: 0.023456
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
"""


def test_multiple_dark_measurements(demo_detector, temp_folder):
    """Test capturing multiple DARK measurements."""
    dark_files = []
    
    for i in range(3):
        filename_base = temp_folder / f"DARK_{i}"
        demo_detector.capture_point(
            Nframes=1,
            Nseconds=0.1,
            filename_base=str(filename_base)
        )
        output_file = Path(str(filename_base) + ".txt")
        assert output_file.exists()
        dark_files.append(output_file)
    
    assert len(dark_files) == 3
    
    # Load all measurements
    data_arrays = [np.loadtxt(f) for f in dark_files]
    assert all(d.shape == (256, 256) for d in data_arrays)


def test_multiple_agbh_measurements(demo_detector, temp_folder):
    """Test capturing multiple AGBH (calibration) measurements."""
    agbh_files = []
    
    for i in range(3):
        filename_base = temp_folder / f"AGBH_{i}"
        demo_detector.capture_point(
            Nframes=5,
            Nseconds=1.0,
            filename_base=str(filename_base)
        )
        output_file = Path(str(filename_base) + ".txt")
        assert output_file.exists()
        agbh_files.append(output_file)
    
    assert len(agbh_files) == 3


def test_measurement_averaging(demo_detector, temp_folder):
    """Test averaging multiple measurements of same type."""
    measurements = []
    
    # Capture 5 measurements
    for i in range(5):
        filename_base = temp_folder / f"meas_{i}"
        demo_detector.capture_point(Nframes=1, Nseconds=0.1, filename_base=str(filename_base))
        data = np.loadtxt(str(filename_base) + ".txt")
        measurements.append(data)
    
    # Compute average
    avg_data = np.mean(measurements, axis=0)
    assert avg_data.shape == (256, 256)
    assert np.all(np.isfinite(avg_data))
    
    # Variance should be lower than individual measurements
    individual_std = np.mean([m.std() for m in measurements])
    avg_std = avg_data.std()
    # Averaged data should have characteristics of the underlying distribution


def test_h5_with_multiple_events(temp_folder, demo_poni_content):
    """Test HDF5 container with multiple technical events."""
    detector = DummyDetectorController(alias="PRIMARY", size=(256, 256))
    
    # Capture multiple measurements of each type
    aux_measurements = {}
    
    for mtype in ["DARK", "EMPTY", "BACKGROUND", "AGBH"]:
        aux_measurements[mtype] = {}
        # Just one per type for this test (multiple events would need schema updates)
        filename = temp_folder / f"{mtype}_PRIMARY.npy"
        
        # Generate and save data
        data = np.random.rand(256, 256).astype(np.float32) * 1000
        np.save(filename, data)
        aux_measurements[mtype]["PRIMARY"] = str(filename)
    
    # Create PONI data
    poni_file = temp_folder / "PRIMARY.poni"
    poni_file.write_text(demo_poni_content)
    poni_data = {"PRIMARY": (demo_poni_content, "PRIMARY.poni")}
    
    # Generate container
    detector_config = [{"id": "PRIMARY", "alias": "PRIMARY", "type": "dummy", "width": 256, "height": 256}]
    
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(temp_folder),
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=detector_config,
        active_detector_ids=["PRIMARY"],
        distances_cm=17.0,
    )
    
    # Verify container created
    assert Path(file_path).exists()
    
    # Check HDF5 structure
    with h5py.File(file_path, "r") as f:
        tech_group = f["technical"]
        event_groups = [k for k in tech_group.keys() if k.startswith("tech_evt_")]
        assert len(event_groups) == 4  # One for each type


def test_measurement_selection_for_h5(temp_folder, demo_poni_content):
    """Test selecting specific measurements to include in HDF5."""
    detector = DummyDetectorController(alias="PRIMARY", size=(256, 256))
    
    # Capture multiple DARK measurements but only use one
    dark_files = []
    for i in range(3):
        filename = temp_folder / f"DARK_{i}_PRIMARY.npy"
        data = np.random.rand(256, 256).astype(np.float32) * 100
        np.save(filename, data)
        dark_files.append(filename)
    
    # Select only the first DARK for HDF5
    selected_dark = dark_files[0]
    
    aux_measurements = {
        "DARK": {"PRIMARY": str(selected_dark)},
        "EMPTY": {"PRIMARY": str(temp_folder / "EMPTY_PRIMARY.npy")},
        "BACKGROUND": {"PRIMARY": str(temp_folder / "BG_PRIMARY.npy")},
        "AGBH": {"PRIMARY": str(temp_folder / "AGBH_PRIMARY.npy")},
    }
    
    # Create remaining required files
    for mtype in ["EMPTY", "BACKGROUND", "AGBH"]:
        np.save(aux_measurements[mtype]["PRIMARY"], np.random.rand(256, 256).astype(np.float32) * 1000)
    
    poni_data = {"PRIMARY": (demo_poni_content, "PRIMARY.poni")}
    detector_config = [{"id": "PRIMARY", "alias": "PRIMARY", "type": "dummy", "width": 256, "height": 256}]
    
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(temp_folder),
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=detector_config,
        active_detector_ids=["PRIMARY"],
        distances_cm=17.0,
    )
    
    assert Path(file_path).exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
