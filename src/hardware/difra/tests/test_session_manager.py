"""Tests for SessionManager GUI integration module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add project src to path
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.difra.gui.session_manager import SessionManager
from hardware.container.v0_1.technical_container import generate_from_aux_table
from hardware.container.v0_1.container_manager import lock_container


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def technical_container(temp_dir):
    """Create and lock a technical container."""
    # Create minimal technical container
    poni_content = """Detector: AdvaPIX
PixelSize1: 5.500e-05
PixelSize2: 5.500e-05
Distance: 0.170000
Poni1: 0.014025
Poni2: 0.014025
"""
    
    # Create dummy measurement files
    dark_file = temp_dir / "dark.npy"
    np.save(dark_file, np.random.rand(256, 256).astype(np.float32))
    
    tech_id, tech_path = generate_from_aux_table(
        folder=temp_dir,
        aux_measurements={"DARK": {"DET1": str(dark_file)}},
        pony_data={"DET1": (poni_content, "DET1_17cm.poni")},
        detector_config=[{
            "id": "DET1",
            "alias": "DET1",
            "type": "AdvaPIX",
            "size": [256, 256],
            "pixel_size_um": 55.0,
        }],
        active_detector_ids=["DET1"],
        distances_cm=17.0,
        validate_poni=True,
    )
    
    # Lock it
    lock_container(tech_path)
    
    return Path(tech_path)


def test_session_manager_create_session(temp_dir, technical_container):
    """Test creating a new session."""
    manager = SessionManager()
    
    # Initially no session
    assert not manager.is_session_active()
    
    # Create session
    session_id, session_path = manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distances_cm=17.0,
        operator_id="test_operator",
    )
    
    # Session is now active
    assert manager.is_session_active()
    assert session_path.exists()
    assert manager.sample_id == "TEST_SAMPLE_001"
    assert manager.session_id == session_id


def test_session_manager_add_points(temp_dir, technical_container):
    """Test adding points to session."""
    manager = SessionManager()
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distances_cm=17.0,
    )
    
    # Add points
    points = [
        {
            "pixel_coordinates": [100, 200],
            "physical_coordinates_mm": [10.0, 20.0],
        },
        {
            "pixel_coordinates": [150, 250],
            "physical_coordinates_mm": [15.0, 25.0],
        },
    ]
    
    paths = manager.add_points(points)
    assert len(paths) == 2


def test_session_manager_attenuation_workflow(temp_dir, technical_container):
    """Test complete attenuation workflow."""
    manager = SessionManager()
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distances_cm=17.0,
    )
    
    # Add points
    points = [
        {
            "pixel_coordinates": [100, 200],
            "physical_coordinates_mm": [10.0, 20.0],
        },
    ]
    manager.add_points(points)
    
    # Add I₀ measurement (without sample)
    i0_data = {"DET1": np.random.randint(800, 1000, (256, 256), dtype=np.uint16)}
    i0_metadata = {"DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5}}
    
    i0_counter = manager.add_attenuation_measurement(
        measurement_data=i0_data,
        detector_metadata=i0_metadata,
        pony_alias_map={"DET1": "DET1"},
        mode="without",
    )
    
    assert i0_counter == 1
    assert manager.i0_counter == 1
    
    # Add I measurement (with sample)
    i_data = {"DET1": np.random.randint(400, 600, (256, 256), dtype=np.uint16)}
    i_metadata = {"DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5}}
    
    i_counter = manager.add_attenuation_measurement(
        measurement_data=i_data,
        detector_metadata=i_metadata,
        pony_alias_map={"DET1": "DET1"},
        mode="with",
    )
    
    assert i_counter == 2
    assert manager.i_counter == 2
    
    # Link to points
    manager.link_attenuation_to_points(num_points=1)
    
    # Get session info
    info = manager.get_session_info()
    assert info["attenuation_complete"] is True


def test_session_manager_add_measurement(temp_dir, technical_container):
    """Test adding regular measurements."""
    manager = SessionManager()
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distances_cm=17.0,
    )
    
    # Add point
    points = [{"pixel_coordinates": [100, 200], "physical_coordinates_mm": [10.0, 20.0]}]
    manager.add_points(points)
    
    # Add measurement at point 1
    meas_data = {"DET1": np.random.randint(0, 100, (256, 256), dtype=np.uint16)}
    meas_metadata = {"DET1": {"integration_time_ms": 1000.0, "beam_energy_keV": 17.5}}
    
    meas_path = manager.add_measurement(
        point_index=1,
        measurement_data=meas_data,
        detector_metadata=meas_metadata,
        pony_alias_map={"DET1": "DET1"},
    )
    
    assert "meas_" in meas_path


def test_session_manager_close_session(temp_dir, technical_container):
    """Test closing session."""
    manager = SessionManager()
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distances_cm=17.0,
    )
    
    assert manager.is_session_active()
    
    manager.close_session()
    
    assert not manager.is_session_active()
    assert manager.session_path is None
    assert manager.sample_id is None


def test_session_manager_requires_active_session(temp_dir):
    """Test that operations require active session."""
    manager = SessionManager()
    
    # Should raise without active session
    with pytest.raises(RuntimeError, match="No active session"):
        manager.add_points([])
    
    with pytest.raises(RuntimeError, match="No active session"):
        manager.add_measurement(1, {}, {}, {})


def test_session_manager_get_session_info(temp_dir, technical_container):
    """Test getting session info."""
    manager = SessionManager()
    
    # No active session
    info = manager.get_session_info()
    assert info["active"] is False
    
    # Create session
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distances_cm=17.0,
    )
    
    info = manager.get_session_info()
    assert info["active"] is True
    assert info["sample_id"] == "TEST_SAMPLE_001"
    assert info["attenuation_complete"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
